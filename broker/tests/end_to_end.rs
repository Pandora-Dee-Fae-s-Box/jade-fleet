//! The whole broker against stand-ins for Ollama (embeddings, decisions), the Xavier (llama-server streaming) and the
//! vision service, with real SQLite + sqlite-vec, real processes and real WebSockets.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Uri};
use axum::response::Response;
use axum::routing::{any, get, post};
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use jade_fleet::config::Config;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_tungstenite::tungstenite::Message as WsMsg;

const DIM: usize = 64;
const TOKEN: &str = "test-token";

// ------------------------------------------------------------- stand-ins

/// Bag-of-words vectors: texts that share words are near each other.
fn embed_text(t: &str) -> Vec<f32> {
    let mut v = vec![0f32; DIM];
    for w in t.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2 && *w != "search_query" && *w != "search_document" && *w != "search" && *w != "query" && *w != "document") {
        let h = w.bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
        v[(h % DIM as u64) as usize] += 1.0;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / n).collect()
}

#[derive(Default)]
struct Seen {
    chat: Vec<Value>,
    headers: Vec<(String, Option<String>, String)>,
    embeds: usize,
}

type Shared = Arc<Mutex<Seen>>;

async fn fake_embed(State(s): State<Shared>, Json(b): Json<Value>) -> Json<Value> {
    let input: Vec<String> = b["input"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    s.lock().unwrap().embeds += input.len();
    Json(json!({"model": b["model"], "embeddings": input.iter().map(|t| embed_text(t)).collect::<Vec<_>>()}))
}

async fn record(State(s): State<Shared>, h: HeaderMap, uri: Uri, body: String) -> Json<Value> {
    s.lock().unwrap().headers.push((uri.to_string(), h.get("authorization").map(|v| v.to_str().unwrap().to_string()), body.clone()));
    Json(json!({"echo_path": uri.path(), "echo_query": uri.query(), "body": body}))
}

async fn fake_chat(State(s): State<Shared>, Json(b): Json<Value>) -> Response {
    s.lock().unwrap().chat.push(b.clone());
    let last = b["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().to_string();
    let slow = last.contains("slow");
    let words: Vec<String> = if slow { (0..40).map(|i| format!("w{i} ")).collect() } else { format!("Echo: {last}").split_inclusive(' ').map(String::from).collect() };
    let stream = async_stream(words, if slow { 50 } else { 2 });
    Response::builder().header("content-type", "text/event-stream").body(Body::from_stream(stream)).unwrap()
}

fn async_stream(words: Vec<String>, ms: u64) -> impl futures_util::Stream<Item = Result<String, std::io::Error>> {
    futures_util::stream::unfold((words.into_iter(), false), move |(mut it, done)| async move {
        if done {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(ms)).await;
        match it.next() {
            Some(w) => Some((Ok(format!("data: {}\n\n", json!({"choices": [{"delta": {"content": w}}]}))), (it, false))),
            None => Some((Ok("data: [DONE]\n\n".to_string()), (it, true))),
        }
    })
}

async fn serve(r: Router) -> String {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, r).await.unwrap() });
    format!("http://{a}")
}

// ------------------------------------------------------------- client helpers

struct T {
    base: String,
    up: jade_fleet::upstream::Upstream,
    seen: Shared,
    vault: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

impl T {
    async fn call(&self, method: &str, path: &str, body: Option<Value>, token: bool) -> (u16, Value) {
        let mut b = hyper::Request::builder().method(method).uri(format!("{}{}", self.base, path)).header("x-jade-device", "phoneA");
        if token {
            b = b.header("authorization", format!("Bearer {TOKEN}"));
        }
        let req = match body {
            Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
            None => b.body(Body::empty()).unwrap(),
        };
        let res = self.up.send(req, Duration::from_secs(30)).await.unwrap();
        let st = res.status().as_u16();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (st, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into())))
    }
    async fn raw(&self, method: &str, path: &str, body: Vec<u8>) -> (u16, Vec<u8>) {
        let req = hyper::Request::builder().method(method).uri(format!("{}{}", self.base, path)).header("authorization", format!("Bearer {TOKEN}")).header("x-jade-device", "phoneA").body(Body::from(body)).unwrap();
        let res = self.up.send(req, Duration::from_secs(30)).await.unwrap();
        let st = res.status().as_u16();
        (st, res.into_body().collect().await.unwrap().to_bytes().to_vec())
    }
    async fn ws(&self, path: &str) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>> {
        let sep = if path.contains('?') { '&' } else { '?' };
        let url = format!("{}{}{}token={TOKEN}", self.base.replace("http://", "ws://"), path, sep);
        tokio_tungstenite::connect_async(url).await.unwrap().0
    }
}

async fn next_json(ws: &mut (impl StreamExt<Item = Result<WsMsg, tokio_tungstenite::tungstenite::Error>> + Unpin)) -> Value {
    loop {
        match tokio::time::timeout(Duration::from_secs(10), ws.next()).await.expect("no message within 10s") {
            Some(Ok(WsMsg::Text(t))) => return serde_json::from_str(&t).unwrap(),
            Some(Ok(_)) => continue,
            other => panic!("socket ended: {other:?}"),
        }
    }
}

async fn start() -> T {
    let seen: Shared = Default::default();
    let ollama = serve(Router::new().route("/api/embed", post(fake_embed)).route("/api/tags", any(record)).route("/v1/systemone", any(record)).with_state(seen.clone())).await;
    let xavier = serve(Router::new().route("/v1/chat/completions", post(fake_chat)).route("/v1/models", get(record)).route("/health", get(record)).with_state(seen.clone())).await;
    let vision = serve(Router::new().route("/v1/vision/{*rest}", any(record)).with_state(seen.clone())).await;
    let ollaya = serve(Router::new().route("/v1/systemone", any(record)).route("/api/tags", any(record)).with_state(seen.clone())).await;
    let dir = tempfile::tempdir().unwrap();
    std::env::remove_var("FLEET_TOKEN");
    let mut cfg = Config::from_env();
    cfg.bind = "127.0.0.1:0".into();
    cfg.data_dir = dir.path().join("data");
    cfg.vault_dir = dir.path().join("vault");
    cfg.web_dir = dir.path().join("web");
    std::fs::create_dir_all(&cfg.web_dir).unwrap();
    std::fs::write(cfg.web_dir.join("index.html"), "<title>Fleet client</title>").unwrap();
    cfg.token = TOKEN.into();
    cfg.ollama_url = ollama.clone();
    cfg.decide_url = ollaya.clone();
    cfg.ollaya_url = ollaya;
    cfg.llm_url = format!("{xavier}/v1");
    cfg.vision_url = vision;
    cfg.embed_model = "test-embed".into();
    cfg.embed_dim = DIM;
    cfg.embed_query_prefix = "search_query: ".into();
    cfg.embed_doc_prefix = "search_document: ".into();
    cfg.chunk_chars = 300;
    cfg.rag_k = 3;
    let vault = cfg.vault_dir.clone();
    let app = jade_fleet::build(cfg).await.unwrap();
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    tokio::spawn(jade_fleet::serve(app, l, std::future::pending()));
    T { base, up: jade_fleet::upstream::Upstream::new(), seen, vault, _dir: dir }
}

// ------------------------------------------------------------- the tests

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fleet_end_to_end() {
    let t = start().await;

    // --- access
    let (st, v) = t.call("GET", "/health", None, false).await;
    assert!(st == 200 && v["ok"] == true, "health {st} {v}");
    let (st, _) = t.call("GET", "/v1/info", None, false).await;
    assert_eq!(st, 401, "the API needs the token");
    let (st, v) = t.call("GET", "/v1/info", None, true).await;
    assert!(st == 200 && v["sqlite_vec"].as_str().unwrap().starts_with("v0.1"), "info: {v}");
    let (st, page) = t.raw("GET", "/", vec![]).await;
    assert!(st == 200 && String::from_utf8_lossy(&page).contains("Fleet client"), "the browser client is served at /");

    // --- the shared vault, with live events to another phone
    let mut ev = t.ws("/v1/events?device=phoneB").await;
    let hello = next_json(&mut ev).await;
    assert_eq!(hello["type"], "hello");
    let (st, _) = t.raw("PUT", "/v1/vault/file?path=/Trips/Lisbon.md", b"# Lisbon\n\nFlights to Lisbon in May. Pasteis de nata in Belem, tram 28 to Alfama.".to_vec()).await;
    assert_eq!(st, 200);
    let changed = loop {
        let e = next_json(&mut ev).await;
        if e["type"] == "changed" && e["paths"].as_array().unwrap().iter().any(|p| p == "/Trips/Lisbon.md") {
            break e;
        }
    };
    assert_eq!(changed["by"]["/Trips/Lisbon.md"], "phoneA", "the event says which phone wrote it: {changed}");
    let (_, body) = t.raw("GET", "/v1/vault/file?path=/Trips/Lisbon.md&off=2&len=6", vec![]).await;
    assert_eq!(body, b"Lisbon");
    t.raw("PUT", "/v1/vault/file?path=/Recipes/Bread.md", b"# Bread\n\nFlour, water, salt, yeast. Knead and bake at 230C.".to_vec()).await;
    let (_, scan) = t.call("GET", "/v1/vault/scan", None, true).await;
    let paths: Vec<&str> = scan.as_array().unwrap().iter().map(|r| r[0].as_str().unwrap()).collect();
    assert!(paths.contains(&"/Trips/Lisbon.md") && paths.contains(&"/Recipes") && paths.contains(&"/Recipes/Bread.md"), "{paths:?}");
    let (st, v) = t.call("POST", "/v1/vault/op", Some(json!({"op": "move", "path": "/Recipes/Bread.md", "to": "/Recipes/Sourdough.md"})), true).await;
    assert!(st == 200 && v["ok"] == true, "{v}");
    let (st, v) = t.call("GET", "/v1/vault/file?path=/../etc/passwd", None, true).await;
    assert!(st == 404 && v["error"].as_str().unwrap().contains("bad path"), "nothing outside the vault: {v}");

    // --- real processes, in the vault folder; their files reach the other phone
    let (st, v) = t.call("POST", "/v1/exec", Some(json!({"cmd": "bash", "script": "echo from-bash > made.txt; ls Trips; echo err >&2; exit 3", "cwd": "/"})), true).await;
    assert!(st == 200 && v["code"] == 3 && v["stdout"].as_str().unwrap().contains("Lisbon.md") && v["stderr"] == "err\n", "bash: {v}");
    assert_eq!(std::fs::read_to_string(t.vault.join("made.txt")).unwrap(), "from-bash\n");
    loop {
        let e = next_json(&mut ev).await;
        if e["type"] == "changed" && e["paths"].as_array().unwrap().iter().any(|p| p == "/made.txt") {
            assert!(e["by"].get("/made.txt").is_none(), "a command's file has no phone as its writer");
            break;
        }
    }
    let (_, v) = t.call("POST", "/v1/exec", Some(json!({"cmd": "python3", "script": "import os,sys\nprint(sorted(os.listdir('.')))\nprint(sys.stdin.read().upper())", "cwd": "/Trips", "stdin": "hi"})), true).await;
    assert_eq!(v["stdout"], "['Lisbon.md']\nHI\n", "python3: {v}");
    let (_, v) = t.call("POST", "/v1/exec", Some(json!({"cmd": "node", "script": "const fs=require('fs');console.log(fs.readFileSync('Trips/Lisbon.md','utf8').split('\\n')[0], process.env.JADE_DEVICE)"})), true).await;
    assert_eq!(v["stdout"], "# Lisbon phoneA\n", "node: {v}");
    let t0 = Instant::now();
    let (_, v) = t.call("POST", "/v1/exec", Some(json!({"cmd": "bash", "script": "sleep 30 & sleep 30; echo never", "timeout": 1})), true).await;
    assert!(v["timed_out"] == true && v["code"] == 124 && t0.elapsed() < Duration::from_secs(5), "time limit: {v}");
    let (st, v) = t.call("POST", "/v1/exec", Some(json!({"cmd": "perl", "script": "print 1"})), true).await;
    assert!(st == 400 && v["error"].as_str().unwrap().contains("not allowed"), "{v}");
    // kill by id while it runs
    let tt = Arc::new(t);
    let t2 = tt.clone();
    let long = tokio::spawn(async move { t2.call("POST", "/v1/exec", Some(json!({"cmd": "bash", "script": "sleep 20", "id": "job1", "timeout": 60})), true).await });
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (_, k) = tt.call("POST", "/v1/exec/kill", Some(json!({"id": "job1"})), true).await;
    let (_, v) = long.await.unwrap();
    assert!(k["killed"] == true && v["killed"] == true, "kill: {k} {v}");
    let t = tt;

    // --- an interactive terminal on a real pseudo-terminal
    let mut term = t.ws("/v1/term?cmd=bash&cols=100&rows=30").await;
    term.send(WsMsg::Binary(b"stty size; echo hi-$((40+2))\n".to_vec().into())).await.unwrap();
    let mut out = String::new();
    while !(out.contains("hi-42") && out.contains("30 100")) {
        match tokio::time::timeout(Duration::from_secs(10), term.next()).await.expect("terminal output") {
            Some(Ok(WsMsg::Binary(b))) => out.push_str(&String::from_utf8_lossy(&b)),
            Some(Ok(_)) => {}
            other => panic!("terminal ended: {other:?} after {out:?}"),
        }
    }
    term.send(WsMsg::Text(json!({"type": "resize", "cols": 120, "rows": 40}).to_string().into())).await.unwrap();
    term.send(WsMsg::Binary(b"stty size; exit 7\n".to_vec().into())).await.unwrap();
    let mut code = None;
    let mut out2 = String::new();
    while code.is_none() {
        match tokio::time::timeout(Duration::from_secs(10), term.next()).await.expect("terminal exit") {
            Some(Ok(WsMsg::Binary(b))) => out2.push_str(&String::from_utf8_lossy(&b)),
            Some(Ok(WsMsg::Text(t))) => code = serde_json::from_str::<Value>(&t).unwrap()["code"].as_i64(),
            Some(Ok(_)) => {}
            other => panic!("terminal: {other:?}"),
        }
    }
    assert!(code == Some(7) && out2.contains("40 120"), "exit code and resize: {code:?} {out2:?}");

    // --- memory, and the vault's notes indexed for vector search
    let (st, v) = t.call("POST", "/v1/memory", Some(json!({"content": "The owner prefers window seats on flights", "category": "preferences"})), true).await;
    assert!(st == 200 && v["id"].as_i64().unwrap() > 0, "{v}");
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let (_, i) = t.call("GET", "/v1/info", None, true).await;
        if i["notes_indexed"].as_i64().unwrap_or(0) >= 2 {
            break;
        }
        assert!(Instant::now() < deadline, "notes were not indexed: {i}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let (_, v) = t.call("GET", "/v1/search?q=lisbon%20tram%20flights&k=3", None, true).await;
    let hits = v["hits"].as_array().unwrap();
    assert_eq!(hits[0]["source"], "/Trips/Lisbon.md", "the Lisbon note comes first: {v}");
    assert!(hits.iter().any(|h| h["kind"] == "memory"), "memories are searched too: {v}");
    let (_, v) = t.call("GET", "/v1/search?q=flour%20yeast&kind=note", None, true).await;
    assert_eq!(v["hits"][0]["source"], "/Recipes/Sourdough.md", "a moved note is found at its new path: {v}");
    std::fs::remove_file(t.vault.join("Recipes/Sourdough.md")).unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let (_, v) = t.call("GET", "/v1/search?q=flour%20yeast&kind=note", None, true).await;
    assert!(!v["hits"].as_array().unwrap().iter().any(|h| h["source"] == "/Recipes/Sourdough.md"), "a deleted note leaves the index: {v}");

    // --- the WebSocket chat: context, streamed reply, history
    let mut c = t.ws("/ws/alpha").await;
    c.send(WsMsg::Text(json!({"message": "what about the lisbon tram", "id": 1}).to_string().into())).await.unwrap();
    let ctx = next_json(&mut c).await;
    assert!(ctx["type"] == "context" && ctx["hits"][0]["source"] == "/Trips/Lisbon.md", "{ctx}");
    let mut text = String::new();
    let done = loop {
        let m = next_json(&mut c).await;
        if m["type"] == "token_delta" {
            assert!(m["device_id"] == "alpha" && m["avatar_expression"] == "neutral" && m["id"] == 1);
            text.push_str(m["delta"].as_str().unwrap());
        } else {
            break m;
        }
    };
    assert!(done["type"] == "done" && done["cancelled"] == false, "{done}");
    assert_eq!(text, "Echo: what about the lisbon tram");
    let sys = t.seen.lock().unwrap().chat.last().unwrap()["messages"][0]["content"].as_str().unwrap().to_string();
    assert!(sys.starts_with("You are Jade") && sys.contains("Relevant Vault Context") && sys.contains("tram 28") && sys.contains("(/Trips/Lisbon.md)"), "{sys}");
    c.send(WsMsg::Text(json!({"message": "and the bread?"}).to_string().into())).await.unwrap();
    loop {
        if next_json(&mut c).await["type"] == "done" {
            break;
        }
    }
    let msgs = t.seen.lock().unwrap().chat.last().unwrap()["messages"].as_array().unwrap().clone();
    assert!(msgs.len() == 4 && msgs[1]["content"] == "what about the lisbon tram" && msgs[2]["content"] == "Echo: what about the lisbon tram", "history: {msgs:?}");

    // busy, then cancel
    c.send(WsMsg::Text(json!({"message": "slow one", "id": "s"}).to_string().into())).await.unwrap();
    c.send(WsMsg::Text(json!({"message": "another", "id": "x"}).to_string().into())).await.unwrap();
    let mut got_busy = false;
    let mut deltas = 0;
    loop {
        let m = next_json(&mut c).await;
        if m["type"] == "error" && m["code"] == "busy" && m["id"] == "x" {
            got_busy = true;
        }
        if m["type"] == "token_delta" {
            deltas += 1;
            if deltas == 3 {
                c.send(WsMsg::Text(json!({"type": "cancel"}).to_string().into())).await.unwrap();
            }
        }
        if m["type"] == "done" {
            assert!(m["cancelled"] == true && m["id"] == "s", "{m}");
            break;
        }
    }
    assert!(got_busy && deltas < 15, "busy reported, reply stopped after cancel ({deltas} deltas)");
    c.send(WsMsg::Text(json!({"type": "remember", "content": "Alpha phone lives in the kitchen", "id": 9}).to_string().into())).await.unwrap();
    loop {
        let m = next_json(&mut c).await;
        if m["type"] == "remembered" {
            assert!(m["memory_id"].as_i64().unwrap() > 0 && m["id"] == 9);
            break;
        }
    }

    // many phones at once: five slow replies (2 s each) overlap
    let t0 = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..5 {
        let mut s = t.ws(&format!("/ws/phone{i}")).await;
        tasks.push(tokio::spawn(async move {
            s.send(WsMsg::Text(json!({"message": "slow please"}).to_string().into())).await.unwrap();
            let mut n = 0;
            loop {
                let m = next_json(&mut s).await;
                if m["type"] == "token_delta" {
                    n += 1;
                }
                if m["type"] == "done" {
                    return n;
                }
            }
        }));
    }
    for h in tasks {
        assert_eq!(h.await.unwrap(), 40);
    }
    let took = t0.elapsed();
    assert!(took < Duration::from_secs(5), "five 2-second replies took {took:?}, not 10 s back to back");

    // --- straight through to the models, decisions and vision; the fleet token is not passed on
    let (st, v) = t.call("GET", "/v1/models?token=test-token&x=1", None, true).await;
    assert!(st == 200 && v["echo_path"] == "/v1/models" && v["echo_query"] == "x=1", "{v}");
    let (_, v) = t.call("POST", "/v1/systemone", Some(json!({"model": "pplx-decider", "state": "s"})), true).await;
    assert!(v["echo_path"] == "/v1/systemone" && v["body"].as_str().unwrap().contains("pplx-decider"), "decisions go to Ollaya: {v}");
    let (_, v) = t.call("POST", "/ollaya/v1/systemone", Some(json!({"model": "laya"})), true).await;
    assert!(v["echo_path"] == "/v1/systemone" && v["body"].as_str().unwrap().contains("laya"), "Ollaya directly: {v}");
    let (_, v) = t.call("POST", "/ollama/v1/systemone", Some(json!({"model": "pplx-decider"})), true).await;
    assert_eq!(v["echo_path"], "/v1/systemone", "Ollama decisions directly: {v}");
    let (_, v) = t.call("GET", "/api/tags", None, true).await;
    assert_eq!(v["echo_path"], "/api/tags");
    let (_, v) = t.call("POST", "/v1/vision/detect", Some(json!({"image": "AAAA"})), true).await;
    assert_eq!(v["echo_path"], "/v1/vision/detect", "vision goes to the Xavier: {v}");
    let (_, v) = t.call("GET", "/llm/health", None, true).await;
    assert_eq!(v["echo_path"], "/health");
    assert!(t.seen.lock().unwrap().headers.iter().all(|(_, auth, _)| auth.is_none()), "the fleet token never leaves the broker");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn models_down() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = Config::from_env();
    cfg.data_dir = dir.path().join("d");
    cfg.vault_dir = dir.path().join("v");
    cfg.token = TOKEN.into();
    cfg.ollama_url = "http://127.0.0.1:9".into();
    cfg.decide_url = cfg.ollama_url.clone();
    cfg.llm_url = "http://127.0.0.1:9/v1".into();
    cfg.embed_dim = DIM;
    let app = jade_fleet::build(cfg).await.unwrap();
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    tokio::spawn(jade_fleet::serve(app, l, std::future::pending()));
    let t = T { base, up: jade_fleet::upstream::Upstream::new(), seen: Default::default(), vault: dir.path().join("v"), _dir: tempfile::tempdir().unwrap() };
    let mut c = t.ws("/ws/z").await;
    c.send(WsMsg::Text(json!({"message": "hello"}).to_string().into())).await.unwrap();
    let w = next_json(&mut c).await;
    assert!(w["type"] == "warning" && w["message"].as_str().unwrap().contains("Ollama"), "no context, said why: {w}");
    let e = next_json(&mut c).await;
    assert!(e["type"] == "error" && e["message"].as_str().unwrap().contains("model unreachable"), "{e}");
    let (st, v) = t.call("POST", "/v1/chat/completions", Some(json!({"messages": []})), true).await;
    assert!(st == 502 && v["ok"] == false, "{st} {v}");
    let (st, v) = t.call("GET", "/v1/search?q=x", None, true).await;
    assert!(st == 502 && v["error"].as_str().unwrap().contains("Ollama"), "{v}");
}
