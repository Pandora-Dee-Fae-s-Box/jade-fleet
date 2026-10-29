//! The broker's HTTP and WebSocket API. Everything except /health and the browser client needs the fleet token
//! (`Authorization: Bearer TOKEN`, or `?token=TOKEN` where headers cannot be set, as on a browser WebSocket).
//! A phone names itself with `X-Jade-Device: NAME` or `?device=NAME`.

use crate::exec::{self, ExecReq, Pty};
use crate::App;
use axum::body::{Body, Bytes};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, Path as AxPath, Query, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::Arc;

pub fn router(app: App) -> Router {
    let api = Router::new()
        .route("/v1/info", get(info))
        // the shared vault (same contract as the phone app's vault folder)
        .route("/v1/vault/info", get(vault_info))
        .route("/v1/vault/scan", get(vault_scan))
        .route("/v1/vault/file", get(vault_get).put(vault_put))
        .route("/v1/vault/op", post(vault_op))
        .route("/v1/events", get(events))
        // memory and vector search over memories and the vault's notes
        .route("/v1/memory", post(memory_add))
        .route("/v1/memory/{id}", delete(memory_del))
        .route("/v1/search", get(search))
        // real processes on this machine
        .route("/v1/exec", post(exec_run))
        .route("/v1/exec/kill", post(exec_kill))
        .route("/v1/term", get(term))
        // the draft's WebSocket chat (RAG + streamed reply from the Xavier)
        .route("/ws/{device}", get(crate::chat::ws))
        // through to the models
        .route("/v1/chat/completions", any(to_llm))
        .route("/v1/completions", any(to_llm))
        .route("/v1/models", any(to_llm))
        .route("/llm/{*rest}", any(to_llm_root))
        .route("/v1/systemone", any(to_decide))
        .route("/api/decide", any(to_decide))
        .route("/api/{*rest}", any(to_ollama))
        .route("/ollama/{*rest}", any(to_ollama))
        .route("/ollaya/{*rest}", any(to_ollaya))
        .route("/v1/vision/{*rest}", any(to_vision))
        .layer(middleware::from_fn_with_state(app.clone(), auth))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024));
    Router::new().route("/health", get(health)).merge(api).fallback(get(web)).with_state(app)
}

// ------------------------------------------------------------------ helpers

pub fn fail(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({"ok": false, "error": msg.into()}))).into_response()
}

fn ok(v: Value) -> Response {
    Json(v).into_response()
}

fn query_map(uri: &Uri) -> HashMap<String, String> {
    uri.query().map(|q| url_decode_pairs(q)).unwrap_or_default()
}

fn url_decode_pairs(q: &str) -> HashMap<String, String> {
    q.split('&').filter(|s| !s.is_empty()).map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (pct(k), pct(v))
    }).collect()
}

fn pct(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                    Ok(x) => {
                        out.push(x);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn device_of(h: &HeaderMap, uri: &Uri) -> String {
    let d = h.get("x-jade-device").and_then(|v| v.to_str().ok()).map(String::from).or_else(|| query_map(uri).remove("device")).unwrap_or_else(|| "anon".into());
    d.chars().filter(|c| c.is_alphanumeric() || "-_.".contains(*c)).take(64).collect::<String>().to_string()
}

fn same(a: &[u8], b: &[u8]) -> bool {
    // constant time for equal lengths
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn auth(State(app): State<App>, req: Request, next: Next) -> Response {
    let tok = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).map(|s| s.trim().to_string())
        .or_else(|| req.headers().get("x-jade-token").and_then(|v| v.to_str().ok()).map(String::from))
        .or_else(|| query_map(req.uri()).remove("token"));
    match tok {
        Some(t) if same(t.as_bytes(), app.cfg.token.as_bytes()) => next.run(req).await,
        _ => fail(StatusCode::UNAUTHORIZED, "missing or wrong fleet token"),
    }
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.expect("blocking task")
}

// ------------------------------------------------------------------ status

async fn health() -> Response {
    ok(json!({"ok": true, "service": "jade-fleet", "version": env!("CARGO_PKG_VERSION")}))
}

async fn info(State(app): State<App>) -> Response {
    let a = app.clone();
    let (counts, sessions, vec) = blocking(move || (a.db.counts().unwrap_or_default(), a.db.sessions().unwrap_or_default(), a.db.vec_version())).await;
    let online: Vec<Value> = app.devices.lock().unwrap().iter().map(|(d, n)| json!({"device": d, "connections": n})).collect();
    let p = &app.progress;
    use std::sync::atomic::Ordering::Relaxed;
    ok(json!({
        "ok": true, "version": env!("CARGO_PKG_VERSION"), "uptime_s": app.started.elapsed().as_secs(),
        "vault": {"root": app.cfg.vault_dir, "free": app.vault.free()},
        "memories": counts.0, "notes_indexed": counts.1, "devices_known": counts.2, "online": online,
        "sessions": sessions.iter().map(|(d, a, t)| json!({"device": d, "alias": a, "last_active": t})).collect::<Vec<_>>(),
        "index": {"queued": p.queued.load(Relaxed), "done": p.done.load(Relaxed), "failed": p.failed.load(Relaxed), "enabled": app.cfg.index_vault},
        "embed": {"model": app.cfg.embed_model, "dim": app.cfg.embed_dim, "ollama": app.cfg.ollama_url},
        "llm": {"url": app.cfg.llm_url, "model": app.cfg.llm_model},
        "decide": app.cfg.decide_url, "vision": app.cfg.vision_url, "shells": app.cfg.shells, "sqlite_vec": vec,
    }))
}

// ------------------------------------------------------------------ vault

async fn vault_info(State(app): State<App>) -> Response {
    ok(json!({"root": "fleet:".to_string() + &app.cfg.vault_dir.to_string_lossy(), "mode": "fleet", "sharedOk": true, "free": app.vault.free(), "warn": ""}))
}

async fn vault_scan(State(app): State<App>) -> Response {
    let v = app.vault.clone();
    match blocking(move || v.scan()).await {
        Ok(s) => ([(header::CONTENT_TYPE, "application/json")], s).into_response(),
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
struct FileQ {
    path: String,
    #[serde(default)]
    off: u64,
    #[serde(default)]
    len: Option<usize>,
    #[serde(default)]
    first: Option<String>,
    #[serde(default)]
    last: Option<String>,
}

async fn vault_get(State(app): State<App>, Query(q): Query<FileQ>) -> Response {
    let v = app.vault.clone();
    let len = q.len.unwrap_or(usize::MAX / 2);
    match blocking(move || v.read(&q.path, q.off, len)).await {
        Ok(b) => ([(header::CONTENT_TYPE, "application/octet-stream")], b).into_response(),
        Err(e) => fail(StatusCode::NOT_FOUND, e.to_string()),
    }
}

async fn vault_put(State(app): State<App>, h: HeaderMap, uri: Uri, Query(q): Query<FileQ>, body: Bytes) -> Response {
    let dev = device_of(&h, &uri);
    let flag = |s: &Option<String>| s.as_deref().map(|x| x == "1" || x == "true").unwrap_or(true);
    let (first, last) = (flag(&q.first), flag(&q.last));
    app.writers.mark(&q.path, &dev);
    let v = app.vault.clone();
    let path = q.path.clone();
    match blocking(move || v.put(&path, &body, first, last, &dev)).await {
        Ok(m) => ok(json!({"ok": true, "mtime": m})),
        Err(e) => fail(StatusCode::BAD_REQUEST, e.to_string()),
    }
}

#[derive(Deserialize)]
struct OpReq {
    op: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    to: String,
    #[serde(default)]
    mtime: i64,
}

async fn vault_op(State(app): State<App>, h: HeaderMap, uri: Uri, Json(r): Json<OpReq>) -> Response {
    let dev = device_of(&h, &uri);
    app.writers.mark(&r.path, &dev);
    if !r.to.is_empty() {
        app.writers.mark(&r.to, &dev);
    }
    let v = app.vault.clone();
    let res: Result<Value, std::io::Error> = blocking(move || match r.op.as_str() {
        "mkdir" => v.mkdir(&r.path).map(|_| json!({})),
        "delete" => v.delete(&r.path).map(|_| json!({})),
        "move" => v.move_(&r.path, &r.to).map(|_| json!({})),
        "touch" => v.touch(&r.path, r.mtime).map(|m| json!({"mtime": m})),
        "clear" => v.clear().map(|_| json!({})),
        o => Err(std::io::Error::other(format!("unknown op {o}"))),
    })
    .await;
    match res {
        Ok(mut v) => {
            v["ok"] = json!(true);
            ok(v)
        }
        Err(e) => fail(StatusCode::BAD_REQUEST, e.to_string()),
    }
}

/// Live events for a phone: {"type":"hello"}, {"type":"changed","paths":[…],"by":{path: device}},
/// {"type":"device","device":…,"online":bool}, and {"type":"resync"} if it fell behind (rescan then).
async fn events(State(app): State<App>, h: HeaderMap, uri: Uri, ws: WebSocketUpgrade) -> Response {
    let dev = device_of(&h, &uri);
    ws.on_upgrade(move |sock| events_loop(app, dev, sock))
}

async fn events_loop(app: App, dev: String, mut sock: WebSocket) {
    let mut rx = app.events.subscribe();
    {
        *app.devices.lock().unwrap().entry(dev.clone()).or_default() += 1;
    }
    let (a, d) = (app.clone(), dev.clone());
    let _ = blocking(move || a.db.touch_session(&d, None)).await;
    let _ = app.events.send(Arc::new(json!({"type": "device", "device": dev, "online": true})));
    let online: Vec<String> = app.devices.lock().unwrap().keys().cloned().collect();
    let _ = sock.send(Message::Text(json!({"type": "hello", "device": dev, "online": online}).to_string().into())).await;
    loop {
        tokio::select! {
            ev = rx.recv() => match ev {
                Ok(v) => if sock.send(Message::Text(v.to_string().into())).await.is_err() { break },
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => if sock.send(Message::Text(json!({"type": "resync"}).to_string().into())).await.is_err() { break },
                Err(_) => break,
            },
            m = sock.recv() => match m {
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                Some(Ok(Message::Ping(p))) => { let _ = sock.send(Message::Pong(p)).await; }
                _ => {}
            },
        }
    }
    let gone = {
        let mut m = app.devices.lock().unwrap();
        let n = m.entry(dev.clone()).or_default();
        *n = n.saturating_sub(1);
        if *n == 0 {
            m.remove(&dev);
            true
        } else {
            false
        }
    };
    if gone {
        let _ = app.events.send(Arc::new(json!({"type": "device", "device": dev, "online": false})));
    }
}

// ------------------------------------------------------------------ memory and search

#[derive(Deserialize)]
struct MemReq {
    content: String,
    #[serde(default)]
    category: Option<String>,
}

async fn memory_add(State(app): State<App>, h: HeaderMap, uri: Uri, Json(r): Json<MemReq>) -> Response {
    let dev = device_of(&h, &uri);
    let text = r.content.trim().to_string();
    if text.is_empty() {
        return fail(StatusCode::BAD_REQUEST, "content is empty");
    }
    let emb = match app.emb.documents(&[text.clone()], true).await {
        Ok(mut v) => v.remove(0),
        Err(e) => return fail(StatusCode::BAD_GATEWAY, e),
    };
    let cat = r.category.unwrap_or_else(|| "general".into());
    let a = app.clone();
    match blocking(move || a.db.add_memory(&dev, &cat, &text, &emb)).await {
        Ok(id) => ok(json!({"ok": true, "id": id})),
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn memory_del(State(app): State<App>, AxPath(id): AxPath<i64>) -> Response {
    let a = app.clone();
    match blocking(move || a.db.delete_memory(id)).await {
        Ok(found) => ok(json!({"ok": true, "deleted": found})),
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
struct SearchQ {
    q: String,
    #[serde(default)]
    k: Option<usize>,
    /// all (default), memory, note
    #[serde(default)]
    kind: Option<String>,
}

async fn search(State(app): State<App>, Query(q): Query<SearchQ>) -> Response {
    match crate::chat::rag(&app, &q.q, q.k.unwrap_or(8), q.kind.as_deref().unwrap_or("all")).await {
        Ok(hits) => ok(json!({"ok": true, "hits": hits})),
        Err(e) => fail(StatusCode::BAD_GATEWAY, e),
    }
}

// ------------------------------------------------------------------ processes

async fn exec_run(State(app): State<App>, h: HeaderMap, uri: Uri, Json(r): Json<ExecReq>) -> Response {
    let dev = device_of(&h, &uri);
    let ctx = exec::Ctx { vault: &app.vault, data_dir: &app.cfg.data_dir, shells: &app.cfg.shells, max: app.cfg.exec_max, device: &dev };
    match exec::run(r, ctx, app.running.clone()).await {
        Ok(v) => ok(v),
        Err(e) => fail(StatusCode::BAD_REQUEST, e),
    }
}

#[derive(Deserialize)]
struct KillReq {
    id: String,
}

async fn exec_kill(State(app): State<App>, Json(r): Json<KillReq>) -> Response {
    ok(json!({"ok": true, "killed": app.running.kill(&r.id)}))
}

/// A terminal: binary frames both ways carry the bytes; text frames from the phone are control messages
/// ({"type":"resize","cols":80,"rows":24}); the broker ends with {"type":"exit","code":N}.
async fn term(State(app): State<App>, h: HeaderMap, uri: Uri, ws: WebSocketUpgrade) -> Response {
    let q = query_map(&uri);
    let dev = device_of(&h, &uri);
    let cmd = q.get("cmd").cloned().unwrap_or_else(|| "bash".into());
    if !app.cfg.shells.contains(&cmd) {
        return fail(StatusCode::BAD_REQUEST, format!("{cmd} is not allowed here"));
    }
    let cwd = match app.vault.file(q.get("cwd").map(String::as_str).unwrap_or("/")) {
        Ok(p) if p.is_dir() => p,
        _ => return fail(StatusCode::BAD_REQUEST, "no such folder in the vault"),
    };
    let cols = q.get("cols").and_then(|x| x.parse().ok()).unwrap_or(80u16);
    let rows = q.get("rows").and_then(|x| x.parse().ok()).unwrap_or(24u16);
    let args: Vec<String> = if cmd == "bash" { vec!["-l".into()] } else { vec![] };
    let env = vec![("JADE_VAULT".to_string(), app.vault.root.to_string_lossy().into_owned()), ("JADE_DEVICE".to_string(), dev)];
    let pty = match Pty::spawn(&cmd, &args, &cwd, &env, cols, rows) {
        Ok(p) => p,
        Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{cmd}: {e}")),
    };
    ws.on_upgrade(move |sock| term_loop(pty, sock))
}

async fn term_loop(pty: Pty, mut sock: WebSocket) {
    let fd = pty.fd();
    let pid = pty.pid;
    let rd = unsafe { std::fs::File::from_raw_fd(libc::dup(fd)) };
    let mut wr = unsafe { std::fs::File::from_raw_fd(libc::dup(fd)) };
    let (otx, mut orx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    std::thread::spawn(move || {
        let mut rd = rd;
        let mut buf = vec![0u8; 16384];
        loop {
            match rd.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if otx.blocking_send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });
    let (itx, irx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        while let Ok(b) = irx.recv() {
            if wr.write_all(&b).is_err() {
                break;
            }
        }
    });
    let exit = tokio::task::spawn_blocking(move || Pty::wait(pid));
    tokio::pin!(exit);
    let mut code: Option<i32> = None;
    loop {
        tokio::select! {
            out = orx.recv() => match out {
                Some(b) => if sock.send(Message::Binary(b.into())).await.is_err() { break },
                None => {
                    // output closed: the program ended (or closed its terminal)
                    let c = (&mut exit).await.unwrap_or(-1);
                    code = Some(c);
                    break;
                }
            },
            m = sock.recv() => match m {
                Some(Ok(Message::Binary(b))) => { let _ = itx.send(b.to_vec()); }
                Some(Ok(Message::Text(t))) => {
                    if let Ok(v) = serde_json::from_str::<Value>(&t) {
                        if v["type"] == "resize" {
                            exec::set_size(pty.fd(), v["cols"].as_u64().unwrap_or(80) as u16, v["rows"].as_u64().unwrap_or(24) as u16);
                        } else if v["type"] == "input" {
                            let _ = itx.send(v["data"].as_str().unwrap_or("").as_bytes().to_vec());
                        }
                    }
                }
                Some(Ok(Message::Ping(p))) => { let _ = sock.send(Message::Pong(p)).await; }
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                _ => {}
            },
        }
    }
    match code {
        Some(c) => {
            let _ = sock.send(Message::Text(json!({"type": "exit", "code": c}).to_string().into())).await;
            let _ = sock.send(Message::Close(None)).await;
        }
        None => pty.hangup(),
    }
    let _ = fd.as_raw_fd();
}

// ------------------------------------------------------------------ through to the models

/// Forward a request as it is (method, headers, streaming body) and stream the answer back.
pub async fn proxy(app: &App, target: String, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let qs: String = parts.uri.query().map(|raw| raw.split('&').filter(|kv| !kv.starts_with("token=") && !kv.starts_with("device=")).collect::<Vec<_>>().join("&")).unwrap_or_default();
    let url = if qs.is_empty() { target } else { format!("{target}?{qs}") };
    let mut b = hyper::Request::builder().method(parts.method.clone()).uri(&url);
    for (k, v) in parts.headers.iter() {
        let n = k.as_str();
        if matches!(n, "host" | "authorization" | "x-jade-token" | "cookie" | "connection" | "upgrade" | "origin" | "referer") {
            continue;
        }
        b = b.header(k, v);
    }
    let up = match b.body(body) {
        Ok(r) => r,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e.to_string()),
    };
    match app.up.send(up, app.cfg.llm_timeout).await {
        Ok(res) => {
            let (p, inc) = res.into_parts();
            let mut r = Response::from_parts(p, Body::new(inc));
            r.headers_mut().remove(header::ACCESS_CONTROL_ALLOW_ORIGIN);
            r
        }
        Err(e) => fail(StatusCode::BAD_GATEWAY, e),
    }
}

fn path_after<'a>(uri: &'a Uri, prefix: &str) -> &'a str {
    uri.path().strip_prefix(prefix).unwrap_or("")
}

async fn to_llm(State(app): State<App>, req: Request) -> Response {
    let rest = path_after(req.uri(), "/v1").to_string();
    let t = format!("{}{}", app.cfg.llm_url, rest);
    proxy(&app, t, req).await
}

async fn to_llm_root(State(app): State<App>, req: Request) -> Response {
    let rest = path_after(req.uri(), "/llm").to_string();
    let root = app.cfg.llm_url.trim_end_matches("/v1").to_string();
    proxy(&app, format!("{root}{rest}"), req).await
}

async fn to_ollama(State(app): State<App>, req: Request) -> Response {
    let p = req.uri().path().to_string();
    let rest = p.strip_prefix("/ollama").unwrap_or(&p).to_string();
    proxy(&app, format!("{}{}", app.cfg.ollama_url, rest), req).await
}

async fn to_ollaya(State(app): State<App>, req: Request) -> Response {
    let rest = path_after(req.uri(), "/ollaya").to_string();
    proxy(&app, format!("{}{}", app.cfg.ollaya_url, rest), req).await
}

async fn to_decide(State(app): State<App>, req: Request) -> Response {
    let p = req.uri().path().to_string();
    proxy(&app, format!("{}{}", app.cfg.decide_url, p), req).await
}

async fn to_vision(State(app): State<App>, req: Request) -> Response {
    let p = req.uri().path().to_string();
    proxy(&app, format!("{}{}", app.cfg.vision_url, p), req).await
}

// ------------------------------------------------------------------ the browser client

async fn web(State(app): State<App>, method: Method, uri: Uri) -> Response {
    if method != Method::GET {
        return fail(StatusCode::NOT_FOUND, "not found");
    }
    let rel = uri.path().trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    if rel.split('/').any(|s| s == ".." || s.starts_with('.')) {
        return fail(StatusCode::NOT_FOUND, "not found");
    }
    let f = app.cfg.web_dir.join(rel);
    match tokio::fs::read(&f).await {
        Ok(b) => {
            let ct = match f.extension().and_then(|e| e.to_str()).unwrap_or("") {
                "html" => "text/html; charset=utf-8",
                "js" | "mjs" => "text/javascript; charset=utf-8",
                "css" => "text/css; charset=utf-8",
                "json" => "application/json",
                "wasm" => "application/wasm",
                "svg" => "image/svg+xml",
                "png" => "image/png",
                "ico" => "image/x-icon",
                _ => "application/octet-stream",
            };
            let mut r = (StatusCode::OK, b).into_response();
            r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(ct));
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            r
        }
        Err(_) if rel == "index.html" => (StatusCode::OK, [(header::CONTENT_TYPE, "text/html; charset=utf-8")], "<!doctype html><title>Jade Fleet</title><p>Jade Fleet broker is running. The browser client is not installed in FLEET_WEB.</p>").into_response(),
        Err(_) => fail(StatusCode::NOT_FOUND, "not found"),
    }
}
