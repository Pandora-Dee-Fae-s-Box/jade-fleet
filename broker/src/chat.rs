//! The WebSocket chat from the original design, for thin terminals: a phone sends a message, the broker finds the
//! nearest memories and vault notes (Ollama embedding + sqlite-vec), and streams the Xavier's reply back token by
//! token. Many phones at once; each can cancel its reply.
//!
//! Phone → broker:  {"message": "...", "id": "optional"}   {"type":"cancel"}   {"type":"reset"}
//!                  {"type":"remember","content":"...","category":"..."}   {"type":"ping"}
//! Broker → phone:  {"type":"context","hits":[…]}  {"type":"token_delta","delta":"…","device_id":…,"avatar_expression":…}
//!                  {"type":"done","cancelled":bool,"ms":N}  {"type":"warning"|"error","message":…}
//!                  {"type":"remembered","memory_id":N}  {"type":"pong"}   (each carries the request's "id")

use crate::db::Hit;
use crate::App;
use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, Uri};
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Nearest memories and/or note chunks for a piece of text.
pub async fn rag(app: &App, text: &str, k: usize, kind: &str) -> Result<Vec<Hit>, String> {
    let v = app.emb.query(text).await?;
    let (mem, notes) = match kind {
        "memory" => (true, false),
        "note" => (false, true),
        _ => (true, true),
    };
    let db = app.db.clone();
    tokio::task::spawn_blocking(move || db.search(&v, k, mem, notes)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
}

pub async fn ws(State(app): State<App>, Path(device): Path<String>, h: HeaderMap, uri: Uri, ws: WebSocketUpgrade) -> Response {
    let device: String = device.chars().filter(|c| c.is_alphanumeric() || "-_.".contains(*c)).take(64).collect();
    let alias = crate::api::device_of(&h, &uri);
    ws.on_upgrade(move |s| session(app, device, if alias == "anon" { None } else { Some(alias) }, s))
}

type History = Arc<Mutex<Vec<(String, String)>>>;

async fn session(app: App, device: String, alias: Option<String>, sock: WebSocket) {
    tracing::info!("terminal connected: {device}");
    let (a, d) = (app.clone(), device.clone());
    let _ = tokio::task::spawn_blocking(move || a.db.touch_session(&d, alias.as_deref())).await;
    let (mut sink, mut stream) = sock.split();
    let (tx, mut rx) = mpsc::channel::<Value>(256);
    let writer = tokio::spawn(async move {
        while let Some(v) = rx.recv().await {
            if sink.send(Message::Text(v.to_string().into())).await.is_err() {
                break;
            }
        }
    });
    let history: History = Arc::new(Mutex::new(Vec::new()));
    let mut current: Option<(tokio::task::JoinHandle<()>, Value)> = None;
    while let Some(Ok(m)) = stream.next().await {
        let text = match m {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };
        let Ok(p) = serde_json::from_str::<Value>(&text) else {
            let _ = tx.send(json!({"type": "error", "message": "send JSON"})).await;
            continue;
        };
        let id = p.get("id").cloned().unwrap_or(Value::Null);
        let kind = p.get("type").and_then(Value::as_str).unwrap_or(if p.get("message").is_some() { "chat" } else { "" });
        if let Some((h, _)) = &current {
            if h.is_finished() {
                current = None;
            }
        }
        match kind {
            "chat" => {
                if let Some((_, cid)) = &current {
                    let _ = tx.send(json!({"type": "error", "id": id, "code": "busy", "message": "a reply is still streaming; send {\"type\":\"cancel\"} first", "busy_with": cid})).await;
                    continue;
                }
                let msg = p.get("message").and_then(Value::as_str).unwrap_or("").to_string();
                let h = tokio::spawn(reply(app.clone(), device.clone(), msg, id.clone(), tx.clone(), history.clone()));
                current = Some((h, id));
            }
            "cancel" => {
                if let Some((h, cid)) = current.take() {
                    h.abort();
                    let _ = tx.send(json!({"type": "done", "id": cid, "device_id": device, "cancelled": true})).await;
                }
            }
            "reset" => history.lock().unwrap().clear(),
            "remember" => {
                let (app, tx, dev) = (app.clone(), tx.clone(), device.clone());
                let content = p.get("content").and_then(Value::as_str).unwrap_or("").trim().to_string();
                let cat = p.get("category").and_then(Value::as_str).unwrap_or("general").to_string();
                tokio::spawn(async move {
                    if content.is_empty() {
                        let _ = tx.send(json!({"type": "error", "id": id, "message": "content is empty"})).await;
                        return;
                    }
                    let r = match app.emb.documents(&[content.clone()], true).await {
                        Ok(mut v) => {
                            let e = v.remove(0);
                            let db = app.db.clone();
                            tokio::task::spawn_blocking(move || db.add_memory(&dev, &cat, &content, &e)).await.map_err(|e| e.to_string()).and_then(|r| r.map_err(|e| e.to_string()))
                        }
                        Err(e) => Err(e),
                    };
                    let _ = tx.send(match r {
                        Ok(mid) => json!({"type": "remembered", "id": id, "memory_id": mid}),
                        Err(e) => json!({"type": "error", "id": id, "message": e}),
                    }).await;
                });
            }
            "ping" => {
                let _ = tx.send(json!({"type": "pong", "id": id})).await;
            }
            other => {
                let _ = tx.send(json!({"type": "error", "id": id, "message": format!("unknown type {other:?}")})).await;
            }
        }
    }
    if let Some((h, _)) = current {
        h.abort();
    }
    drop(tx);
    let _ = writer.await;
    tracing::info!("terminal disconnected: {device}");
}

async fn reply(app: App, device: String, msg: String, id: Value, tx: mpsc::Sender<Value>, history: History) {
    let t0 = std::time::Instant::now();
    // 1. nearest memories and notes (a failure here only means no context)
    let mut system = app.cfg.system_prompt.clone();
    if app.cfg.rag_k > 0 {
        match rag(&app, &msg, app.cfg.rag_k, "all").await {
            Ok(hits) if !hits.is_empty() => {
                system.push_str("\nRelevant Vault Context:\n");
                for h in &hits {
                    let src = if h.kind == "note" { format!(" ({})", h.source) } else { String::new() };
                    system.push_str(&format!("- {}{}\n", h.content.replace('\n', " "), src));
                }
                let brief: Vec<Value> = hits.iter().map(|h| json!({"kind": h.kind, "source": h.source, "distance": (h.distance * 10000.0).round() / 10000.0})).collect();
                let _ = tx.send(json!({"type": "context", "id": id, "hits": brief})).await;
            }
            Ok(_) => {}
            Err(e) => {
                let _ = tx.send(json!({"type": "warning", "id": id, "message": format!("no vault context: {e}")})).await;
            }
        }
    }
    // 2. the prompt: system, recent turns, this message
    let mut messages = vec![json!({"role": "system", "content": system})];
    for (u, a) in history.lock().unwrap().iter() {
        messages.push(json!({"role": "user", "content": u}));
        messages.push(json!({"role": "assistant", "content": a}));
    }
    messages.push(json!({"role": "user", "content": msg}));
    // 3. stream from the Xavier
    let body = json!({"model": app.cfg.llm_model, "messages": messages, "stream": true, "temperature": app.cfg.llm_temperature});
    let req = hyper::Request::builder().method("POST").uri(format!("{}/chat/completions", app.cfg.llm_url)).header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    let res = match app.up.send(req, std::time::Duration::from_secs(60)).await {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(json!({"type": "error", "id": id, "message": format!("model unreachable: {e}")})).await;
            return;
        }
    };
    if !res.status().is_success() {
        let st = res.status();
        let b = res.into_body().collect().await.map(|c| c.to_bytes()).unwrap_or_default();
        let _ = tx.send(json!({"type": "error", "id": id, "message": format!("model: HTTP {st}: {}", String::from_utf8_lossy(&b).chars().take(300).collect::<String>())})).await;
        return;
    }
    let mut body = res.into_body();
    let mut buf: Vec<u8> = Vec::new();
    let mut answer = String::new();
    let idle = app.cfg.llm_timeout;
    'outer: loop {
        let frame = match tokio::time::timeout(idle, body.frame()).await {
            Err(_) => {
                let _ = tx.send(json!({"type": "error", "id": id, "message": "the model stopped sending"})).await;
                return;
            }
            Ok(None) => break,
            Ok(Some(Err(e))) => {
                let _ = tx.send(json!({"type": "error", "id": id, "message": format!("stream: {e}")})).await;
                return;
            }
            Ok(Some(Ok(f))) => f,
        };
        let Ok(data) = frame.into_data() else { continue };
        buf.extend_from_slice(&data);
        while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim();
            let Some(d) = line.strip_prefix("data:") else { continue };
            let d = d.trim();
            if d == "[DONE]" {
                break 'outer;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(d) else { continue };
            let delta = chunk["choices"][0]["delta"]["content"].as_str().unwrap_or("");
            if !delta.is_empty() {
                answer.push_str(delta);
                if tx.send(json!({"device_id": device, "id": id, "type": "token_delta", "delta": delta, "avatar_expression": "neutral"})).await.is_err() {
                    return;
                }
            }
        }
    }
    {
        let mut h = history.lock().unwrap();
        h.push((msg, answer.clone()));
        let keep = app.cfg.history_turns;
        if h.len() > keep {
            let cut = h.len() - keep;
            h.drain(..cut);
        }
    }
    let _ = tx.send(json!({"type": "done", "id": id, "device_id": device, "cancelled": false, "ms": t0.elapsed().as_millis() as u64, "chars": answer.chars().count()})).await;
}
