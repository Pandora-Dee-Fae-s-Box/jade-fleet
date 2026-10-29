//! HTTP to the machines behind the broker: Ollama on this box (embeddings) and llama-server on the Xavier (chat).
//! Plain HTTP on the LAN, kept-alive connections, many requests at once.

use axum::body::Body;
use http_body_util::BodyExt;
use hyper::{Method, Request, Response};
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::TokioExecutor;
use serde_json::Value;
use std::time::Duration;

#[derive(Clone)]
pub struct Upstream {
    pub client: Client<HttpConnector, Body>,
}

impl Default for Upstream {
    fn default() -> Self {
        Self::new()
    }
}

impl Upstream {
    pub fn new() -> Upstream {
        let mut c = HttpConnector::new();
        c.set_connect_timeout(Some(Duration::from_secs(5)));
        c.set_nodelay(true);
        c.set_keepalive(Some(Duration::from_secs(60)));
        let client = Client::builder(TokioExecutor::new()).pool_idle_timeout(Duration::from_secs(90)).build(c);
        Upstream { client }
    }

    /// Send a request and get the response head; the body streams.
    pub async fn send(&self, req: Request<Body>, timeout: Duration) -> Result<Response<hyper::body::Incoming>, String> {
        let uri = req.uri().to_string();
        match tokio::time::timeout(timeout, self.client.request(req)).await {
            Err(_) => Err(format!("{uri}: no answer within {}s", timeout.as_secs())),
            Ok(Err(e)) => Err(format!("{uri}: {}", describe(&e))),
            Ok(Ok(r)) => Ok(r),
        }
    }

    pub async fn post_json(&self, url: &str, body: &Value, timeout: Duration) -> Result<(u16, Value), String> {
        let req = Request::builder().method(Method::POST).uri(url).header("content-type", "application/json").body(Body::from(body.to_string())).map_err(|e| e.to_string())?;
        let fut = async {
            let res = self.send(req, timeout).await?;
            let status = res.status().as_u16();
            let bytes = res.into_body().collect().await.map_err(|e| format!("{url}: {e}"))?.to_bytes();
            let v = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).chars().take(500).collect()));
            Ok((status, v))
        };
        tokio::time::timeout(timeout, fut).await.map_err(|_| format!("{url}: no answer within {}s", timeout.as_secs()))?
    }

    pub async fn get_json(&self, url: &str, timeout: Duration) -> Result<(u16, Value), String> {
        let req = Request::builder().method(Method::GET).uri(url).body(Body::empty()).map_err(|e| e.to_string())?;
        let fut = async {
            let res = self.send(req, timeout).await?;
            let status = res.status().as_u16();
            let bytes = res.into_body().collect().await.map_err(|e| format!("{url}: {e}"))?.to_bytes();
            Ok((status, serde_json::from_slice(&bytes).unwrap_or(Value::Null)))
        };
        tokio::time::timeout(timeout, fut).await.map_err(|_| format!("{url}: no answer within {}s", timeout.as_secs()))?
    }
}

fn describe(e: &hyper_util::client::legacy::Error) -> String {
    let mut s = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(x) = src {
        s = format!("{s}: {x}");
        src = x.source();
    }
    s
}
