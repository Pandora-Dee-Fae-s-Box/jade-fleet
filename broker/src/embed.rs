//! Embeddings from Ollama on the Xavier. A few at a time, so a big re-index does not starve chat requests.

use crate::config::Config;
use crate::upstream::Upstream;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct Embedder {
    up: Upstream,
    cfg: Arc<Config>,
    /// requests to Ollama in flight; search queries take a separate permit pool so they never wait behind indexing
    bulk: Arc<Semaphore>,
    live: Arc<Semaphore>,
}

impl Embedder {
    pub fn new(up: Upstream, cfg: Arc<Config>) -> Embedder {
        Embedder { up, cfg, bulk: Arc::new(Semaphore::new(1)), live: Arc::new(Semaphore::new(4)) }
    }

    /// Vector for a search query.
    pub async fn query(&self, text: &str) -> Result<Vec<f32>, String> {
        let _p = self.live.acquire().await.map_err(|e| e.to_string())?;
        let t = format!("{}{}", self.cfg.embed_query_prefix, text);
        Ok(self.call(&[t]).await?.remove(0))
    }

    /// Vectors for things being stored (memories, note chunks). `live`: someone is waiting for it.
    pub async fn documents(&self, texts: &[String], live: bool) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let sem = if live { &self.live } else { &self.bulk };
        let _p = sem.acquire().await.map_err(|e| e.to_string())?;
        let t: Vec<String> = texts.iter().map(|x| format!("{}{}", self.cfg.embed_doc_prefix, x)).collect();
        self.call(&t).await
    }

    async fn call(&self, input: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let url = format!("{}/api/embed", self.cfg.ollama_url);
        let (st, v) = self.up.post_json(&url, &json!({"model": self.cfg.embed_model, "input": input}), self.cfg.embed_timeout).await.map_err(|e| format!("could not reach Ollama for embeddings ({e})"))?;
        let out: Vec<Vec<f32>> = if st == 404 && v.get("error").is_none() {
            // an older Ollama without /api/embed: one text at a time
            let mut out = Vec::new();
            for t in input {
                let (st, v) = self.up.post_json(&format!("{}/api/embeddings", self.cfg.ollama_url), &json!({"model": self.cfg.embed_model, "prompt": t}), self.cfg.embed_timeout).await?;
                if st != 200 {
                    return Err(format!("Ollama embeddings: HTTP {st}: {}", err_text(&v)));
                }
                out.push(floats(v.get("embedding")));
            }
            out
        } else if st != 200 {
            return Err(format!("Ollama embeddings: HTTP {st}: {}{}", err_text(&v), if st == 404 { format!(" (run: ollama pull {})", self.cfg.embed_model) } else { String::new() }));
        } else {
            v.get("embeddings").and_then(Value::as_array).map(|a| a.iter().map(|e| floats(Some(e))).collect()).unwrap_or_default()
        };
        if out.len() != input.len() {
            return Err(format!("Ollama returned {} embeddings for {} texts", out.len(), input.len()));
        }
        if let Some(bad) = out.iter().find(|e| e.len() != self.cfg.embed_dim) {
            return Err(format!("{} gives {}-dimension vectors but EMBED_DIM is {}: set EMBED_DIM={} (the index is rebuilt on restart)", self.cfg.embed_model, bad.len(), self.cfg.embed_dim, bad.len()));
        }
        Ok(out)
    }
}

fn floats(v: Option<&Value>) -> Vec<f32> {
    v.and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default()
}

fn err_text(v: &Value) -> String {
    v.get("error").and_then(Value::as_str).map(String::from).unwrap_or_else(|| v.to_string().chars().take(300).collect())
}
