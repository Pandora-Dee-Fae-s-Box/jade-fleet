//! Watches the vault folder. Anything that changes a file there (a phone's write, a command run on this box, scp, an
//! editor) is batched and announced to every connected phone, and handed to the indexer.

use notify::{RecursiveMode, Watcher};
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};

/// Which phone wrote which path recently, so the change event can say who made it (a phone ignores its own echoes).
#[derive(Default)]
pub struct Writers(Mutex<HashMap<String, (String, Instant)>>);

impl Writers {
    pub fn mark(&self, path: &str, device: &str) {
        let mut m = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if m.len() > 4096 {
            m.retain(|_, (_, t)| t.elapsed() < Duration::from_secs(5));
        }
        m.insert(path.to_string(), (device.to_string(), Instant::now()));
    }
    fn who(&self, path: &str) -> Option<String> {
        let m = self.0.lock().unwrap_or_else(|e| e.into_inner());
        m.get(path).filter(|(_, t)| t.elapsed() < Duration::from_secs(5)).map(|(d, _)| d.clone())
    }
}

pub fn vault_path(root: &Path, p: &Path) -> Option<String> {
    let rel = p.strip_prefix(root).ok()?;
    let s = rel.to_string_lossy();
    if s.is_empty() || s.contains(crate::vault::TMP) {
        return None;
    }
    Some(format!("/{s}"))
}

/// Start watching. Change events go to `events` as {"type":"changed","paths":[...],"by":{path: device}} and each
/// changed path to `index`. Returns the watcher, which must be kept alive.
pub fn start(root: PathBuf, events: broadcast::Sender<Arc<serde_json::Value>>, index: mpsc::UnboundedSender<String>, writers: Arc<Writers>) -> notify::Result<notify::RecommendedWatcher> {
    let (tx, mut rx) = mpsc::unbounded_channel::<PathBuf>();
    let mut w = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(ev) = res {
            if matches!(ev.kind, notify::EventKind::Access(_)) {
                return;
            }
            for p in ev.paths {
                let _ = tx.send(p);
            }
        }
    })?;
    w.watch(&root, RecursiveMode::Recursive)?;
    tokio::spawn(async move {
        loop {
            let Some(first) = rx.recv().await else { break };
            let mut batch = BTreeSet::new();
            batch.insert(first);
            // collect what else changes in the next moment, so a burst of writes is one event
            let until = tokio::time::Instant::now() + Duration::from_millis(150);
            while let Ok(Some(p)) = tokio::time::timeout_at(until, rx.recv()).await {
                batch.insert(p);
                if batch.len() > 2000 {
                    break;
                }
            }
            let paths: Vec<String> = batch.iter().filter_map(|p| vault_path(&root, p)).collect::<BTreeSet<_>>().into_iter().collect();
            if paths.is_empty() {
                continue;
            }
            let by: serde_json::Map<String, serde_json::Value> = paths.iter().filter_map(|p| writers.who(p).map(|d| (p.clone(), json!(d)))).collect();
            for p in &paths {
                let _ = index.send(p.clone());
            }
            let _ = events.send(Arc::new(json!({"type": "changed", "paths": paths, "by": by})));
        }
    });
    Ok(w)
}
