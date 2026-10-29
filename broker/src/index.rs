//! Keeps the vector index of the shared vault's notes current: each changed note is split into chunks, embedded by
//! Ollama and stored in chunk_vec. Runs in the background; searches never wait for it.

use crate::db::Db;
use crate::embed::Embedder;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

pub const MAX_NOTE: u64 = 2 * 1024 * 1024;

pub fn indexable(p: &str) -> bool {
    let l = p.to_ascii_lowercase();
    (l.ends_with(".md") || l.ends_with(".markdown") || l.ends_with(".txt")) && !p.split('/').any(|s| s.starts_with('.'))
}

/// Split text into chunks of about `max` characters at paragraph, then line, then word boundaries.
pub fn chunk(text: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let push = |cur: &mut String, out: &mut Vec<String>| {
        let t = cur.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
        cur.clear();
    };
    for para in text.split("\n\n") {
        let n = para.chars().count();
        if cur.chars().count() + n + 2 > max && !cur.is_empty() {
            push(&mut cur, &mut out);
        }
        if n > max {
            // a long paragraph: cut at lines, then at words
            for line in para.split('\n') {
                for word in line.split(' ') {
                    if cur.chars().count() + word.chars().count() + 1 > max && !cur.is_empty() {
                        push(&mut cur, &mut out);
                    }
                    if !cur.is_empty() && !cur.ends_with('\n') {
                        cur.push(' ');
                    }
                    cur.push_str(word);
                }
                cur.push('\n');
            }
        } else {
            if !cur.is_empty() {
                cur.push_str("\n\n");
            }
            cur.push_str(para);
        }
    }
    push(&mut cur, &mut out);
    out
}

#[derive(Default)]
pub struct Progress {
    pub queued: AtomicUsize,
    pub done: AtomicUsize,
    pub failed: AtomicUsize,
}

pub struct Indexer {
    pub db: Arc<Db>,
    pub emb: Embedder,
    pub root: PathBuf,
    pub chunk_chars: usize,
    pub progress: Arc<Progress>,
}

impl Indexer {
    /// Run forever: index what arrives on `rx`; at start, everything that changed while the broker was stopped.
    pub async fn run(self, mut rx: mpsc::UnboundedReceiver<String>, full_scan: bool) {
        let me = Arc::new(self);
        if full_scan {
            me.reembed_memories().await;
            let root = me.root.clone();
            let mut all: Vec<String> = tokio::task::spawn_blocking(move || {
                walkdir_files(&root).into_iter().filter(|p| indexable(p)).collect()
            })
            .await
            .unwrap_or_default();
            // notes indexed before but gone now
            let present: BTreeSet<String> = all.iter().cloned().collect();
            let db = me.db.clone();
            if let Ok(Ok(old)) = tokio::task::spawn_blocking(move || db.indexed_paths()).await {
                for p in old {
                    if !present.contains(&p) {
                        let db = me.db.clone();
                        let _ = tokio::task::spawn_blocking(move || db.drop_note(&p)).await;
                    }
                }
            }
            all.sort();
            me.progress.queued.fetch_add(all.len(), Ordering::Relaxed);
            for p in all {
                me.one(&p).await;
            }
        }
        let mut pending: BTreeSet<String> = BTreeSet::new();
        loop {
            if pending.is_empty() {
                match rx.recv().await {
                    Some(p) => {
                        pending.insert(p);
                    }
                    None => break,
                }
            }
            // let a burst of writes to the same note settle first
            tokio::time::sleep(Duration::from_millis(400)).await;
            while let Ok(p) = rx.try_recv() {
                pending.insert(p);
            }
            let batch: Vec<String> = std::mem::take(&mut pending).into_iter().collect();
            for p in batch {
                if indexable(&p) {
                    me.progress.queued.fetch_add(1, Ordering::Relaxed);
                    me.one(&p).await;
                } else {
                    // a folder that moved or was deleted: drop what was indexed under it
                    let (db, root, pre) = (me.db.clone(), me.root.clone(), format!("{p}/"));
                    if let Ok(Ok(old)) = tokio::task::spawn_blocking(move || db.indexed_paths()).await {
                        for q in old.into_iter().filter(|q| q.starts_with(&pre)) {
                            if !root.join(&q[1..]).is_file() {
                                let db = me.db.clone();
                                let _ = tokio::task::spawn_blocking(move || db.drop_note(&q)).await;
                            }
                        }
                    }
                }
            }
        }
    }

    async fn reembed_memories(&self) {
        let db = self.db.clone();
        let todo = tokio::task::spawn_blocking(move || db.memories_without_vectors()).await.ok().and_then(|r| r.ok()).unwrap_or_default();
        for (id, text) in todo {
            match self.emb.documents(&[text], false).await {
                Ok(mut v) => {
                    let (db, e) = (self.db.clone(), v.remove(0));
                    let _ = tokio::task::spawn_blocking(move || db.set_memory_vector(id, &e)).await;
                }
                Err(e) => {
                    tracing::warn!("memory {id}: {e}");
                    return;
                }
            }
        }
    }

    async fn one(&self, p: &str) {
        let f = self.root.join(&p[1..]);
        let meta = tokio::fs::metadata(&f).await.ok().filter(|m| m.is_file());
        let db = self.db.clone();
        let Some(meta) = meta else {
            let p = p.to_string();
            let _ = tokio::task::spawn_blocking(move || db.drop_note(&p)).await;
            self.progress.done.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let sig = format!("{}:{}", meta.len(), crate::vault::mtime_ms(&meta));
        let pp = p.to_string();
        if tokio::task::spawn_blocking(move || db.note_sig(&pp)).await.ok().and_then(|r| r.ok()).flatten().as_deref() == Some(sig.as_str()) {
            self.progress.done.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if meta.len() > MAX_NOTE {
            self.progress.done.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let text = match tokio::fs::read(&f).await {
            Ok(b) => String::from_utf8_lossy(&b).into_owned(),
            Err(_) => return,
        };
        let title = p.rsplit('/').next().unwrap_or(p).trim_end_matches(".md").to_string();
        let chunks = chunk(&text, self.chunk_chars);
        // what the model embeds carries the note's name, so "the Lisbon trip" finds /Trips/Lisbon.md
        let inputs: Vec<String> = chunks.iter().map(|c| format!("{title}\n{c}")).collect();
        let mut vecs = Vec::new();
        for part in inputs.chunks(16) {
            match self.emb.documents(part, false).await {
                Ok(v) => vecs.extend(v),
                Err(e) => {
                    self.progress.failed.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("index {p}: {e}");
                    return;
                }
            }
        }
        let rows: Vec<(String, Vec<f32>)> = chunks.into_iter().zip(vecs).collect();
        let (db, p2) = (self.db.clone(), p.to_string());
        match tokio::task::spawn_blocking(move || db.set_note(&p2, &sig, &rows)).await {
            Ok(Ok(())) => {
                self.progress.done.fetch_add(1, Ordering::Relaxed);
            }
            other => {
                self.progress.failed.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("index {p}: {other:?}");
            }
        }
    }
}

fn walkdir_files(root: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let Ok(t) = e.file_type() else { continue };
            if t.is_dir() {
                stack.push(e.path());
            } else if t.is_file() {
                if let Some(v) = crate::watch::vault_path(root, &e.path()) {
                    out.push(v);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::chunk;

    #[test]
    fn chunks_keep_paragraphs_together() {
        let t = "one\n\ntwo\n\n".to_string() + &"word ".repeat(100);
        let c = chunk(&t, 120);
        assert_eq!(c[0], "one\n\ntwo");
        assert!(c.len() > 3 && c.iter().all(|x| x.chars().count() <= 125), "{c:?}");
        assert!(chunk("", 100).is_empty());
    }
}
