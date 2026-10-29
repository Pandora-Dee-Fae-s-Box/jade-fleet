//! SQLite in WAL mode with sqlite-vec compiled in. One writer connection and a small pool of readers, so vector
//! searches from many phones run at the same time while the indexer writes.
//!
//! Tables: terminal_sessions (devices), vault_notes + vault_vec (memories, as in the original design),
//! vault_chunks + chunk_vec (the shared vault's notes, split into chunks and embedded), meta.

use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};

extern "C" {
    fn sqlite3_vec_init(db: *mut rusqlite::ffi::sqlite3, err: *mut *mut std::os::raw::c_char, api: *const rusqlite::ffi::sqlite3_api_routines) -> std::os::raw::c_int;
}

static VEC: Once = Once::new();

/// Make sqlite-vec available on every connection opened from now on.
pub fn register_vec() {
    VEC.call_once(|| unsafe {
        // sqlite3_auto_extension takes the entry point as a generic function pointer
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<*const (), unsafe extern "C" fn(*mut rusqlite::ffi::sqlite3, *mut *mut std::os::raw::c_char, *const rusqlite::ffi::sqlite3_api_routines) -> std::os::raw::c_int>(sqlite3_vec_init as *const ())));
    });
}

pub fn blob(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Hit {
    /// "memory" or "note"
    pub kind: &'static str,
    pub id: i64,
    /// the note's vault path (notes) or the memory's category
    pub source: String,
    pub content: String,
    pub distance: f64,
}

pub struct Db {
    writer: Mutex<Connection>,
    readers: Vec<Mutex<Connection>>,
    next: AtomicUsize,
    pub dim: usize,
}

fn open(path: &Path) -> rusqlite::Result<Connection> {
    let c = Connection::open(path)?;
    c.busy_timeout(std::time::Duration::from_secs(10))?;
    c.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")?;
    Ok(c)
}

impl Db {
    /// Opens (and creates) the database. `fresh_vectors` is true when the vector tables were (re)made, for example
    /// after switching embedding model or dimension: everything must then be embedded again.
    pub fn open(path: &Path, dim: usize, model: &str, readers: usize) -> rusqlite::Result<(Arc<Db>, bool)> {
        register_vec();
        let w = open(path)?;
        w.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE IF NOT EXISTS terminal_sessions (
                 device_id TEXT PRIMARY KEY, alias TEXT, last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
             CREATE TABLE IF NOT EXISTS vault_notes (
                 id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL, category TEXT DEFAULT 'general',
                 content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
             CREATE TABLE IF NOT EXISTS vault_chunks (
                 id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, chunk INTEGER NOT NULL,
                 content TEXT NOT NULL, sig TEXT NOT NULL);
             CREATE INDEX IF NOT EXISTS vault_chunks_path ON vault_chunks(path);",
        )?;
        let want = format!("{model}:{dim}");
        let have: Option<String> = w.query_row("SELECT value FROM meta WHERE key = 'vectors'", [], |r| r.get(0)).optional()?;
        let fresh = have.as_deref() != Some(want.as_str());
        if fresh {
            w.execute_batch("DROP TABLE IF EXISTS vault_vec; DROP TABLE IF EXISTS chunk_vec; DELETE FROM vault_chunks;")?;
        }
        w.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vault_vec USING vec0(note_id INTEGER PRIMARY KEY, embedding float[{dim}]);
             CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{dim}]);"
        ))?;
        w.execute("INSERT INTO meta (key, value) VALUES ('vectors', ?1) ON CONFLICT(key) DO UPDATE SET value = excluded.value", params![want])?;
        let mut rs = Vec::new();
        for _ in 0..readers.max(1) {
            rs.push(Mutex::new(open(path)?));
        }
        Ok((Arc::new(Db { writer: Mutex::new(w), readers: rs, next: AtomicUsize::new(0), dim }), fresh))
    }

    pub fn write<R>(&self, f: impl FnOnce(&mut Connection) -> rusqlite::Result<R>) -> rusqlite::Result<R> {
        let mut c = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut c)
    }

    pub fn read<R>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<R>) -> rusqlite::Result<R> {
        let n = self.readers.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        // take a free reader if there is one, else wait for the next in turn
        for k in 0..n {
            if let Ok(c) = self.readers[(start + k) % n].try_lock() {
                return f(&c);
            }
        }
        let c = self.readers[start % n].lock().unwrap_or_else(|e| e.into_inner());
        f(&c)
    }

    pub fn touch_session(&self, device: &str, alias: Option<&str>) -> rusqlite::Result<()> {
        self.write(|c| {
            c.execute(
                "INSERT INTO terminal_sessions (device_id, alias, last_active) VALUES (?1, ?2, CURRENT_TIMESTAMP)
                 ON CONFLICT(device_id) DO UPDATE SET alias = COALESCE(excluded.alias, alias), last_active = CURRENT_TIMESTAMP",
                params![device, alias],
            )
            .map(|_| ())
        })
    }

    pub fn sessions(&self) -> rusqlite::Result<Vec<(String, Option<String>, String)>> {
        self.read(|c| {
            let mut s = c.prepare("SELECT device_id, alias, last_active FROM terminal_sessions ORDER BY last_active DESC")?;
            let rows = s.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
            rows.collect()
        })
    }

    pub fn add_memory(&self, device: &str, category: &str, content: &str, emb: &[f32]) -> rusqlite::Result<i64> {
        self.write(|c| {
            let tx = c.transaction()?;
            tx.execute("INSERT INTO vault_notes (device_id, category, content) VALUES (?1, ?2, ?3)", params![device, category, content])?;
            let id = tx.last_insert_rowid();
            tx.execute("INSERT INTO vault_vec (note_id, embedding) VALUES (?1, ?2)", params![id, blob(emb)])?;
            tx.commit()?;
            Ok(id)
        })
    }

    pub fn delete_memory(&self, id: i64) -> rusqlite::Result<bool> {
        self.write(|c| {
            let tx = c.transaction()?;
            tx.execute("DELETE FROM vault_vec WHERE note_id = ?1", params![id])?;
            let n = tx.execute("DELETE FROM vault_notes WHERE id = ?1", params![id])?;
            tx.commit()?;
            Ok(n > 0)
        })
    }

    /// Memories that have no vector yet (after a model change).
    pub fn memories_without_vectors(&self) -> rusqlite::Result<Vec<(i64, String)>> {
        self.read(|c| {
            let mut s = c.prepare("SELECT id, content FROM vault_notes WHERE id NOT IN (SELECT note_id FROM vault_vec)")?;
            let rows = s.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect()
        })
    }

    pub fn set_memory_vector(&self, id: i64, emb: &[f32]) -> rusqlite::Result<()> {
        self.write(|c| c.execute("INSERT OR REPLACE INTO vault_vec (note_id, embedding) VALUES (?1, ?2)", params![id, blob(emb)]).map(|_| ()))
    }

    /// The signature the vault note was indexed with ("size:mtime"), if any.
    pub fn note_sig(&self, path: &str) -> rusqlite::Result<Option<String>> {
        self.read(|c| c.query_row("SELECT sig FROM vault_chunks WHERE path = ?1 LIMIT 1", params![path], |r| r.get(0)).optional())
    }

    pub fn indexed_paths(&self) -> rusqlite::Result<Vec<String>> {
        self.read(|c| {
            let mut s = c.prepare("SELECT DISTINCT path FROM vault_chunks")?;
            let rows = s.query_map([], |r| r.get(0))?;
            rows.collect()
        })
    }

    /// Replace a note's chunks and their vectors.
    pub fn set_note(&self, path: &str, sig: &str, chunks: &[(String, Vec<f32>)]) -> rusqlite::Result<()> {
        self.write(|c| {
            let tx = c.transaction()?;
            tx.execute("DELETE FROM chunk_vec WHERE chunk_id IN (SELECT id FROM vault_chunks WHERE path = ?1)", params![path])?;
            tx.execute("DELETE FROM vault_chunks WHERE path = ?1", params![path])?;
            for (i, (text, emb)) in chunks.iter().enumerate() {
                tx.execute("INSERT INTO vault_chunks (path, chunk, content, sig) VALUES (?1, ?2, ?3, ?4)", params![path, i as i64, text, sig])?;
                let id = tx.last_insert_rowid();
                tx.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?1, ?2)", params![id, blob(emb)])?;
            }
            tx.commit()
        })
    }

    pub fn drop_note(&self, path: &str) -> rusqlite::Result<()> {
        self.set_note(path, "", &[])
    }

    /// The k nearest memories and the k nearest note chunks, merged, nearest first.
    pub fn search(&self, emb: &[f32], k: usize, memories: bool, notes: bool) -> rusqlite::Result<Vec<Hit>> {
        let q = blob(emb);
        let k = k.clamp(1, 100) as i64;
        self.read(|c| {
            let mut out = Vec::new();
            if memories {
                let mut s = c.prepare(
                    "WITH m AS (SELECT note_id, distance FROM vault_vec WHERE embedding MATCH ?1 AND k = ?2)
                     SELECT m.note_id, n.category, n.content, m.distance FROM m JOIN vault_notes n ON n.id = m.note_id",
                )?;
                let rows = s.query_map(params![q, k], |r| Ok(Hit { kind: "memory", id: r.get(0)?, source: r.get(1)?, content: r.get(2)?, distance: r.get(3)? }))?;
                for h in rows {
                    out.push(h?);
                }
            }
            if notes {
                let mut s = c.prepare(
                    "WITH m AS (SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ?1 AND k = ?2)
                     SELECT m.chunk_id, v.path, v.content, m.distance FROM m JOIN vault_chunks v ON v.id = m.chunk_id",
                )?;
                let rows = s.query_map(params![q, k], |r| Ok(Hit { kind: "note", id: r.get(0)?, source: r.get(1)?, content: r.get(2)?, distance: r.get(3)? }))?;
                for h in rows {
                    out.push(h?);
                }
            }
            out.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            out.truncate(k as usize);
            Ok(out)
        })
    }

    pub fn counts(&self) -> rusqlite::Result<(i64, i64, i64)> {
        self.read(|c| {
            Ok((
                c.query_row("SELECT count(*) FROM vault_notes", [], |r| r.get(0))?,
                c.query_row("SELECT count(DISTINCT path) FROM vault_chunks", [], |r| r.get(0))?,
                c.query_row("SELECT count(*) FROM terminal_sessions", [], |r| r.get(0))?,
            ))
        })
    }

    pub fn vec_version(&self) -> String {
        self.read(|c| c.query_row("SELECT vec_version()", [], |r| r.get(0))).unwrap_or_default()
    }
}
