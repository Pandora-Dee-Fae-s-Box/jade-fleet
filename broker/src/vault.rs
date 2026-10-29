//! The fleet's shared vault: vault paths ("/notes/a.md") map onto ordinary files under one folder on the OptiPlex.
//! Same rules and messages as the phone app's vault folder (jade-vault rust/android/src/disk.rs), so the app on a
//! phone cannot tell the difference: writes go to a temp file and are renamed into place, nothing escapes the root.

use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const TMP: &str = ".jvtmp";

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::other(msg.into())
}

pub fn mtime_ms(m: &fs::Metadata) -> i64 {
    m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// The canonical form of a path that may not exist yet: the longest existing prefix canonicalised, the rest appended
/// (what java.io.File.getCanonicalPath does).
fn canonical(p: &Path) -> PathBuf {
    let mut existing = p.to_path_buf();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(c) = existing.canonicalize() {
            let mut out = c;
            for r in rest.iter().rev() {
                out.push(r);
            }
            return out;
        }
        match (existing.file_name().map(|s| s.to_os_string()), existing.parent()) {
            (Some(n), Some(par)) => {
                rest.push(n);
                existing = par.to_path_buf();
            }
            _ => return p.to_path_buf(),
        }
    }
}

pub struct Disk {
    pub root: PathBuf,
    canon: PathBuf,
}

impl Disk {
    pub fn open(root: &str) -> io::Result<Disk> {
        let r = PathBuf::from(root);
        if !r.is_dir() {
            fs::create_dir_all(&r).map_err(|_| err(format!("cannot create {root}")))?;
        }
        let canon = r.canonicalize()?;
        Ok(Disk { root: r, canon })
    }

    /// The root folder with links resolved (what the folder watcher reports paths under).
    pub fn canon_root(&self) -> PathBuf {
        self.canon.clone()
    }

    /// Vault path to a file inside the root. Rejects anything that could escape it.
    pub fn file(&self, vp: &str) -> io::Result<PathBuf> {
        let bad = || err(format!("bad path: {vp}"));
        if !vp.starts_with('/') {
            return Err(bad());
        }
        let mut f = self.root.clone();
        if vp != "/" {
            for s in vp[1..].split('/') {
                if s.is_empty() || s == "." || s == ".." || s.contains('\0') {
                    return Err(bad());
                }
                f.push(s);
            }
        }
        let c = canonical(&f);
        if c != self.canon && !c.starts_with(&self.canon) {
            return Err(err(format!("outside the vault: {vp}")));
        }
        // starts_with compares whole components, so /vault2 never passes for /vault
        if c.components().any(|x| matches!(x, Component::ParentDir)) {
            return Err(bad());
        }
        Ok(f)
    }

    /// [[path,isDir,size,mtime],...] for everything under the root (not the root itself), sorted like Java's
    /// File[] sort. Abandoned temp files are removed. Sub-folders are walked on all cores.
    pub fn scan(&self) -> io::Result<String> {
        let mut rows: Vec<(String, bool, u64, i64)> = Vec::new();
        walk(&self.root, "", &mut rows, true);
        let arr: Vec<Value> = rows.into_iter().map(|(p, d, s, m)| json!([p, if d { 1 } else { 0 }, s, m])).collect();
        Ok(Value::Array(arr).to_string())
    }

    /// Up to len bytes starting at off (empty at end of file).
    pub fn read(&self, vp: &str, off: u64, len: usize) -> io::Result<Vec<u8>> {
        let f = self.file(vp)?;
        if !f.is_file() {
            return Err(err(format!("no such file: {vp}")));
        }
        let mut file = File::open(&f)?;
        let size = file.metadata()?.len();
        let n = (len as u64).min(size.saturating_sub(off)) as usize;
        let mut out = vec![0u8; n];
        file.seek(SeekFrom::Start(off))?;
        file.read_exact(&mut out)?;
        Ok(out)
    }

    /// Write a file in chunks. The first chunk starts a temp file next to the target, the last renames it into
    /// place, so a crash never leaves a half-written note. Returns the final mtime on the last chunk, else 0.
    pub fn put(&self, vp: &str, bytes: &[u8], first: bool, last: bool, writer: &str) -> io::Result<i64> {
        let f = self.file(vp)?;
        let mut tmp = f.clone().into_os_string();
        tmp.push(TMP);
        let w: String = writer.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').take(40).collect();
        if !w.is_empty() {
            tmp.push(".");
            tmp.push(&w);
        }
        let tmp = PathBuf::from(tmp);
        if first {
            let par = f.parent().ok_or_else(|| err(format!("bad path: {vp}")))?;
            if !par.is_dir() {
                fs::create_dir_all(par).map_err(|_| err(format!("cannot create {}", par.display())))?;
            }
            if f.is_dir() {
                return Err(err(format!("is a directory: {vp}")));
            }
        }
        {
            let mut o = OpenOptions::new().create(true).write(true).append(!first).truncate(first).open(&tmp)?;
            o.write_all(bytes)?;
            o.flush()?;
            let _ = o.sync_all();
        }
        if !last {
            return Ok(0);
        }
        if f.exists() && fs::remove_file(&f).is_err() {
            return Err(err(format!("cannot replace {vp}")));
        }
        fs::rename(&tmp, &f).map_err(|_| err(format!("cannot finish {vp}")))?;
        Ok(mtime_ms(&fs::metadata(&f)?))
    }

    pub fn mkdir(&self, vp: &str) -> io::Result<()> {
        let f = self.file(vp)?;
        if f.is_dir() {
            return Ok(());
        }
        fs::create_dir_all(&f).map_err(|_| err(format!("cannot create {vp}")))
    }

    pub fn delete(&self, vp: &str) -> io::Result<()> {
        if vp == "/" {
            return Err(err("refusing to delete the vault root"));
        }
        let f = self.file(vp)?;
        if f.symlink_metadata().is_ok() {
            delete_tree(&f)?;
        }
        Ok(())
    }

    pub fn move_(&self, a: &str, b: &str) -> io::Result<()> {
        let (fa, fb) = (self.file(a)?, self.file(b)?);
        if !fa.exists() {
            return Err(err(format!("no such file: {a}")));
        }
        if let Some(par) = fb.parent() {
            if !par.is_dir() {
                fs::create_dir_all(par).map_err(|_| err(format!("cannot create {}", par.display())))?;
            }
        }
        if fb.exists() {
            delete_tree(&fb)?;
        }
        if fs::rename(&fa, &fb).is_err() {
            copy_tree(&fa, &fb)?;
            delete_tree(&fa)?;
        }
        Ok(())
    }

    pub fn touch(&self, vp: &str, mtime: i64) -> io::Result<i64> {
        let f = self.file(vp)?;
        if !f.exists() {
            return Err(err(format!("no such file: {vp}")));
        }
        let t = UNIX_EPOCH + Duration::from_millis(mtime.max(0) as u64);
        if f.is_file() {
            let _ = File::options().write(true).open(&f).and_then(|h| h.set_modified(t));
        } else if let Ok(h) = File::open(&f) {
            let _ = h.set_modified(t);
        }
        Ok(mtime_ms(&fs::metadata(&f)?))
    }

    /// Bytes free on the volume holding the vault.
    pub fn free(&self) -> i64 {
        free_bytes(&self.root)
    }

    /// Empty everything under the root (full-backup restore).
    pub fn clear(&self) -> io::Result<()> {
        for k in fs::read_dir(&self.root)?.flatten() {
            delete_tree(&k.path())?;
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        fs::read_dir(&self.root).map(|mut r| r.next().is_none()).unwrap_or(true)
    }

    /// Copy everything from another vault folder that is not already here, then empty the other one.
    pub fn merge_from(&self, other: &Disk) -> io::Result<i32> {
        let n = copy_missing(&other.root, &self.root)?;
        other.clear()?;
        Ok(n)
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(&self.root).map(|r| r.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
        v.sort();
        v
    }

}

/// Map over items on all cores (keeping their order).
pub fn parallel<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8);
    if items.len() < 16 || n == 1 {
        return items.iter().map(&f).collect();
    }
    let chunk = items.len().div_ceil(n);
    std::thread::scope(|s| {
        let hs: Vec<_> = items.chunks(chunk).map(|c| s.spawn(|| c.iter().map(&f).collect::<Vec<R>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    })
}

/// Java's File[] sort: by path, as UTF-16 code units.
fn java_cmp(a: &std::ffi::OsStr, b: &std::ffi::OsStr) -> std::cmp::Ordering {
    let (a, b) = (a.to_string_lossy(), b.to_string_lossy());
    a.encode_utf16().cmp(b.encode_utf16())
}

fn walk(dir: &Path, vp: &str, out: &mut Vec<(String, bool, u64, i64)>, top: bool) {
    let mut kids: Vec<fs::DirEntry> = match fs::read_dir(dir) {
        Ok(r) => r.flatten().collect(),
        Err(_) => return,
    };
    kids.sort_by(|a, b| java_cmp(&a.file_name(), &b.file_name()));
    // at the top, each sub-folder is walked on its own thread; the rows are put back in order afterwards
    let subs: Vec<Vec<(String, bool, u64, i64)>> = if top {
        let jobs: Vec<(PathBuf, String)> = kids.iter().filter(|k| k.file_type().map(|t| t.is_dir()).unwrap_or(false)).map(|k| (k.path(), format!("{vp}/{}", k.file_name().to_string_lossy()))).collect();
        parallel(&jobs, |(p, v)| {
            let mut rows = Vec::new();
            walk(p, v, &mut rows, false);
            rows
        })
    } else {
        Vec::new()
    };
    let mut subs = subs.into_iter();
    for k in kids {
        let name = k.file_name().to_string_lossy().into_owned();
        let meta = match k.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if name.contains(TMP) {
            // only abandoned ones; a write may be in flight
            if SystemTime::now().duration_since(meta.modified().unwrap_or(SystemTime::now())).map(|d| d.as_millis() > 3_600_000).unwrap_or(false) {
                let _ = fs::remove_file(k.path());
            }
            continue;
        }
        let p = format!("{vp}/{name}");
        let ft = k.file_type();
        if ft.as_ref().map(|t| t.is_dir()).unwrap_or(false) {
            out.push((p.clone(), true, 0, mtime_ms(&meta)));
            if top {
                out.extend(subs.next().unwrap_or_default());
            } else {
                walk(&k.path(), &p, out, false);
            }
        } else if ft.as_ref().map(|t| t.is_symlink()).unwrap_or(false) {
            // a link to a folder is not followed; a link to a file counts as the file
            if let Ok(m) = fs::metadata(k.path()) {
                if m.is_file() {
                    out.push((p, false, m.len(), mtime_ms(&m)));
                }
            }
        } else if meta.is_file() {
            out.push((p, false, meta.len(), mtime_ms(&meta)));
        }
    }
}

fn delete_tree(f: &Path) -> io::Result<()> {
    let m = f.symlink_metadata()?;
    if m.is_dir() {
        for k in fs::read_dir(f)?.flatten() {
            delete_tree(&k.path())?;
        }
        fs::remove_dir(f).map_err(|_| err(format!("cannot delete {}", f.display())))
    } else {
        fs::remove_file(f).map_err(|_| err(format!("cannot delete {}", f.display())))
    }
}

fn copy_tree(a: &Path, b: &Path) -> io::Result<()> {
    if a.is_dir() {
        if !b.is_dir() {
            fs::create_dir_all(b).map_err(|_| err(format!("cannot create {}", b.display())))?;
        }
        for k in fs::read_dir(a)?.flatten() {
            copy_tree(&k.path(), &b.join(k.file_name()))?;
        }
        Ok(())
    } else {
        copy_file(a, b)
    }
}

fn copy_file(a: &Path, b: &Path) -> io::Result<()> {
    fs::copy(a, b)?;
    if let Ok(m) = fs::metadata(a).and_then(|m| m.modified()) {
        let _ = File::options().write(true).open(b).and_then(|h| h.set_modified(m));
    }
    Ok(())
}

fn copy_missing(from: &Path, to: &Path) -> io::Result<i32> {
    let mut n = 0;
    let kids = match fs::read_dir(from) {
        Ok(r) => r,
        Err(_) => return Ok(0),
    };
    for k in kids.flatten() {
        let name = k.file_name();
        if name.to_string_lossy().contains(TMP) {
            continue;
        }
        let t = to.join(&name);
        if k.path().is_dir() {
            if !t.is_dir() {
                fs::create_dir_all(&t).map_err(|_| err(format!("cannot create {}", t.display())))?;
            }
            n += copy_missing(&k.path(), &t)?;
        } else if !t.exists() {
            copy_file(&k.path(), &t)?;
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(unix)]
fn free_bytes(p: &Path) -> i64 {
    use std::os::unix::ffi::OsStrExt;
    let c = match std::ffi::CString::new(p.as_os_str().as_bytes()) {
        Ok(c) => c,
        Err(_) => return 0,
    };
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return 0;
    }
    (s.f_bavail as i64) * (s.f_frsize as i64)
}

#[cfg(not(unix))]
fn free_bytes(_p: &Path) -> i64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_writes_and_escapes() {
        let d = std::env::temp_dir().join(format!("jadedisk-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        let disk = Disk::open(d.to_str().unwrap()).unwrap();
        assert!(disk.is_empty());
        assert!(disk.put("/a/b.md", b"he", true, false, "p1").unwrap() == 0);
        assert!(disk.put("/a/b.md", b"llo [[c]] #x", false, true, "p1").unwrap() > 0);
        assert_eq!(disk.read("/a/b.md", 0, 100).unwrap(), b"hello [[c]] #x");
        assert_eq!(disk.read("/a/b.md", 2, 3).unwrap(), b"llo");
        assert!(disk.file("/../x").is_err() && disk.file("a").is_err() && disk.file("/a//b").is_err());
        disk.mkdir("/z").unwrap();
        disk.move_("/a/b.md", "/z/c.md").unwrap();
        let scan: Value = serde_json::from_str(&disk.scan().unwrap()).unwrap();
        let paths: Vec<&str> = scan.as_array().unwrap().iter().map(|r| r[0].as_str().unwrap()).collect();
        assert_eq!(paths, vec!["/a", "/z", "/z/c.md"]);
        assert!(disk.delete("/").is_err());
        disk.delete("/z").unwrap();
        assert_eq!(disk.names(), vec!["a"]);
        let _ = fs::remove_dir_all(&d);
    }
}
