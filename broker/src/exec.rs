//! Real processes on this machine, in the shared vault folder: bash, python3, node (the list is FLEET_SHELLS).
//! - `run`: run-and-return for Jade's tools (exit code, stdout, stderr), with a time limit and a kill by id.
//! - `Pty`: an interactive terminal (a real pseudo-terminal) for the phone's shell screen.
//! Every command runs in its own process group so a time limit or a kill stops its children too.

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const OUT_CAP: usize = 1024 * 1024;

#[derive(Deserialize, Default, Debug)]
pub struct ExecReq {
    /// bash, sh, python3, node … (must be in FLEET_SHELLS)
    pub cmd: String,
    /// Program text, run from a temp file (no command-line length limit)
    #[serde(default)]
    pub script: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory as a vault path ("/" = the vault folder)
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub stdin: Option<String>,
    /// Seconds (default 120, at most EXEC_MAX)
    #[serde(default)]
    pub timeout: Option<f64>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Caller's id for this run, to kill it with /v1/exec/kill
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Default)]
pub struct Running(Mutex<HashMap<String, i32>>);

impl Running {
    pub fn kill(&self, id: &str) -> bool {
        let pg = self.0.lock().unwrap_or_else(|e| e.into_inner()).get(id).copied();
        if let Some(pg) = pg {
            unsafe { libc::killpg(pg, libc::SIGKILL) };
            true
        } else {
            false
        }
    }
}

fn ext(cmd: &str) -> &'static str {
    match cmd {
        "python3" | "python" => ".py",
        "node" => ".js",
        _ => ".sh",
    }
}

fn rand_id() -> String {
    let mut b = [0u8; 8];
    let _ = std::fs::File::open("/dev/urandom").and_then(|mut r| std::io::Read::read_exact(&mut r, &mut b));
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub struct Ctx<'a> {
    pub vault: &'a crate::vault::Disk,
    pub data_dir: &'a Path,
    pub shells: &'a [String],
    pub max: Duration,
    pub device: &'a str,
}

fn base_env(cmd: &mut tokio::process::Command, ctx: &Ctx, cwd: &Path) {
    cmd.env("JADE_VAULT", &ctx.vault.root).env("JADE_DEVICE", ctx.device).env("PWD", cwd).env("PYTHONUNBUFFERED", "1").env("TERM", "dumb");
}

/// Run a program to the end. Answers {code, stdout, stderr, timed_out, killed, truncated, ms}.
pub async fn run(req: ExecReq, ctx: Ctx<'_>, running: Arc<Running>) -> Result<Value, String> {
    if !ctx.shells.iter().any(|s| s == &req.cmd) {
        return Err(format!("{} is not allowed here (FLEET_SHELLS: {})", req.cmd, ctx.shells.join(", ")));
    }
    let cwd = ctx.vault.file(req.cwd.as_deref().unwrap_or("/")).map_err(|e| e.to_string())?;
    if !cwd.is_dir() {
        return Err(format!("{}: no such folder in the vault", req.cwd.unwrap_or_default()));
    }
    let tmpdir = ctx.data_dir.join("run");
    let mut script_file: Option<PathBuf> = None;
    let mut cmd = tokio::process::Command::new(&req.cmd);
    if let Some(src) = &req.script {
        tokio::fs::create_dir_all(&tmpdir).await.map_err(|e| e.to_string())?;
        let f = tmpdir.join(format!("{}{}", rand_id(), ext(&req.cmd)));
        tokio::fs::write(&f, src).await.map_err(|e| e.to_string())?;
        cmd.arg(&f);
        script_file = Some(f);
    }
    cmd.args(&req.args).current_dir(&cwd).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).process_group(0).kill_on_drop(true);
    base_env(&mut cmd, &ctx, &cwd);
    for (k, v) in &req.env {
        cmd.env(k, v);
    }
    let t0 = Instant::now();
    let mut child = cmd.spawn().map_err(|e| format!("{}: {e}", req.cmd))?;
    let pg = child.id().map(|x| x as i32).unwrap_or(0);
    let id = req.id.clone().unwrap_or_else(rand_id);
    running.0.lock().unwrap_or_else(|e| e.into_inner()).insert(id.clone(), pg);
    let mut sin = child.stdin.take();
    let input = req.stdin.clone().unwrap_or_default();
    let feed = async move {
        if let Some(s) = sin.as_mut() {
            let _ = s.write_all(input.as_bytes()).await;
        }
        drop(sin);
    };
    let (mut so, mut se) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let limit = Duration::from_secs_f64(req.timeout.unwrap_or(120.0).max(0.1)).min(ctx.max);
    let work = async {
        let ((o, oc), (e, ec), _) = tokio::join!(read_capped(&mut so), read_capped(&mut se), feed);
        let st = child.wait().await;
        (o, oc, e, ec, st)
    };
    let res = tokio::time::timeout(limit, work).await;
    let timed_out = res.is_err();
    if pg > 0 {
        // the program's own children (a server it started, a pipeline) go with it
        unsafe { libc::killpg(pg, libc::SIGKILL) };
    }
    running.0.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
    if let Some(f) = script_file {
        let _ = tokio::fs::remove_file(f).await;
    }
    let ms = t0.elapsed().as_millis() as u64;
    match res {
        Err(_) => Ok(json!({"code": 124, "stdout": "", "stderr": format!("timed out after {}s\n", limit.as_secs_f64()), "timed_out": true, "killed": true, "truncated": false, "ms": ms, "id": id})),
        Ok((o, oc, e, ec, st)) => {
            use std::os::unix::process::ExitStatusExt;
            let st = st.map_err(|e| e.to_string())?;
            let sig = st.signal();
            let code = st.code().unwrap_or(128 + sig.unwrap_or(0));
            Ok(json!({"code": code, "stdout": String::from_utf8_lossy(&o), "stderr": String::from_utf8_lossy(&e), "timed_out": timed_out, "killed": sig == Some(libc::SIGKILL), "truncated": oc || ec, "ms": ms, "id": id}))
        }
    }
}

async fn read_capped<R: tokio::io::AsyncRead + Unpin>(r: &mut R) -> (Vec<u8>, bool) {
    let mut out = Vec::new();
    let mut buf = [0u8; 16384];
    let mut cut = false;
    loop {
        match r.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if out.len() < OUT_CAP {
                    let take = n.min(OUT_CAP - out.len());
                    out.extend_from_slice(&buf[..take]);
                    cut |= take < n;
                } else {
                    cut = true;
                }
            }
        }
    }
    (out, cut)
}

/// An interactive terminal: the program runs on a pseudo-terminal, like an ssh session.
pub struct Pty {
    pub master: OwnedFd,
    pub pid: i32,
}

pub fn set_size(fd: i32, cols: u16, rows: u16) {
    let ws = libc::winsize { ws_row: rows.max(1), ws_col: cols.max(1), ws_xpixel: 0, ws_ypixel: 0 };
    unsafe { libc::ioctl(fd, libc::TIOCSWINSZ, &ws) };
}

impl Pty {
    pub fn spawn(program: &str, args: &[String], cwd: &Path, env: &[(String, String)], cols: u16, rows: u16) -> std::io::Result<Pty> {
        use std::os::unix::process::CommandExt;
        let (mut m, mut s) = (0i32, 0i32);
        let ws = libc::winsize { ws_row: rows.max(1), ws_col: cols.max(1), ws_xpixel: 0, ws_ypixel: 0 };
        if unsafe { libc::openpty(&mut m, &mut s, std::ptr::null_mut(), std::ptr::null(), &ws) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let master = unsafe { OwnedFd::from_raw_fd(m) };
        let slave = unsafe { OwnedFd::from_raw_fd(s) };
        unsafe { libc::fcntl(m, libc::F_SETFD, libc::FD_CLOEXEC) };
        let mut cmd = std::process::Command::new(program);
        cmd.args(args).current_dir(cwd).env("TERM", "xterm-256color").env("COLORTERM", "truecolor");
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::from(slave.try_clone()?)).stdout(Stdio::from(slave.try_clone()?)).stderr(Stdio::from(slave));
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = cmd.spawn()?;
        let pid = child.id() as i32;
        // reaped by wait() in the reader thread
        std::mem::forget(child);
        Ok(Pty { master, pid })
    }

    pub fn hangup(&self) {
        unsafe {
            libc::killpg(self.pid, libc::SIGHUP);
        }
        let pid = self.pid;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(3));
            unsafe { libc::killpg(pid, libc::SIGKILL) };
        });
    }

    /// Exit code once the program has ended.
    pub fn wait(pid: i32) -> i32 {
        let mut st = 0;
        unsafe { libc::waitpid(pid, &mut st, 0) };
        if libc::WIFEXITED(st) {
            libc::WEXITSTATUS(st)
        } else if libc::WIFSIGNALED(st) {
            128 + libc::WTERMSIG(st)
        } else {
            -1
        }
    }

    pub fn fd(&self) -> i32 {
        self.master.as_raw_fd()
    }
}
