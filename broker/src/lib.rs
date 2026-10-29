//! Jade Fleet broker (tier 2, the OptiPlex): the fleet's shared vault, real shells, vector memory and search, and
//! the way through to the models, all on the Xavier (llama-server, Ollama, Ollaya, the vision service).

pub mod api;
pub mod chat;
pub mod config;
pub mod db;
pub mod embed;
pub mod exec;
pub mod index;
pub mod upstream;
pub mod vault;
pub mod watch;

use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};

pub struct State {
    pub cfg: Arc<config::Config>,
    pub db: Arc<db::Db>,
    pub vault: Arc<vault::Disk>,
    pub up: upstream::Upstream,
    pub emb: embed::Embedder,
    pub events: broadcast::Sender<Arc<serde_json::Value>>,
    pub writers: Arc<watch::Writers>,
    pub running: Arc<exec::Running>,
    pub progress: Arc<index::Progress>,
    pub devices: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    pub started: std::time::Instant,
    _watcher: std::sync::Mutex<Option<notify::RecommendedWatcher>>,
}

pub type App = Arc<State>;

/// Open everything and start the background work (folder watcher, indexer). Must run inside a tokio runtime.
pub async fn build(mut cfg: config::Config) -> Result<App, String> {
    cfg.resolve_token().map_err(|e| format!("token: {e}"))?;
    std::fs::create_dir_all(&cfg.data_dir).map_err(|e| format!("{}: {e}", cfg.data_dir.display()))?;
    let vault = vault::Disk::open(cfg.vault_dir.to_str().ok_or("vault path is not UTF-8")?).map_err(|e| format!("vault {}: {e}", cfg.vault_dir.display()))?;
    let (db, fresh) = db::Db::open(&cfg.data_dir.join("jade_vault.db"), cfg.embed_dim, &cfg.embed_model, cfg.db_readers).map_err(|e| format!("database: {e}"))?;
    if fresh {
        tracing::info!("vector index (re)built for {} ({} dims)", cfg.embed_model, cfg.embed_dim);
    }
    let cfg = Arc::new(cfg);
    let up = upstream::Upstream::new();
    let emb = embed::Embedder::new(up.clone(), cfg.clone());
    let (events, _) = broadcast::channel(1024);
    let writers = Arc::new(watch::Writers::default());
    let progress = Arc::new(index::Progress::default());
    let (itx, irx) = mpsc::unbounded_channel();
    let watcher = watch::start(vault.canon_root(), events.clone(), itx, writers.clone()).map_err(|e| format!("watch {}: {e}", cfg.vault_dir.display()))?;
    if cfg.index_vault {
        let ix = index::Indexer { db: db.clone(), emb: emb.clone(), root: vault.canon_root(), chunk_chars: cfg.chunk_chars, progress: progress.clone() };
        tokio::spawn(ix.run(irx, true));
    } else {
        tokio::spawn(async move {
            let mut irx = irx;
            while irx.recv().await.is_some() {}
        });
    }
    Ok(Arc::new(State {
        cfg,
        db,
        vault: Arc::new(vault),
        up,
        emb,
        events,
        writers,
        running: Arc::new(exec::Running::default()),
        progress,
        devices: Default::default(),
        started: std::time::Instant::now(),
        _watcher: std::sync::Mutex::new(Some(watcher)),
    }))
}

/// Serve until the listener closes or the future `stop` completes.
pub async fn serve(app: App, listener: tokio::net::TcpListener, stop: impl std::future::Future<Output = ()> + Send + 'static) -> std::io::Result<()> {
    let router = api::router(app);
    axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).with_graceful_shutdown(stop).await
}
