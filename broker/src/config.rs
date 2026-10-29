//! Settings, from environment variables (systemd reads them from /etc/jade-fleet/fleet.env).

use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Config {
    /// Address to listen on, e.g. 0.0.0.0:8765
    pub bind: String,
    /// Folder for the database and the token file
    pub data_dir: PathBuf,
    /// The shared vault folder (ordinary files, like the phone app's JadeVault folder)
    pub vault_dir: PathBuf,
    /// Bearer token every phone must send. Empty: read or create data_dir/token.
    pub token: String,
    /// Ollama on the Xavier: embeddings and decision models (all model compute is on the Xavier)
    pub ollama_url: String,
    pub embed_model: String,
    pub embed_dim: usize,
    /// Prefixes some embedding models want (nomic-embed-text: "search_query: " / "search_document: ")
    pub embed_query_prefix: String,
    pub embed_doc_prefix: String,
    /// OpenAI-compatible server on the Xavier (llama-server), base URL ending in /v1
    pub llm_url: String,
    pub llm_model: String,
    pub llm_temperature: f64,
    /// Notes put into a WebSocket chat's system prompt
    pub rag_k: usize,
    /// Chat turns a WebSocket chat keeps per connection
    pub history_turns: usize,
    pub system_prompt: String,
    /// Index the vault's notes for vector search
    pub index_vault: bool,
    /// Characters per indexed chunk
    pub chunk_chars: usize,
    pub db_readers: usize,
    /// Ollaya on the Xavier (decision models such as laya)
    pub ollaya_url: String,
    /// Where /v1/systemone and /api/decide go: Ollaya (default) or Ollama 0.35+, both on the Xavier
    pub decide_url: String,
    /// The vision service on the Xavier (YOLO / MobileNet on the DLA cores)
    pub vision_url: String,
    /// Folder with the browser client (index.html …), served at /
    pub web_dir: PathBuf,
    /// Interpreters phones may run on this machine
    pub shells: Vec<String>,
    /// Longest a run-and-return command may take
    pub exec_max: Duration,
    pub embed_timeout: Duration,
    pub llm_timeout: Duration,
}

fn env(k: &str, d: &str) -> String {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| d.to_string())
}

fn envn<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(d)
}

impl Config {
    pub fn from_env() -> Config {
        let data_dir = PathBuf::from(env("FLEET_DATA", "/var/lib/jade-fleet"));
        let vault_dir = std::env::var("FLEET_VAULT").ok().filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(|| data_dir.join("vault"));
        let embed_model = env("EMBED_MODEL", "nomic-embed-text");
        let nomic = embed_model.starts_with("nomic-embed");
        Config {
            bind: env("FLEET_BIND", "0.0.0.0:8765"),
            data_dir,
            vault_dir,
            token: env("FLEET_TOKEN", ""),
            ollama_url: env("OLLAMA_URL", "http://192.168.1.100:11434").trim_end_matches('/').to_string(),
            embed_query_prefix: std::env::var("EMBED_QUERY_PREFIX").unwrap_or_else(|_| if nomic { "search_query: ".into() } else { String::new() }),
            embed_doc_prefix: std::env::var("EMBED_DOC_PREFIX").unwrap_or_else(|_| if nomic { "search_document: ".into() } else { String::new() }),
            embed_model,
            embed_dim: envn("EMBED_DIM", 768),
            llm_url: env("LLM_URL", "http://192.168.1.100:8080/v1").trim_end_matches('/').trim_end_matches("/chat/completions").to_string(),
            llm_model: env("LLM_MODEL", "gemma4-e2b-custom"),
            llm_temperature: envn("LLM_TEMPERATURE", 0.7),
            rag_k: envn("RAG_K", 3),
            history_turns: envn("HISTORY_TURNS", 8),
            system_prompt: env("SYSTEM_PROMPT", "You are Jade, an edge-hosted AI assistant."),
            index_vault: env("INDEX_VAULT", "1") != "0",
            chunk_chars: envn("CHUNK_CHARS", 1200usize).clamp(200, 8000),
            db_readers: envn("DB_READERS", 4usize).clamp(1, 32),
            ollaya_url: env("OLLAYA_URL", "http://192.168.1.100:11435").trim_end_matches('/').to_string(),
            decide_url: std::env::var("DECIDE_URL").ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| env("OLLAYA_URL", "http://192.168.1.100:11435")).trim_end_matches('/').to_string(),
            vision_url: env("VISION_URL", "http://192.168.1.100:8090").trim_end_matches('/').to_string(),
            web_dir: PathBuf::from(env("FLEET_WEB", "/usr/share/jade-fleet/web")),
            shells: env("FLEET_SHELLS", "bash,sh,python3,node").split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
            exec_max: Duration::from_secs(envn("EXEC_MAX", 1800)),
            embed_timeout: Duration::from_secs(envn("EMBED_TIMEOUT", 30)),
            llm_timeout: Duration::from_secs(envn("LLM_TIMEOUT", 300)),
        }
    }

    /// The token phones use: FLEET_TOKEN, or one made on first start and kept in data_dir/token (mode 600).
    pub fn resolve_token(&mut self) -> std::io::Result<()> {
        if !self.token.is_empty() {
            return Ok(());
        }
        let f = self.data_dir.join("token");
        if let Ok(t) = std::fs::read_to_string(&f) {
            if !t.trim().is_empty() {
                self.token = t.trim().to_string();
                return Ok(());
            }
        }
        std::fs::create_dir_all(&self.data_dir)?;
        let mut b = [0u8; 24];
        std::fs::File::open("/dev/urandom").and_then(|mut r| std::io::Read::read_exact(&mut r, &mut b))?;
        self.token = b.iter().map(|x| format!("{x:02x}")).collect();
        std::fs::write(&f, format!("{}\n", self.token))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }
}
