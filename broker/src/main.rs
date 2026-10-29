//! jade-fleet: the broker on the OptiPlex. Settings come from the environment (see README / deploy/fleet.env).

use jade_fleet::config::Config;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("jade-fleet {}\nThe Jade Fleet broker. Configure with environment variables (see /etc/jade-fleet/fleet.env).\n  jade-fleet           serve\n  jade-fleet token     print the fleet token (made on first start)", env!("CARGO_PKG_VERSION"));
        return;
    }
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_env("FLEET_LOG").unwrap_or_else(|_| "jade_fleet=info".into())).init();
    let mut cfg = Config::from_env();
    if args.first().map(String::as_str) == Some("token") {
        match cfg.resolve_token() {
            Ok(()) => println!("{}", cfg.token),
            Err(e) => {
                eprintln!("token: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    let bind = cfg.bind.clone();
    let app = match jade_fleet::build(cfg).await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("jade-fleet: {e}");
            std::process::exit(1);
        }
    };
    let listener = match tokio::net::TcpListener::bind(&bind).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("jade-fleet: cannot listen on {bind}: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("listening on {bind}; vault {}; LLM {}; embeddings {} via {}", app.cfg.vault_dir.display(), app.cfg.llm_url, app.cfg.embed_model, app.cfg.ollama_url);
    let stop = async {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal");
        tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
    };
    if let Err(e) = jade_fleet::serve(app, listener, stop).await {
        eprintln!("jade-fleet: {e}");
        std::process::exit(1);
    }
}
