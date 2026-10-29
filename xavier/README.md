# The Xavier (tier 1): all model compute

The Jetson AGX Xavier runs every model; the OptiPlex only brokers. Four services, all on the LAN:

| Service | Port | Used for | Broker setting |
| --- | --- | --- | --- |
| llama-server | 8080 | Chat (Gemma 4 E2B GGUF), OpenAI-compatible | `LLM_URL=http://XAVIER:8080/v1` |
| Ollama | 11434 | Embeddings (`nomic-embed-text`); decision models too (0.35+) | `OLLAMA_URL=http://XAVIER:11434` |
| Ollaya | 11435 | Decision models (`laya`) | `OLLAYA_URL=http://XAVIER:11435` |
| Vision service | 8090 | YOLO and MobileNet on the DLA cores (coming next) | `VISION_URL=http://XAVIER:8090` |

Set the power mode first: `sudo nvpmodel -m 0 && sudo jetson_clocks` (MAXN).

## llama-server

```sh
llama-server -m gemma-4-e2b-q4.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --parallel 4
```

`--parallel` lets several phones be answered at once (the context is shared between the slots).

## Ollama and Ollaya

Both are built for the Xavier on your side. What the fleet needs from them:

- **Listen on the network**, not only on localhost, so the OptiPlex can reach them: `OLLAMA_HOST=0.0.0.0:11434` for
  Ollama, and the equivalent for Ollaya on 11435.
- **Models:** `nomic-embed-text` in Ollama (768-dimension vectors, the broker's `EMBED_DIM`); `laya` in Ollaya. A
  decision model in Ollama needs Ollama 0.35 or newer (`/v1/systemone`).
- **Keep them loaded** so the first request after a pause is not slow: `OLLAMA_KEEP_ALIVE=-1`,
  `OLLAMA_MAX_LOADED_MODELS=3`.

Decisions go to Ollaya by default (`/v1/systemone` on the broker). To send them to Ollama instead, set
`DECIDE_URL=http://XAVIER:11434` in the broker's settings. Both stay reachable directly through the broker as
`/ollaya/…` and `/ollama/…`.

Check from the OptiPlex:

```sh
curl http://XAVIER:11434/api/embed -d '{"model":"nomic-embed-text","input":["hello"]}' | head -c 120
curl http://XAVIER:11435/api/tags
curl http://XAVIER:8080/v1/models
```

Memory: the Xavier's 32 GB is shared between CPU and GPU. Gemma 4 E2B Q4 (~2 GB with its context), nomic-embed-text
(~0.3 GB) and laya fit together with room for the vision engines.

Nothing on the Xavier needs the fleet token: only the OptiPlex should reach these ports. If other machines share the
network, allow only the OptiPlex, for example `sudo ufw allow from OPTIPLEX_IP to any port 8080,8090,11434,11435 proto tcp`.
