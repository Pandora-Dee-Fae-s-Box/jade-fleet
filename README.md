# Jade Fleet

Jade Vault, spread across a fleet: phones and browsers are the terminals, a small PC holds what they share, and a
Jetson does all the model compute.

```
 Phones (Fleet APK) · any browser                 TIER 3  Jade's agent, avatar, phone tools (TTS, screen, Sentry)
            │  HTTP + WebSocket, one fleet token
 OptiPlex 3020 Micro (Debian) · jade-fleet        TIER 2  shared vault, real shells, memory and search, routing
   └─ SQLite (WAL) + sqlite-vec, compiled into the broker
            │  HTTP on the LAN
 Jetson AGX Xavier                                TIER 1  all model compute
   ├─ llama-server: Gemma 4 E2B (chat)
   ├─ Ollama: nomic-embed-text (embeddings), decision models
   ├─ Ollaya: laya (decisions)
   └─ vision service: YOLO, MobileNet on the DLA cores
```

## The broker (`broker/`, Rust)

One binary on the OptiPlex, async (tokio) throughout, so many phones are served at once.

| What | How |
| --- | --- |
| **Shared vault** | One folder of ordinary files with the same rules as the Jade Vault app's vault folder: writes land in a temp file and are renamed into place, nothing can escape the root. Phones read and write it over HTTP. |
| **Live sync** | The vault folder is watched. Any change (a phone's write, a command run here, scp, an editor) is batched and pushed to every connected phone, saying which phone made it so a phone can ignore its own echo. |
| **Real shells** | `bash`, `python3`, `node` (the list is `FLEET_SHELLS`) run as real processes on the OptiPlex, in the vault folder. Run-and-return for Jade's tools, with a time limit and kill by id; an interactive terminal on a real pseudo-terminal for the shell screen. Each command is its own process group, so a time limit or kill stops its children too. |
| **Memory and RAG** | Memories and every note in the vault are embedded by Ollama on the Xavier and searched here with sqlite-vec. Notes are re-indexed in the background when they change; deleted or moved notes leave the index. One writer and a pool of readers, so searches run while the indexer writes. Switching embedding model or dimension rebuilds the vectors. |
| **Model routing** | Straight through, streaming: chat to the Xavier's llama-server (OpenAI-compatible), Ollama's API to Ollama, decisions (`/v1/systemone`) to Ollaya or Ollama, vision to the Xavier's vision service. The fleet token never leaves the broker. |
| **WebSocket chat** | The original design's thin-terminal chat: a message gets the nearest memories and notes put into the system prompt, and the Xavier's reply streams back token by token. Per-connection history, cancel, many phones at once. |
| **Browser client** | Served at `/` from `FLEET_WEB`, so it is same-origin with the API (no CORS). |

## Install on the OptiPlex

On Debian 12 or 13, from a clone of this repository:

```sh
sudo deploy/install.sh
```

This installs build tools, builds the broker (Rust is installed for your user if it is missing), creates a `jade`
system user, and starts the `jade-fleet` service. The OptiPlex runs no models; set up the Xavier with
[`xavier/README.md`](xavier/README.md). It prints the fleet
token at the end; print it again with `sudo -u jade jade-fleet token`.

Settings are in `/etc/jade-fleet/fleet.env` (see [`deploy/fleet.env`](deploy/fleet.env)): the Xavier's address
(`LLM_URL`, `OLLAMA_URL`, `OLLAYA_URL`, `VISION_URL`), the embedding model, the vault folder, which shells phones may
run. Restart after changes: `sudo systemctl restart jade-fleet`. Logs: `journalctl -u jade-fleet -f`.

**Security.** Anyone with the fleet token has a shell on the OptiPlex as the `jade` user, can read and change the
vault, and can use the models. Keep the token private, keep the broker on your own network (it speaks plain HTTP),
and do not give the `jade` user sudo.

## API

All endpoints except `/health` and the browser client need the token: `Authorization: Bearer TOKEN`, or `?token=TOKEN`
where a header cannot be set (a browser WebSocket). A phone names itself with `X-Jade-Device: NAME` or `?device=NAME`.
Errors are `{"ok": false, "error": "…"}`.

**Status**

| | |
| --- | --- |
| `GET /health` | `{"ok": true}`, no token. |
| `GET /v1/info` | Version, vault folder and free space, memory and index counts, devices online and known, model addresses, sqlite-vec version. |

**Vault** (the same contract as the Jade Vault app's vault folder)

| | |
| --- | --- |
| `GET /v1/vault/scan` | `[[path, isDir, size, mtime], …]` for the whole vault. |
| `GET /v1/vault/file?path=P&off=N&len=N` | The bytes. |
| `PUT /v1/vault/file?path=P&first=1&last=1` | Write the request body. Large files can be sent in chunks: `first=1` starts, `last=1` finishes and renames into place. Answers `{"mtime": …}`. |
| `POST /v1/vault/op` | `{"op": "mkdir" \| "delete" \| "move" \| "touch", "path": …, "to": …, "mtime": …}` |
| `GET /v1/events` (WebSocket) | `{"type":"hello","online":[…]}`, then `{"type":"changed","paths":[…],"by":{path: device}}`, `{"type":"device","device":…,"online":bool}`, and `{"type":"resync"}` if the phone fell behind (rescan). |

**Memory and search**

| | |
| --- | --- |
| `POST /v1/memory` | `{"content": …, "category": …}` → `{"id": …}` |
| `DELETE /v1/memory/ID` | |
| `GET /v1/search?q=TEXT&k=8&kind=all\|memory\|note` | Nearest memories and note chunks: `{"hits": [{"kind", "id", "source", "content", "distance"}]}` (`source` is the note's vault path, or the memory's category). |

**Processes on the OptiPlex**

| | |
| --- | --- |
| `POST /v1/exec` | `{"cmd": "bash" \| "python3" \| "node" …, "script": "…", "args": […], "cwd": "/vault/path", "stdin": "…", "timeout": 120, "env": {}, "id": "…"}` → `{"code", "stdout", "stderr", "timed_out", "killed", "truncated", "ms"}`. Output is kept up to 1 MB each. Commands see `JADE_VAULT` and `JADE_DEVICE`. |
| `POST /v1/exec/kill` | `{"id": …}` |
| `GET /v1/term?cmd=bash&cwd=/&cols=80&rows=24` (WebSocket) | A terminal. Binary frames carry the bytes both ways; the phone sends `{"type":"resize","cols":…,"rows":…}` as text; the broker ends with `{"type":"exit","code":N}`. Closing the socket hangs up the shell. |

**Models** (passed through as they are, streaming included)

| | |
| --- | --- |
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | The Xavier (`LLM_URL`). A phone can use the broker as an OpenAI-compatible provider. |
| `/llm/…` | Anything else on the Xavier's llama-server (`/llm/health`, `/llm/props` …). |
| `/api/…`, `/ollama/…` | Ollama on the Xavier (`/api/chat`, `/api/tags`, `/api/embed` …). A phone can use the broker as its Ollama host. |
| `/ollaya/…` | Ollaya on the Xavier (`/ollaya/v1/systemone`, `/ollaya/api/tags` …). |
| `/v1/systemone`, `/api/decide` | Decisions on `DECIDE_URL`: Ollaya by default, or Ollama 0.35+. Either is also reachable directly as `/ollaya/v1/systemone` or `/ollama/v1/systemone`. |
| `/v1/vision/…` | The vision service on the Xavier (`VISION_URL`). |

**WebSocket chat** `GET /ws/DEVICE`

| Phone → broker | Broker → phone |
| --- | --- |
| `{"message": "…", "id": …}` | `{"type":"context","hits":[{kind, source, distance}]}` |
| `{"type": "cancel"}` | `{"device_id", "type":"token_delta", "delta", "avatar_expression"}` … |
| `{"type": "reset"}` (forget the history) | `{"type":"done","cancelled":bool,"ms":N}` |
| `{"type": "remember", "content": …, "category": …}` | `{"type":"remembered","memory_id":N}` |
| `{"type": "ping"}` | `{"type":"warning" \| "error","message":…}`, `{"type":"pong"}` |

Every reply carries the request's `id`. One reply streams at a time per connection; another message meanwhile gets
`{"type":"error","code":"busy"}`.

## Develop

```sh
cargo build                 # sqlite-vec is compiled in from broker/vendor/sqlite-vec (AVX2 on x86_64, NEON on aarch64)
cargo test                  # unit tests + end-to-end tests against stand-in Ollama / Xavier / vision servers
FLEET_DATA=./data FLEET_TOKEN=dev LLM_URL=http://xavier:8080/v1 cargo run
```

The end-to-end tests use real SQLite + sqlite-vec, real `bash` / `python3` / `node`, a real pseudo-terminal and real
WebSockets: vault writes and live events between two phones, commands and their time limits and kills, the terminal
(size, resize, exit code), memories and note indexing (including moved and deleted notes), the chat (context, streamed
reply, history, busy, cancel), five phones streaming at once, and every pass-through route. Building for Debian 12
needs to happen on Debian 12 (a binary built on a newer system needs a newer glibc); `deploy/install.sh` builds there.

## Status

- [x] Broker: shared vault, live sync, real shells and terminal, memory and vault RAG, model / decision / vision routing, WebSocket chat
- [x] Vision service on the Xavier: YOLO and MobileNet as TensorRT engines on the two DLA cores (`/v1/vision/detect`, `/v1/vision/classify`), see [`xavier/README.md`](xavier/README.md). Tested here with stand-in engines; the DLA itself only runs on the Xavier.
- [ ] Fleet client (APK and browser): Jade Vault's agent, tools and avatar with the vault, shell, memory and models on the broker
- [ ] Ollama as a decision provider in the client, alongside Ollaya (both on the Xavier)

## Licenses

Jade Fleet is MIT. `broker/vendor/sqlite-vec` is sqlite-vec 0.1.9 by Alex Garcia (MIT or Apache-2.0, see its license
files). SQLite is public domain.
