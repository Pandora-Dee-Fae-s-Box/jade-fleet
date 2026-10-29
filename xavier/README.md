# The Xavier (tier 1): all model compute

The Jetson AGX Xavier runs every model; the OptiPlex only brokers. Four services, all on the LAN:

| Service | Port | Used for | Broker setting |
| --- | --- | --- | --- |
| llama-server | 8080 | Chat (Gemma 4 E2B GGUF), OpenAI-compatible | `LLM_URL=http://XAVIER:8080/v1` |
| Ollama | 11434 | Embeddings (`nomic-embed-text`); decision models too (0.35+) | `OLLAMA_URL=http://XAVIER:11434` |
| Ollaya | 11435 | Decision models (`laya`) | `OLLAYA_URL=http://XAVIER:11435` |
| Vision service | 8090 | YOLO and MobileNet on the DLA cores | `VISION_URL=http://XAVIER:8090` |

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

## Vision service: YOLO and MobileNet on the DLA cores

The Xavier has two NVDLA cores next to the GPU. `vision/` runs YOLO (object detection) on DLA core 0 and MobileNet
(classification) on DLA core 1 as TensorRT engines, so the two run at the same time and leave the GPU to the chat
model.

1. **Export the models** on any machine with PyTorch (a PC is quickest):
   `pip install ultralytics torchvision onnx onnxsim && python3 vision/export_models.py`
   This writes `models/yolov8n.onnx`, `models/mobilenet_v2.onnx` and their label lists. Other YOLO weights work too
   (`--yolo yolov8s.pt`, `--yolo yolo11n.pt`).
2. **Build the engines on the Xavier** (an engine only runs on the TensorRT and device it was built on): copy
   `models/` into `vision/` there and run `./build_engines.sh`. FP16 by default. The log shows which layers the DLA
   could not run; those fall back to the GPU automatically. `PRECISION=int8` is faster but needs a calibration cache
   (`CALIB=…`) for trustworthy accuracy.
3. **Install the service:** `sudo ./install.sh` (in `vision/`). It runs as `jade-vision` under systemd with the
   settings in `/etc/jade-fleet/vision.env`, and needs only what JetPack ships (Python 3.8, numpy, OpenCV, TensorRT).

Through the broker, phones call:

```sh
curl -H "Authorization: Bearer $FLEET_TOKEN" http://OPTIPLEX:8765/v1/vision/detect \
     -H 'Content-Type: image/jpeg' --data-binary @street.jpg
# {"detections": [{"label": "person", "class_id": 0, "confidence": 0.91, "box": [x1, y1, x2, y2]}, …],
#  "width": 1280, "height": 720, "device": "DLA0", "ms": {"decode", "pre", "infer", "post", "total"}}
curl -H "Authorization: Bearer $FLEET_TOKEN" http://OPTIPLEX:8765/v1/vision/classify \
     -H 'Content-Type: application/json' -d '{"image": "<base64>", "top": 5}'
# {"top": [{"label": "tabby cat", "class_id": 281, "prob": 0.79}, …], "device": "DLA1", …}
```

Boxes are in the image's own pixels. Detect takes `conf` (0.25), `iou` (0.45) and `max` (100);
`/v1/vision/models` lists the loaded engines. `python3 vision/test_jade_vision.py` tests everything except TensorRT
itself (stand-in engines) on any machine with numpy and OpenCV.

Memory: the Xavier's 32 GB is shared between CPU and GPU. Gemma 4 E2B Q4 (~2 GB with its context), nomic-embed-text
(~0.3 GB) and laya fit together with room for the vision engines.

Nothing on the Xavier needs the fleet token: only the OptiPlex should reach these ports. If other machines share the
network, allow only the OptiPlex, for example `sudo ufw allow from OPTIPLEX_IP to any port 8080,8090,11434,11435 proto tcp`.
