#!/bin/sh
# Build TensorRT engines for the Xavier's two DLA cores (NVDLA). Run on the Xavier, in this folder, after copying
# models/ from export_models.py:
#   ./build_engines.sh            FP16 (no calibration needed)
#   PRECISION=int8 ./build_engines.sh   INT8 without calibration data: faster, accuracy NOT guaranteed; use a
#                                        calibration cache for real use (CALIB=cache.bin)
# YOLO goes to DLA core 0 and MobileNet to DLA core 1, so the two run at the same time. Layers the DLA cannot run
# fall back to the GPU (--allowGPUFallback); trtexec's log lists them.
set -eu
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
PRECISION=${PRECISION:-fp16}
YOLO=${YOLO:-models/yolov8n.onnx}
MOBILENET=${MOBILENET:-models/mobilenet_v2.onnx}
[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC (JetPack installs it there; set TRTEXEC=…)"; exit 1; }
mkdir -p engines logs
flags="--$PRECISION --allowGPUFallback --memPoolSize=workspace:1024"
[ -n "${CALIB:-}" ] && flags="$flags --calib=$CALIB"

build() {   # onnx  engine  dla-core
  echo "== $2 on DLA $3 ($PRECISION)"
  $TRTEXEC --onnx="$1" --saveEngine="$2" --useDLACore="$3" $flags --verbose > "logs/$(basename "$2").log" 2>&1 || {
    echo "failed: see logs/$(basename "$2").log"; tail -20 "logs/$(basename "$2").log"; exit 1; }
  grep -iE "running on (dla|gpu)|layers? (are|is) running|DLA|GPU fallback" "logs/$(basename "$2").log" | grep -iE "fall|gpu|layers" | head -8 || true
  $TRTEXEC --loadEngine="$2" --useDLACore="$3" --iterations=50 --warmUp=500 2>&1 | grep -E "Throughput|Latency: min" | head -2
}
build "$YOLO" "engines/$(basename "$YOLO" .onnx)_dla0_$PRECISION.engine" 0
build "$MOBILENET" "engines/$(basename "$MOBILENET" .onnx)_dla1_$PRECISION.engine" 1
echo "Engines are in $(pwd)/engines. Point jade-vision at them (see jade-vision.env)."
