#!/usr/bin/env python3
"""Jade Fleet vision service for the Jetson AGX Xavier: YOLO object detection and MobileNet classification as
TensorRT engines on the two DLA cores (NVDLA), served over HTTP for the broker on the OptiPlex.

    POST /v1/vision/detect     {"image": "<base64 or data: URL>", "conf": 0.25, "iou": 0.45, "max": 100}
                               (or the raw image bytes with Content-Type: image/…)
        -> {"model", "width", "height", "detections": [{"label", "class_id", "confidence", "box": [x1, y1, x2, y2]}],
            "ms": {"decode", "pre", "infer", "post", "total"}, "device": "DLA0"}
    POST /v1/vision/classify   {"image": …, "top": 5}
        -> {"model", "width", "height", "top": [{"label", "class_id", "prob"}], "ms": {…}, "device": "DLA1"}
    GET  /v1/vision/models     the loaded engines;   GET /health

Needs only what JetPack ships: Python 3.8+, numpy, OpenCV (cv2) and TensorRT's Python bindings; CUDA is reached
through libcudart with ctypes. Each engine has its own execution context and lock, so detection and classification
run at the same time on their two DLA cores. Settings: command-line flags or JADE_VISION_* variables (see --help).
"""
import argparse
import base64
import ctypes
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MAX_BODY = 25 * 1024 * 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------- images

def decode_image(data):
    """Bytes of a JPEG/PNG/… -> HxWx3 RGB uint8."""
    try:
        import cv2
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("not an image OpenCV can read")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    except ImportError:
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def resize(img, w, h):
    try:
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def letterbox(img, size):
    """Fit into size x size keeping the aspect ratio, pad with grey (114). Returns the tensor and how to map back."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    pad = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    pad[dy:dy + nh, dx:dx + nw] = resize(img, nw, nh) if (nw, nh) != (w, h) else img
    x = pad.astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), (r, dx, dy)


def center_crop(img, size, resize_to=None):
    """Torchvision's classification preprocessing: shorter side to 256 (for 224), centre crop, ImageNet normalise."""
    resize_to = resize_to or int(round(size * 256 / 224))
    h, w = img.shape[:2]
    r = resize_to / min(h, w)
    nw, nh = max(size, int(round(w * r))), max(size, int(round(h * r)))
    im = resize(img, nw, nh)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    im = im[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255.0
    im = (im - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(im.transpose(2, 0, 1)[None])


# ----------------------------------------------------------------------------- model outputs

def nms(boxes, scores, iou):
    """Indices kept by non-maximum suppression (boxes x1,y1,x2,y2), best first."""
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        ov = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[ov <= iou]
    return keep


def decode_yolo(out, n_classes, conf, iou, max_det, mapping, width, height):
    """YOLOv8/11 output (1, 4+nc, N) or (1, N, 4+nc), or YOLOv5 (1, N, 5+nc) with objectness -> detections."""
    p = np.asarray(out, dtype=np.float32)
    p = p.reshape(p.shape[-2], p.shape[-1]) if p.ndim == 3 else p
    if p.shape[0] < p.shape[1] and p.shape[0] in (4 + n_classes, 5 + n_classes):
        p = p.T                                       # (N, C)
    c = p.shape[1]
    if c == 5 + n_classes:                            # v5: objectness x class score
        cls_scores = p[:, 5:] * p[:, 4:5]
    elif c == 4 + n_classes:
        cls_scores = p[:, 4:]
    else:
        raise ValueError("unexpected YOLO output shape %s for %d classes" % (list(out.shape), n_classes))
    cls = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(cls)), cls]
    m = score >= conf
    if not m.any():
        return []
    xywh, cls, score = p[m, :4], cls[m], score[m]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    # one NMS for all classes: shift each class's boxes far apart
    off = cls[:, None].astype(np.float32) * 8192.0
    keep = nms(boxes + off, score, iou)[:max_det]
    r, dx, dy = mapping
    out_l = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        x1, x2 = [float(np.clip((v - dx) / r, 0, width)) for v in (x1, x2)]
        y1, y2 = [float(np.clip((v - dy) / r, 0, height)) for v in (y1, y2)]
        out_l.append({"class_id": int(cls[i]), "confidence": round(float(score[i]), 4), "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    return out_l


def decode_classes(out, labels, top):
    v = np.asarray(out, dtype=np.float64).reshape(-1)
    if len(labels) and v.size == len(labels) + 1:       # TF-style models have a background class first
        v = v[1:]
    if v.min() < 0 or abs(v.sum() - 1.0) > 1e-3:        # logits -> probabilities
        e = np.exp(v - v.max())
        v = e / e.sum()
    idx = v.argsort()[::-1][:top]
    return [{"class_id": int(i), "label": labels[i] if i < len(labels) else "class %d" % i, "prob": round(float(v[i]), 4)} for i in idx]


# ----------------------------------------------------------------------------- TensorRT on the DLA

class Cuda(object):
    """The few CUDA runtime calls needed, through ctypes (no pycuda to install)."""
    H2D, D2H = 1, 2

    def __init__(self):
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                self.rt = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise OSError("libcudart not found (is JetPack's CUDA installed?)")

    def check(self, code, what):
        if code != 0:
            raise RuntimeError("%s failed: CUDA error %d" % (what, code))

    def malloc(self, n):
        p = ctypes.c_void_p()
        self.check(self.rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "cudaMalloc")
        return p.value

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.rt.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s.value

    def copy(self, dst, src, n, kind, stream):
        self.check(self.rt.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(n), ctypes.c_int(kind), ctypes.c_void_p(stream)), "cudaMemcpyAsync")

    def sync(self, stream):
        self.check(self.rt.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize")


class TrtEngine(object):
    """One serialized engine, its own context, CUDA stream and device buffers. Works with TensorRT 8.5 (JetPack 5,
    the Xavier's) and the newer tensor-name API."""

    def __init__(self, path, dla_core, cuda):
        import tensorrt as trt
        self.trt, self.cuda, self.path, self.dla = trt, cuda, path, dla_core
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        if dla_core is not None and dla_core >= 0:
            self.runtime.DLA_core = int(dla_core)
        with open(path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("%s: TensorRT could not load this engine (built for another TensorRT or device?)" % path)
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.stream()
        self.lock = threading.Lock()
        self.io = []      # (name, is_input, shape, dtype, device_ptr, nbytes)
        new_api = hasattr(self.engine, "num_io_tensors")
        n = self.engine.num_io_tensors if new_api else self.engine.num_bindings
        for i in range(n):
            if new_api:
                name = self.engine.get_tensor_name(i)
                is_in = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            else:
                name = self.engine.get_binding_name(i)
                is_in = self.engine.binding_is_input(i)
                shape = tuple(self.engine.get_binding_shape(i))
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
            if any(d < 0 for d in shape):
                raise RuntimeError("%s: dynamic shapes are not supported here; export with a fixed batch and size" % path)
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            ptr = cuda.malloc(nbytes)
            self.io.append((name, is_in, shape, dtype, ptr, nbytes))
            if new_api:
                self.ctx.set_tensor_address(name, ptr)
        self.new_api = new_api
        self.inputs = [x for x in self.io if x[1]]
        self.outputs = [x for x in self.io if not x[1]]
        self.input_shape = self.inputs[0][2]

    def run(self, x):
        name, _, shape, dtype, ptr, nbytes = self.inputs[0]
        x = np.ascontiguousarray(x.reshape(shape).astype(dtype))
        outs = [np.empty(o[2], o[3]) for o in self.outputs]
        with self.lock:
            self.cuda.copy(ptr, x.ctypes.data, nbytes, Cuda.H2D, self.stream)
            if self.new_api:
                ok = self.ctx.execute_async_v3(self.stream)
            else:
                ok = self.ctx.execute_async_v2([b[4] for b in self.io], self.stream)
            if not ok:
                raise RuntimeError("TensorRT execution failed")
            for o, h in zip(self.outputs, outs):
                self.cuda.copy(h.ctypes.data, o[4], o[5], Cuda.D2H, self.stream)
            self.cuda.sync(self.stream)
        return outs

    def describe(self):
        return {"engine": os.path.basename(self.path), "device": "DLA%d" % self.dla if self.dla is not None and self.dla >= 0 else "GPU",
                "inputs": [{"name": x[0], "shape": list(x[2]), "dtype": np.dtype(x[3]).name} for x in self.inputs],
                "outputs": [{"name": x[0], "shape": list(x[2]), "dtype": np.dtype(x[3]).name} for x in self.outputs]}


# ----------------------------------------------------------------------------- the service

def read_labels(path, n_default=0):
    if path and os.path.exists(path):
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]
    return ["class %d" % i for i in range(n_default)]


class Vision(object):
    def __init__(self, detector=None, det_labels=None, classifier=None, cls_labels=None, det_name="yolo", cls_name="mobilenet"):
        self.det, self.cls = detector, classifier
        self.det_labels, self.cls_labels = det_labels or [], cls_labels or []
        self.det_name, self.cls_name = det_name, cls_name

    def detect(self, img, conf=0.25, iou=0.45, max_det=100):
        if self.det is None:
            raise LookupError("no detection engine is loaded (--yolo)")
        t0 = time.time()
        h, w = img.shape[:2]
        size = self.det.input_shape[-1]
        x, mapping = letterbox(img, size)
        t1 = time.time()
        out = self.det.run(x)[0]
        t2 = time.time()
        n = len(self.det_labels) or (out.shape[1] - 4 if out.shape[1] < out.shape[2] else out.shape[2] - 4)
        dets = decode_yolo(out, n, conf, iou, max_det, mapping, w, h)
        for d in dets:
            d["label"] = self.det_labels[d["class_id"]] if d["class_id"] < len(self.det_labels) else "class %d" % d["class_id"]
        t3 = time.time()
        return {"model": self.det_name, "width": w, "height": h, "detections": dets, "ms": {"pre": ms(t0, t1), "infer": ms(t1, t2), "post": ms(t2, t3)}, "device": self.det.describe()["device"]}

    def classify(self, img, top=5):
        if self.cls is None:
            raise LookupError("no classification engine is loaded (--mobilenet)")
        t0 = time.time()
        h, w = img.shape[:2]
        x = center_crop(img, self.cls.input_shape[-1])
        t1 = time.time()
        out = self.cls.run(x)[0]
        t2 = time.time()
        res = decode_classes(out, self.cls_labels, max(1, min(int(top), 50)))
        t3 = time.time()
        return {"model": self.cls_name, "width": w, "height": h, "top": res, "ms": {"pre": ms(t0, t1), "infer": ms(t1, t2), "post": ms(t2, t3)}, "device": self.cls.describe()["device"]}

    def models(self):
        out = {}
        if self.det is not None:
            out["detect"] = dict(self.det.describe(), name=self.det_name, classes=len(self.det_labels))
        if self.cls is not None:
            out["classify"] = dict(self.cls.describe(), name=self.cls_name, classes=len(self.cls_labels))
        return out


def ms(a, b):
    return round((b - a) * 1000.0, 2)


def make_handler(vision, token=None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "jade-vision/1"

        def log_message(self, fmt, *args):
            if os.environ.get("JADE_VISION_LOG"):
                sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

        def send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            if not token:
                return True
            if self.headers.get("Authorization", "") == "Bearer " + token:
                return True
            self.send(401, {"ok": False, "error": "missing or wrong token"})
            return False

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/health":
                return self.send(200, {"ok": True, "service": "jade-vision"})
            if not self.allowed():
                return
            if p == "/v1/vision/models":
                return self.send(200, {"ok": True, "models": vision.models()})
            self.send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            p = self.path.split("?")[0]
            if not self.allowed():
                return
            if p not in ("/v1/vision/detect", "/v1/vision/classify"):
                return self.send(404, {"ok": False, "error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self.send(413 if n > MAX_BODY else 400, {"ok": False, "error": "send an image (at most %d MB)" % (MAX_BODY // 1048576)})
            body = self.rfile.read(n)
            t0 = time.time()
            try:
                ct = (self.headers.get("Content-Type") or "").split(";")[0].strip()
                if ct.startswith("image/") or ct == "application/octet-stream":
                    args, data = {}, body
                else:
                    args = json.loads(body.decode("utf-8"))
                    b64 = args.get("image") or ""
                    if b64.startswith("data:"):
                        b64 = b64.split(",", 1)[1]
                    data = base64.b64decode(b64)
                img = decode_image(data)
            except Exception as e:
                return self.send(400, {"ok": False, "error": "could not read the image: %s" % e})
            t1 = time.time()
            try:
                if p.endswith("detect"):
                    r = vision.detect(img, float(args.get("conf", 0.25)), float(args.get("iou", 0.45)), int(args.get("max", 100)))
                else:
                    r = vision.classify(img, int(args.get("top", 5)))
            except LookupError as e:
                return self.send(404, {"ok": False, "error": str(e)})
            except Exception as e:
                return self.send(500, {"ok": False, "error": "%s: %s" % (type(e).__name__, e)})
            r["ms"]["decode"] = ms(t0, t1)
            r["ms"]["total"] = ms(t0, time.time())
            r["ok"] = True
            self.send(200, r)
    return H


def main(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(description="Jade Fleet vision service (TensorRT on the Xavier's DLA cores)")
    ap.add_argument("--host", default=env("JADE_VISION_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(env("JADE_VISION_PORT", "8090")))
    ap.add_argument("--yolo", default=env("JADE_VISION_YOLO", "engines/yolov8n_dla0_fp16.engine"))
    ap.add_argument("--yolo-dla", type=int, default=int(env("JADE_VISION_YOLO_DLA", "0")), help="DLA core, -1 for the GPU")
    ap.add_argument("--yolo-labels", default=env("JADE_VISION_YOLO_LABELS", "models/coco.names"))
    ap.add_argument("--mobilenet", default=env("JADE_VISION_MOBILENET", "engines/mobilenet_v2_dla1_fp16.engine"))
    ap.add_argument("--mobilenet-dla", type=int, default=int(env("JADE_VISION_MOBILENET_DLA", "1")))
    ap.add_argument("--mobilenet-labels", default=env("JADE_VISION_MOBILENET_LABELS", "models/imagenet.names"))
    ap.add_argument("--token", default=env("JADE_VISION_TOKEN", ""), help="optional bearer token (the broker does not send one by default)")
    a = ap.parse_args(argv)
    cuda = Cuda()
    det = cls = None
    if a.yolo and os.path.exists(a.yolo):
        det = TrtEngine(a.yolo, a.yolo_dla, cuda)
        print("detect:", json.dumps(det.describe()))
    else:
        print("no detection engine at %s; /v1/vision/detect is off" % a.yolo)
    if a.mobilenet and os.path.exists(a.mobilenet):
        cls = TrtEngine(a.mobilenet, a.mobilenet_dla, cuda)
        print("classify:", json.dumps(cls.describe()))
    else:
        print("no classification engine at %s; /v1/vision/classify is off" % a.mobilenet)
    v = Vision(det, read_labels(a.yolo_labels, 80), cls, read_labels(a.mobilenet_labels, 1000),
               os.path.basename(a.yolo).split("_dla")[0], os.path.basename(a.mobilenet).split("_dla")[0])
    # warm up both engines (the first DLA run is slow)
    for eng in (det, cls):
        if eng is not None:
            eng.run(np.zeros(eng.input_shape, np.float32))
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(v, a.token or None))
    srv.daemon_threads = True
    print("jade-vision on %s:%d" % (a.host, a.port))
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
