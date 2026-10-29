#!/usr/bin/env python3
"""Tests for the vision service without a Jetson: stand-in engines replace TensorRT, everything else is real
(image decoding, letterboxing, YOLO decoding and NMS, classification, the HTTP server, two models at once).
    python3 test_jade_vision.py
"""
import base64
import json
import threading
import time
import unittest
import urllib.request
import urllib.error

import cv2
import numpy as np

import jade_vision as jv


class FakeEngine(object):
    def __init__(self, shape, fn, delay=0.0, device="DLA0"):
        self.input_shape, self.fn, self.delay, self.device = shape, fn, delay, device
        self.lock = threading.Lock()
        self.seen = []

    def run(self, x):
        with self.lock:              # like the real engine: one run at a time per engine
            self.seen.append(x)
            time.sleep(self.delay)
            return [self.fn(x)]

    def describe(self):
        return {"engine": "fake", "device": self.device, "inputs": [{"shape": list(self.input_shape)}], "outputs": []}


def yolo_v8_out(boxes, nc=80, n=8400):
    """Build a (1, 4+nc, n) output with the given (cx, cy, w, h, class, score) rows in model-input pixels."""
    o = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(boxes):
        o[0, :4, i] = [cx, cy, w, h]
        o[0, 4 + c, i] = s
    return o


class Decoding(unittest.TestCase):
    def test_letterbox_maps_back(self):
        img = np.zeros((480, 640, 3), np.uint8)
        x, (r, dx, dy) = jv.letterbox(img, 640)
        self.assertEqual(x.shape, (1, 3, 640, 640))
        self.assertAlmostEqual(r, 1.0)
        self.assertEqual((dx, dy), (0, 80))
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), 114 / 255.0, places=5)   # padding is grey
        # a box at model (100, 180)-(200, 280) is image (100, 100)-(200, 200)
        out = yolo_v8_out([(150, 230, 100, 100, 0, 0.9)])
        d = jv.decode_yolo(out, 80, 0.25, 0.45, 100, (r, dx, dy), 640, 480)
        self.assertEqual(d[0]["box"], [100.0, 100.0, 200.0, 200.0])

    def test_nms_and_classes(self):
        out = yolo_v8_out([
            (100, 100, 50, 50, 0, 0.9),     # person
            (102, 101, 50, 50, 0, 0.8),     # same person again: suppressed
            (101, 100, 50, 50, 2, 0.7),     # a car in the same place: kept (other class)
            (400, 400, 40, 40, 16, 0.6),    # a dog
            (300, 300, 40, 40, 16, 0.1),    # below the threshold
        ])
        d = jv.decode_yolo(out, 80, 0.25, 0.45, 100, (1.0, 0, 0), 640, 640)
        self.assertEqual([(x["class_id"], x["confidence"]) for x in d], [(0, 0.9), (2, 0.7), (16, 0.6)])
        self.assertEqual(len(jv.decode_yolo(out, 80, 0.25, 0.45, 2, (1.0, 0, 0), 640, 640)), 2, "max detections")
        self.assertEqual(jv.decode_yolo(out, 80, 0.95, 0.45, 100, (1.0, 0, 0), 640, 640), [])

    def test_yolov5_layout_with_objectness(self):
        o = np.zeros((1, 25200, 85), np.float32)
        o[0, 0, :5] = [50, 50, 20, 20, 0.9]
        o[0, 0, 5 + 3] = 0.8                # motorcycle, 0.9 * 0.8
        d = jv.decode_yolo(o, 80, 0.25, 0.45, 10, (1.0, 0, 0), 640, 640)
        self.assertEqual((d[0]["class_id"], d[0]["confidence"]), (3, 0.72))

    def test_classification(self):
        labels = ["cat", "dog", "fox"]
        top = jv.decode_classes(np.array([[1.0, 3.0, 2.0]]), labels, 2)
        self.assertEqual([t["label"] for t in top], ["dog", "fox"])
        self.assertAlmostEqual(sum(jv.decode_classes(np.array([1.0, 3.0, 2.0]), labels, 3)[i]["prob"] for i in range(3)), 1.0, places=3)
        top = jv.decode_classes(np.array([0.0, 0.1, 0.7, 0.2]), labels, 1)   # background first, already probabilities
        self.assertEqual(top[0]["label"], "dog")

    def test_center_crop(self):
        img = np.full((300, 600, 3), 255, np.uint8)
        x = jv.center_crop(img, 224)
        self.assertEqual(x.shape, (1, 3, 224, 224))
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), (1 - 0.485) / 0.229, places=4)


class Service(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def det(x):
            return yolo_v8_out([(320, 320, 64, 64, 0, 0.9), (100, 300, 50, 80, 16, 0.75)])

        def clf(x):
            v = np.zeros((1, 1000), np.float32)
            v[0, 281] = 9.0
            v[0, 285] = 7.0
            return v
        cls.det = FakeEngine((1, 3, 640, 640), det, delay=0.3, device="DLA0")
        cls.cls = FakeEngine((1, 3, 224, 224), clf, delay=0.3, device="DLA1")
        labels = ["class %d" % i for i in range(1000)]
        labels[281], labels[285] = "tabby cat", "Egyptian cat"
        coco = ["person"] + ["c%d" % i for i in range(1, 80)]
        coco[16] = "dog"
        v = jv.Vision(cls.det, coco, cls.cls, labels, "yolov8n", "mobilenet_v2")
        cls.srv = jv.ThreadingHTTPServer(("127.0.0.1", 0), jv.make_handler(v, "tok"))
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:%d" % cls.srv.server_address[1]
        img = np.zeros((480, 640, 3), np.uint8)
        img[:, :, 0] = 255                  # pure red (RGB) …
        ok, jpg = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cls.png = jpg.tobytes()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def call(self, path, body=None, ctype="application/json", token="tok"):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET")
        if data is not None:
            req.add_header("Content-Type", ctype)
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_detect(self):
        st, r = self.call("/v1/vision/detect", {"image": "data:image/png;base64," + base64.b64encode(self.png).decode()})
        self.assertEqual(st, 200, r)
        self.assertEqual((r["width"], r["height"], r["device"], r["model"]), (640, 480, "DLA0", "yolov8n"))
        self.assertEqual([d["label"] for d in r["detections"]], ["person", "dog"])
        self.assertEqual(r["detections"][0]["box"], [288.0, 208.0, 352.0, 272.0], "boxes are in the image's own pixels")
        x = self.det.seen[-1]
        self.assertAlmostEqual(float(x[0, 0, 240, 320]), 1.0, places=4, msg="the model saw RGB: red channel full")
        self.assertAlmostEqual(float(x[0, 2, 240, 320]), 0.0, places=4)
        for k in ("decode", "pre", "infer", "post", "total"):
            self.assertIn(k, r["ms"])

    def test_raw_image_and_classify(self):
        st, r = self.call("/v1/vision/classify?x=1", self.png, ctype="image/png")
        self.assertEqual(st, 200, r)
        self.assertEqual([t["label"] for t in r["top"][:2]], ["tabby cat", "Egyptian cat"])
        self.assertAlmostEqual(r["top"][0]["prob"], 0.7946, places=3)
        self.assertEqual(r["device"], "DLA1")

    def test_errors_and_auth(self):
        self.assertEqual(self.call("/v1/vision/detect", {"image": "bm90IGFuIGltYWdl"})[0], 400)
        self.assertEqual(self.call("/v1/vision/detect", {"image": "x"}, token="bad")[0], 401)
        self.assertEqual(self.call("/v1/vision/segment", {"image": "x"})[0], 404)
        st, r = self.call("/health", token=None)
        self.assertTrue(st == 200 and r["ok"])
        st, r = self.call("/v1/vision/models")
        self.assertEqual(sorted(r["models"]), ["classify", "detect"])

    def test_two_dla_cores_at_once(self):
        """Detection and classification each take 0.3 s here; on their own DLA cores they overlap."""
        res = {}

        def go(path):
            res[path] = self.call(path, self.png, ctype="image/png")[0]
        t0 = time.time()
        ts = [threading.Thread(target=go, args=(p,)) for p in ("/v1/vision/detect", "/v1/vision/classify")]
        [t.start() for t in ts]
        [t.join() for t in ts]
        took = time.time() - t0
        self.assertEqual(set(res.values()), {200})
        self.assertLess(took, 0.55, "ran one after the other (%.2fs)" % took)

    def test_missing_engine(self):
        v = jv.Vision(None, [], None, [])
        with self.assertRaises(LookupError):
            v.detect(np.zeros((10, 10, 3), np.uint8))


if __name__ == "__main__":
    unittest.main(verbosity=1)
