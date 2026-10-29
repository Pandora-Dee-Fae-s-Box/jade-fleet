#!/usr/bin/env python3
"""Export the two vision models to ONNX, with their label lists. Run it once on any machine with PyTorch (a PC is
faster; the Xavier works too):

    pip install ultralytics torchvision onnx onnxsim
    python3 export_models.py            # writes models/yolov8n.onnx, models/coco.names,
                                        #        models/mobilenet_v2.onnx, models/imagenet.names

Then copy models/ to the Xavier and build the TensorRT engines there with build_engines.sh (engines are built for the
exact Jetson, JetPack and TensorRT they run on, so they cannot be made elsewhere).
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--yolo", default="yolov8n.pt", help="ultralytics weights (yolov8n.pt, yolov8s.pt, yolo11n.pt …)")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", default="models")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# YOLO: static 1x3xNxN input, opset 13 (TensorRT 8.5 on JetPack 5 reads it), simplified graph
from ultralytics import YOLO

y = YOLO(a.yolo)
onnx = y.export(format="onnx", imgsz=a.imgsz, opset=13, simplify=True, dynamic=False, batch=1)
name = os.path.splitext(os.path.basename(a.yolo))[0]
os.replace(onnx, os.path.join(a.out, name + ".onnx"))
with open(os.path.join(a.out, "coco.names"), "w") as f:
    f.write("\n".join(y.names[i] for i in range(len(y.names))) + "\n")
print("YOLO:", os.path.join(a.out, name + ".onnx"), len(y.names), "classes")

# MobileNet v2 (ImageNet, 1000 classes): input 1x3x224x224, normalised as torchvision does
import torch
import torchvision

w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
m = torchvision.models.mobilenet_v2(weights=w).eval()
torch.onnx.export(m, torch.zeros(1, 3, 224, 224), os.path.join(a.out, "mobilenet_v2.onnx"), opset_version=13,
                  input_names=["input"], output_names=["logits"])
with open(os.path.join(a.out, "imagenet.names"), "w") as f:
    f.write("\n".join(w.meta["categories"]) + "\n")
print("MobileNet:", os.path.join(a.out, "mobilenet_v2.onnx"), len(w.meta["categories"]), "classes")
