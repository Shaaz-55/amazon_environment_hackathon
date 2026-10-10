#!/usr/bin/env python3
"""
Household Waste Classifier - Prediction CLI & API Helper
Takes an image path and outputs top 3 predicted classes with confidence,
applying the confidence threshold to flag uncertain predictions.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

def load_image(path):
    with Image.open(path) as img:
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        else:
            return img.convert('RGB')

def preprocess(img):
    # Resize shortest edge to 256, center crop to 224
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(h * (256 / w))
    else:
        new_w, new_h = int(w * (256 / h)), 256
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Center crop
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    # Convert to numpy float32 and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # Transpose to (1, 3, 224, 224)
    tensor = np.transpose(arr, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def predict(image_path, model_path="model.onnx", labels_path="labels.json", threshold=0.60):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path) as f:
        labels_info = json.load(f)
    classes = labels_info["classes"]

    # Preprocess
    img = load_image(image_path)
    input_tensor = preprocess(img)

    # Inference via ONNX Runtime if .onnx, else PyTorch
    if model_path.endswith(".onnx"):
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: input_tensor})[0]
    else:
        import torch
        from torchvision.models import resnet50
        model = resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(input_tensor)).numpy()

    probs = softmax(logits)[0]

    # Rank classes
    top_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "class": classes[idx],
            "confidence": round(float(probs[idx]), 4)
        }
        for idx in top_indices
    ]

    top_class = top3[0]["class"]
    top_conf = top3[0]["confidence"]
    unsure = bool(top_conf < threshold)

    result = {
        "image": image_path,
        "top3": top3,
        "top_class": top_class,
        "confidence": top_conf,
        "threshold": threshold,
        "unsure": unsure,
        "decision": "not sure, try another angle" if unsure else f"Accepted ({top_class})"
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Predict household waste material from image.")
    parser.add_argument("image_path", help="Path to input photo of waste item")
    parser.add_argument("--model", default="model.onnx", help="Path to ONNX model (or PyTorch checkpoint)")
    parser.add_argument("--labels", default="labels.json", help="Path to labels.json")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold")
    parser.add_argument("--json", action="store_true", help="Output pure JSON format")
    args = parser.parse_args()

    # If model.onnx doesn't exist yet, check best_model.pt
    model_to_use = args.model
    if not os.path.exists(model_to_use):
        if os.path.exists("checkpoints/best_model.pt"):
            model_to_use = "checkpoints/best_model.pt"

    res = predict(
        image_path=args.image_path,
        model_path=model_to_use,
        labels_path=args.labels,
        threshold=args.threshold
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("=" * 50)
        print("HOUSEHOLD WASTE MATERIAL CLASSIFICATION")
        print("=" * 50)
        print(f"Image:      {res['image']}")
        print(f"Top 1:      {res['top3'][0]['class']} ({res['top3'][0]['confidence']*100:.1f}%)")
        print(f"Top 2:      {res['top3'][1]['class']} ({res['top3'][1]['confidence']*100:.1f}%)")
        print(f"Top 3:      {res['top3'][2]['class']} ({res['top3'][2]['confidence']*100:.1f}%)")
        print("-" * 50)
        if res["unsure"]:
            print(f"STATUS:     UNSURE (< {args.threshold*100:.0f}%) -> {res['decision']}")
        else:
            print(f"STATUS:     CONFIDENT -> Classified as '{res['top_class']}'")
        print("=" * 50)

if __name__ == "__main__":
    main()
