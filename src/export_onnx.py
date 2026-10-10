import os
import sys
import json
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import onnx
import onnxruntime as ort

from dataset import load_image_with_white_bg, get_val_transforms
from train import build_model

def export_to_onnx(
    checkpoint_path="checkpoints/best_model.pt",
    output_onnx_path="model.onnx",
    test_manifest_path="manifests/test_manifest.json"
):
    print("=" * 60)
    print("PHASE 6: ONNX EXPORT AND VERIFICATION")
    print("=" * 60)

    # 1. Load labels
    with open("labels.json") as f:
        labels_info = json.load(f)
    class_names = labels_info["classes"]
    num_classes = len(class_names)

    # 2. Build model and load weights
    model = build_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }

    print(f"Exporting PyTorch model to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamo=False
    )

    # Check ONNX model validity
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"ONNX model verified successfully! File size: {onnx_file_size_mb:.2f} MB")

    # 4. Verification against PyTorch on at least 50 test images
    print(f"\nVerifying ONNX output matches PyTorch on 50 test images...")
    with open(test_manifest_path) as f:
        test_samples = json.load(f)

    sample_subset = test_samples[:50]
    val_transform = get_val_transforms()

    ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])

    max_abs_diff = 0.0
    differences = []

    for item in sample_subset:
        img = load_image_with_white_bg(item["path"])
        tensor = val_transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

        # PyTorch forward
        with torch.no_grad():
            pt_out = model(tensor).numpy()

        # ONNX Runtime forward
        ort_inputs = {ort_session.get_inputs()[0].name: tensor.numpy()}
        ort_out = ort_session.run(None, ort_inputs)[0]

        diff = np.max(np.abs(pt_out - ort_out))
        differences.append(diff)
        if diff > max_abs_diff:
            max_abs_diff = diff

    print(f"Max absolute difference over 50 images: {max_abs_diff:.6e}")
    assert max_abs_diff < 1e-4, f"Discrepancy too large: {max_abs_diff}"
    print("Verification passed! PyTorch and ONNX predictions are numerically consistent.")

    # 5. Measure CPU inference latency per image
    print(f"\nBenchmarking CPU inference latency over 100 test iterations...")
    benchmark_tensor = val_transform(load_image_with_white_bg(sample_subset[0]["path"])).unsqueeze(0).numpy()

    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: benchmark_tensor})

    latencies = []
    num_runs = 100
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: benchmark_tensor})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    avg_latency_ms = np.mean(latencies)
    p50_latency_ms = np.median(latencies)
    p95_latency_ms = np.percentile(latencies, 95)
    print(f"CPU Latency per image: Mean = {avg_latency_ms:.2f} ms | P50 = {p50_latency_ms:.2f} ms | P95 = {p95_latency_ms:.2f} ms")

    quantized_path = None
    quantized_size_mb = None
    quantized_latency_ms = None

    if avg_latency_ms > 500.0:
        print("Average CPU latency exceeds 500 ms! Quantizing model...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = "model_quantized.onnx"
        quantize_dynamic(output_onnx_path, quantized_path, weight_type=QuantType.QUInt8)
        quantized_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)

        # Benchmark quantized model
        q_session = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])
        q_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = q_session.run(None, {q_session.get_inputs()[0].name: benchmark_tensor})
            t1 = time.perf_counter()
            q_latencies.append((t1 - t0) * 1000.0)
        quantized_latency_ms = np.mean(q_latencies)
        print(f"Quantized model size: {quantized_size_mb:.2f} MB | Latency: {quantized_latency_ms:.2f} ms")
    else:
        print(f"Inference latency is well below 500 ms threshold ({avg_latency_ms:.2f} ms < 500 ms). No quantization required.")

    export_report = {
        "onnx_model_path": output_onnx_path,
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "float32",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "classes": class_names,
        "file_size_mb": round(onnx_file_size_mb, 2),
        "verification": {
            "samples_tested": len(sample_subset),
            "max_absolute_difference": float(max_abs_diff),
            "mean_difference": float(np.mean(differences)),
            "status": "PASSED"
        },
        "cpu_inference_latency": {
            "mean_ms": round(float(avg_latency_ms), 2),
            "median_ms": round(float(p50_latency_ms), 2),
            "p95_ms": round(float(p95_latency_ms), 2),
            "meets_500ms_sla": bool(avg_latency_ms < 500.0)
        },
        "quantization": {
            "performed": bool(avg_latency_ms > 500.0),
            "quantized_path": quantized_path,
            "quantized_size_mb": quantized_size_mb,
            "quantized_latency_ms": quantized_latency_ms
        }
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/onnx_export_report.json", "w") as f:
        json.dump(export_report, f, indent=2)

    print("ONNX export report saved to reports/onnx_export_report.json")
    return export_report

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    export_to_onnx(checkpoint_path=ckpt)
