import time
import platform
import psutil
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_system_info():
    info = {}
    info["os"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    
    # Apple Silicon chip detection
    try:
        import subprocess
        chip_out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        info["cpu_brand"] = chip_out
    except Exception:
        info["cpu_brand"] = platform.processor()

    # Memory
    vm = psutil.virtual_memory()
    info["ram_total_gb"] = round(vm.total / (1024**3), 2)
    info["ram_available_gb"] = round(vm.available / (1024**3), 2)

    # Disk
    du = psutil.disk_usage(".")
    info["disk_total_gb"] = round(du.total / (1024**3), 2)
    info["disk_free_gb"] = round(du.free / (1024**3), 2)

    # Compute Device
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon GPU (Metal Performance Shaders / MPS)"
    else:
        info["device"] = "cpu"
        info["device_name"] = "CPU only"

    return info

def run_speed_test(batch_size=32, num_warmup=5, num_batches=20, estimated_train_images=12000):
    info = get_system_info()
    device_str = info["device"]
    device = torch.device(device_str)
    
    print("=" * 60)
    print("SYSTEM AND HARDWARE SPECS")
    print("=" * 60)
    print(f"OS:            {info['os']}")
    print(f"Chip/CPU:      {info['cpu_brand']}")
    print(f"RAM:           {info['ram_total_gb']} GB total ({info['ram_available_gb']} GB available)")
    print(f"Free Disk:     {info['disk_free_gb']} GB free")
    print(f"Device:        {device_str.upper()} -> {info['device_name']}")
    print("=" * 60)

    # Build model
    print(f"\nInitializing ResNet50 on device: {device}...")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Synthetic batch
    dummy_inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_targets = torch.randint(0, 10, (batch_size,), device=device)

    # --- TEST STAGE 1: Head Only (Backbone Frozen) ---
    print("\n[Stage 1 Benchmark: Backbone Frozen, Linear Head Training]")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer_s1 = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    # Warmup
    model.train()
    for _ in range(num_warmup):
        optimizer_s1.zero_grad()
        out = model(dummy_inputs)
        loss = criterion(out, dummy_targets)
        loss.backward()
        optimizer_s1.step()
    if device_str == "mps":
        torch.mps.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(num_batches):
        optimizer_s1.zero_grad()
        out = model(dummy_inputs)
        loss = criterion(out, dummy_targets)
        loss.backward()
        optimizer_s1.step()
    if device_str == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()

    stage1_total_time = t1 - t0
    stage1_time_per_batch = stage1_total_time / num_batches
    stage1_throughput = (num_batches * batch_size) / stage1_total_time
    stage1_epoch_sec = (estimated_train_images / batch_size) * stage1_time_per_batch

    print(f"Stage 1 batch time:     {stage1_time_per_batch*1000:.1f} ms/batch ({stage1_throughput:.1f} images/s)")
    print(f"Stage 1 1-epoch est:    {stage1_epoch_sec:.1f} s ({stage1_epoch_sec/60:.2f} min)")

    # --- TEST STAGE 2: Fine-tuning (Last 2 ResNet Blocks + Head Unfrozen) ---
    print("\n[Stage 2 Benchmark: Last 2 Blocks + Head Unfrozen]")
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer_s2 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    # Warmup
    for _ in range(num_warmup):
        optimizer_s2.zero_grad()
        out = model(dummy_inputs)
        loss = criterion(out, dummy_targets)
        loss.backward()
        optimizer_s2.step()
    if device_str == "mps":
        torch.mps.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(num_batches):
        optimizer_s2.zero_grad()
        out = model(dummy_inputs)
        loss = criterion(out, dummy_targets)
        loss.backward()
        optimizer_s2.step()
    if device_str == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()

    stage2_total_time = t1 - t0
    stage2_time_per_batch = stage2_total_time / num_batches
    stage2_throughput = (num_batches * batch_size) / stage2_total_time
    stage2_epoch_sec = (estimated_train_images / batch_size) * stage2_time_per_batch

    print(f"Stage 2 batch time:     {stage2_time_per_batch*1000:.1f} ms/batch ({stage2_throughput:.1f} images/s)")
    print(f"Stage 2 1-epoch est:    {stage2_epoch_sec:.1f} s ({stage2_epoch_sec/60:.2f} min)")

    # --- FULL PLAN PROJECTION ---
    # Phase 4 Plan: Stage 1 = up to 5 epochs, Stage 2 = up to 15 epochs
    total_plan_sec = (5 * stage1_epoch_sec) + (15 * stage2_epoch_sec)
    print("\n" + "=" * 60)
    print("PROJECTED RUNTIME FOR FULL TRAINING PLAN (RESNET50)")
    print("=" * 60)
    print(f"Estimated train images:   {estimated_train_images} (approx 80% split)")
    print(f"Batch size:               {batch_size}")
    print(f"Stage 1 (5 epochs):       {5 * stage1_epoch_sec / 60:.2f} minutes")
    print(f"Stage 2 (up to 15 epochs):{15 * stage2_epoch_sec / 60:.2f} minutes")
    print(f"Total training estimate:  {total_plan_sec / 60:.2f} minutes ({total_plan_sec / 3600:.2f} hours)")
    print("=" * 60)

    results = {
        "system_info": info,
        "batch_size": batch_size,
        "stage1": {
            "batch_time_ms": stage1_time_per_batch * 1000,
            "throughput_img_per_sec": stage1_throughput,
            "epoch_sec": stage1_epoch_sec,
            "epoch_min": stage1_epoch_sec / 60,
            "five_epochs_min": 5 * stage1_epoch_sec / 60
        },
        "stage2": {
            "batch_time_ms": stage2_time_per_batch * 1000,
            "throughput_img_per_sec": stage2_throughput,
            "epoch_sec": stage2_epoch_sec,
            "epoch_min": stage2_epoch_sec / 60,
            "fifteen_epochs_min": 15 * stage2_epoch_sec / 60
        },
        "full_plan_min": total_plan_sec / 60,
        "full_plan_hours": total_plan_sec / 3600
    }
    
    import json
    with open("reports/speed_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_speed_test()
