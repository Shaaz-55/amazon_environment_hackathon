import os
import sys
import json
import time
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

# Enable MPS CPU fallback for unsupported operators if any
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dataset import GarbageDataset, get_train_transforms, get_val_transforms
from train import set_seed, get_class_weights, evaluate, plot_curves, build_model, configure_stage2_backbone

def run_stage_2(
    base_checkpoint="checkpoints/best_model_stage1.pt",
    batch_size=32,
    num_workers=2,
    epochs=15,
    patience=4,
    lr_head=1e-4,
    lr_layer4=5e-5,
    lr_layer3=2e-5,
    label_smoothing=0.1
):
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- STARTING STAGE 2 TRAINING (FINE-TUNING LAST 2 RESIDUAL BLOCKS + HEAD) ---")
    print(f"Device: {device}")
    print(f"Loading Stage 1 base checkpoint: {base_checkpoint}")
    print(f"Batch Size: {batch_size}, Workers: {num_workers}, Max Epochs: {epochs}, Early Stopping Patience: {patience}")

    # Load labels
    with open("labels.json") as f:
        labels_info = json.load(f)
    class_names = labels_info["classes"]
    num_classes = len(class_names)

    # Build model and load Stage 1 weights
    model = build_model(num_classes=num_classes)
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded weights from Stage 1 (epoch {checkpoint.get('epoch', '?')}, val macro-F1: {checkpoint.get('val_macro_f1', 0.0):.4f})")

    # Configure Stage 2 (unfreeze layer3, layer4, fc)
    configure_stage2_backbone(model)
    model.to(device)

    # DataLoaders
    train_dataset = GarbageDataset("manifests/train_manifest.json", transform=get_train_transforms())
    val_dataset = GarbageDataset("manifests/val_manifest.json", transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    # Class-weighted loss
    class_weights = get_class_weights("reports/splits_summary.json", device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # Differential learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.layer3.parameters(), "lr": lr_layer3},
        {"params": model.layer4.parameters(), "lr": lr_layer4},
        {"params": model.fc.parameters(), "lr": lr_head}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_val_macro_f1 = checkpoint.get("val_macro_f1", 0.0)
    best_checkpoint_path = "checkpoints/best_model.pt"
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        val_res = evaluate(model, val_loader, criterion, device, class_names)
        val_loss = val_res["loss"]
        val_acc = val_res["accuracy"]
        val_macro_f1 = val_res["macro_f1"]
        epoch_time = time.time() - t0

        print(f"[Stage 2 Epoch {epoch:02d}/{epochs:02d}] "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro-F1: {val_macro_f1:.4f}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "epoch_time_sec": epoch_time,
            "per_class_f1": val_res["per_class_f1"]
        }
        history.append(epoch_record)

        # Save checkpoint every epoch
        epoch_ckpt = f"checkpoints/checkpoint_stage2_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "stage": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_macro_f1": val_macro_f1,
            "class_names": class_names
        }, epoch_ckpt)

        # Early stopping check & best model update
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "class_names": class_names
            }, best_checkpoint_path)
            print(f"  --> Saved new best Stage 2 checkpoint (Val Macro-F1: {best_val_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  --> No improvement for {epochs_without_improvement}/{patience} epochs (Best: {best_val_macro_f1:.4f})")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save summary and curves
    plot_curves(history, "reports/training_curves_stage2.png", stage_name="Stage 2 (Fine-Tuning)")
    with open("reports/stage2_metrics.json", "w") as f:
        json.dump({
            "stage": 2,
            "epochs_trained": len(history),
            "best_val_macro_f1": best_val_macro_f1,
            "history": history
        }, f, indent=2)

    print(f"\n--- STAGE 2 FINE-TUNING COMPLETE ---")
    print(f"Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Saved best checkpoint to: {best_checkpoint_path}")
    print(f"Saved metrics to: reports/stage2_metrics.json")
    print(f"Saved training curves to: reports/training_curves_stage2.png")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/best_model_stage1.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    run_stage_2(
        base_checkpoint=args.base_checkpoint,
        batch_size=args.batch_size,
        num_workers=args.workers,
        epochs=args.epochs,
        patience=args.patience
    )
