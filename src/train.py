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
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

# Enable MPS CPU fallback for unsupported operators if any
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dataset import GarbageDataset, get_train_transforms, get_val_transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_class_weights(summary_path="reports/splits_summary.json", device="cpu"):
    with open(summary_path) as f:
        summary = json.load(f)
    classes = summary["classes"]
    weights_dict = summary["class_weights"]
    weights = [weights_dict[cls] for cls in classes]
    return torch.tensor(weights, dtype=torch.float32, device=device)

def build_model(num_classes=10):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    # Replace final classification head
    model.fc = nn.Linear(in_features, num_classes)
    return model

def configure_stage1_backbone(model):
    """Stage 1: Backbone frozen, only linear head trainable."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def configure_stage2_backbone(model):
    """Stage 2: Unfreeze last two residual blocks (layer3, layer4) and head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

def evaluate(model, dataloader, criterion, device, class_names):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    f1_by_class = {cls: float(per_class_f1[i]) for i, cls in enumerate(class_names)}

    return {
        "loss": epoch_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": f1_by_class,
        "preds": all_preds,
        "targets": all_targets
    }

def plot_curves(history, out_path, stage_name="Stage 1"):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, train_loss, 'o-', label='Train Loss')
    axes[0].plot(epochs, val_loss, 's-', label='Val Loss')
    axes[0].set_title(f'{stage_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, train_acc, 'o-', label='Train Accuracy')
    axes[1].plot(epochs, val_acc, 's-', label='Val Accuracy')
    axes[1].set_title(f'{stage_name} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Macro-F1
    axes[2].plot(epochs, val_macro_f1, 'd-', color='green', label='Val Macro-F1')
    axes[2].set_title(f'{stage_name} - Validation Macro-F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro-F1')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def run_stage_1(batch_size=32, num_workers=2, epochs=5, lr=1e-3, label_smoothing=0.1):
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- STARTING STAGE 1 TRAINING (FROZEN BACKBONE, HEAD ONLY) ---")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}, Workers: {num_workers}, Epochs: {epochs}, LR: {lr}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load classes
    with open("labels.json") as f:
        labels_info = json.load(f)
    class_names = labels_info["classes"]
    num_classes = len(class_names)

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

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = build_model(num_classes=num_classes)
    configure_stage1_backbone(model)
    model.to(device)

    # Loss & Weights
    class_weights = get_class_weights("reports/splits_summary.json", device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_macro_f1 = 0.0
    best_checkpoint_path = "checkpoints/best_model_stage1.pt"

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

        print(f"[Epoch {epoch:02d}/{epochs:02d}] "
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

        # Save latest epoch checkpoint
        epoch_ckpt = f"checkpoints/checkpoint_stage1_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "stage": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_macro_f1": val_macro_f1,
            "class_names": class_names
        }, epoch_ckpt)

        # Save best model
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save({
                "epoch": epoch,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "class_names": class_names
            }, best_checkpoint_path)
            print(f"  --> Saved new best Stage 1 checkpoint (Val Macro-F1: {best_val_macro_f1:.4f})")

    # Save summary and curves
    plot_curves(history, "reports/training_curves_stage1.png", stage_name="Stage 1 (Head Only)")
    with open("reports/stage1_metrics.json", "w") as f:
        json.dump({
            "stage": 1,
            "epochs_trained": epochs,
            "best_val_macro_f1": best_val_macro_f1,
            "history": history
        }, f, indent=2)

    print(f"\n--- STAGE 1 TRAINING COMPLETE ---")
    print(f"Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Saved best checkpoint to: {best_checkpoint_path}")
    print(f"Saved metrics to: reports/stage1_metrics.json")
    print(f"Saved training curves to: reports/training_curves_stage1.png")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Stage to train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.stage == 1:
        run_stage_1(batch_size=args.batch_size, num_workers=args.workers, epochs=args.epochs, lr=args.lr)
    else:
        print("Stage 2 script to be executed after Stage 1 pause and user confirmation.")
