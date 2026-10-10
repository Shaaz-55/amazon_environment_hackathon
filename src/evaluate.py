import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from dataset import load_image_with_white_bg, get_val_transforms
from train import build_model

def evaluate_predictions(probs, targets, paths, class_names, reports_dir="reports"):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "misclassified"), exist_ok=True)

    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    num_samples = len(targets)
    num_classes = len(class_names)

    # 1. Overall Accuracy & Macro-F1
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)

    # 2. Top-3 Accuracy
    top3_preds = np.argsort(probs, axis=1)[:, -3:]  # shape: (N, 3)
    top3_correct = [targets[i] in top3_preds[i] for i in range(num_samples)]
    top3_acc = np.mean(top3_correct)

    # 3. Per-class metrics
    clf_report = classification_report(
        targets,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    per_class_metrics = {}
    for cls in class_names:
        per_class_metrics[cls] = {
            "precision": float(clf_report[cls]["precision"]),
            "recall": float(clf_report[cls]["recall"]),
            "f1_score": float(clf_report[cls]["f1-score"]),
            "support": int(clf_report[cls]["support"])
        }

    # 4. Confusion Matrix
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(reports_dir, "confusion_matrix.csv"))

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # 5. Top 5 Most Confused Pairs
    # Find largest off-diagonal entries
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "pred_class": class_names[j],
                    "count": int(cm[i, j])
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    top_5_confused = confused_pairs[:5]

    # Save example misclassified images
    misclassified_examples = []
    saved_counts = {}
    for i in range(num_samples):
        true_lbl = targets[i]
        pred_lbl = preds[i]
        if true_lbl != pred_lbl:
            t_name = class_names[true_lbl]
            p_name = class_names[pred_lbl]
            pair_key = f"{t_name}_as_{p_name}"
            saved_counts[pair_key] = saved_counts.get(pair_key, 0) + 1

            if saved_counts[pair_key] <= 3:  # Save up to 3 per pair
                src_path = paths[i]
                dst_fname = f"misclass_{pair_key}_{saved_counts[pair_key]}.jpg"
                dst_path = os.path.join(reports_dir, "misclassified", dst_fname)
                try:
                    shutil.copy2(src_path, dst_path)
                    misclassified_examples.append({
                        "true_class": t_name,
                        "pred_class": p_name,
                        "confidence": float(confidences[i]),
                        "source_path": src_path,
                        "saved_example": dst_path
                    })
                except Exception as e:
                    pass

    # 6. Calibration Analysis (ECE & Reliability Diagram)
    num_bins = 10
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_assignments = np.digitize(confidences, bins) - 1
    # ensure within [0, num_bins-1]
    bin_assignments = np.clip(bin_assignments, 0, num_bins - 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0

    for b in range(num_bins):
        in_bin = bin_assignments == b
        count = np.sum(in_bin)
        bin_counts.append(int(count))
        if count > 0:
            bin_acc = np.mean(preds[in_bin] == targets[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_accuracies.append(float(bin_acc))
            bin_confidences.append(float(bin_conf))
            ece += (count / num_samples) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((bins[b] + bins[b+1]) / 2.0))

    # Specifically check: is 90% confidence right about 90% of the time?
    high_conf_mask = confidences >= 0.90
    high_conf_count = int(np.sum(high_conf_mask))
    if high_conf_count > 0:
        high_conf_acc = float(np.mean(preds[high_conf_mask] == targets[high_conf_mask]))
        high_conf_avg = float(np.mean(confidences[high_conf_mask]))
    else:
        high_conf_acc = 0.0
        high_conf_avg = 0.0

    # Plot Calibration Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(bin_confidences, bin_accuracies, "s-", color="blue", label="Model Reliability")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "calibration_curve.png"), dpi=200)
    plt.close()

    # 7. Confidence Threshold & Coverage Trade-Off Analysis
    thresholds = np.arange(0.20, 0.96, 0.05)
    tradeoff = []
    recommended_threshold = 0.60

    for thresh_val in thresholds:
        passed = confidences >= thresh_val
        coverage = np.mean(passed)
        if np.sum(passed) > 0:
            filtered_acc = np.mean(preds[passed] == targets[passed])
        else:
            filtered_acc = 1.0

        tradeoff.append({
            "threshold": round(float(thresh_val), 2),
            "coverage": round(float(coverage), 4),
            "accuracy": round(float(filtered_acc), 4)
        })

    # Pick threshold with >= 92% accuracy and high coverage, or best balanced
    best_candidate = None
    for item in tradeoff:
        if item["accuracy"] >= 0.92 and item["coverage"] >= 0.80:
            best_candidate = item["threshold"]
    if best_candidate is None:
        # fallback to threshold that maximizes coverage * accuracy
        best_candidate = max(tradeoff, key=lambda x: x["coverage"] * x["accuracy"])["threshold"]
    recommended_threshold = best_candidate

    # Plot Trade-off
    fig, ax1 = plt.subplots(figsize=(8, 5))
    thresh_list = [t["threshold"] for t in tradeoff]
    cov_list = [t["coverage"] * 100 for t in tradeoff]
    acc_list = [t["accuracy"] * 100 for t in tradeoff]

    ax1.plot(thresh_list, cov_list, 'b-o', label='Coverage (%)')
    ax1.plot(thresh_list, acc_list, 'g-s', label='Filtered Accuracy (%)')
    ax1.axvline(x=recommended_threshold, color='red', linestyle='--', label=f'Recommended: {recommended_threshold:.2f}')
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Coverage vs. Accuracy Trade-Off')
    ax1.legend(loc='lower left')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confidence_threshold_tradeoff.png"), dpi=200)
    plt.close()

    results = {
        "overall_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "top3_accuracy": float(top3_acc),
        "total_test_samples": num_samples,
        "per_class_metrics": per_class_metrics,
        "top_5_confused_pairs": top_5_confused,
        "calibration": {
            "expected_calibration_error": float(ece),
            "high_confidence_subset_90": {
                "threshold": 0.90,
                "count": high_conf_count,
                "percentage_of_data": round(high_conf_count / num_samples * 100, 2),
                "average_confidence": high_conf_avg,
                "actual_accuracy": high_conf_acc
            }
        },
        "threshold_analysis": {
            "recommended_threshold": float(recommended_threshold),
            "tradeoff_table": tradeoff
        },
        "misclassified_examples": misclassified_examples
    }

    return results

class FolderDataset(Dataset):
    def __init__(self, folder_root, class_names, transform=None):
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.transform = transform
        for cls in class_names:
            cls_dir = os.path.join(folder_root, cls)
            if os.path.exists(cls_dir):
                for fname in os.listdir(cls_dir):
                    if not fname.startswith('.'):
                        fpath = os.path.join(cls_dir, fname)
                        if os.path.isfile(fpath):
                            self.samples.append((fpath, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_image_with_white_bg(path)
        if self.transform:
            img = self.transform(img)
        return img, label, path

def run_evaluation(checkpoint_path="checkpoints/best_model.pt"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading checkpoint for evaluation: {checkpoint_path}")

    with open("labels.json") as f:
        labels_info = json.load(f)
    class_names = labels_info["classes"]
    num_classes = len(class_names)

    model = build_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_transforms = get_val_transforms()

    # 1. Evaluate on Test Set Manifest
    test_manifest_path = "manifests/test_manifest.json"
    with open(test_manifest_path) as f:
        test_samples = json.load(f)
    print(f"\nEvaluating on held-out test set ({len(test_samples)} images)...")

    from dataset import GarbageDataset
    test_dataset = GarbageDataset(test_manifest_path, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    all_probs = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        for inputs, targets, paths in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.extend(targets.numpy())
            all_paths.extend(paths)

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.array(all_targets)

    test_results = evaluate_predictions(all_probs, all_targets, all_paths, class_names, reports_dir="reports/test_eval")

    # 2. Check and Evaluate on india_photos/
    india_root = "./india_photos"
    india_dataset = FolderDataset(india_root, class_names, transform=val_transforms)
    india_results = None

    if len(india_dataset) > 0:
        print(f"\nFound {len(india_dataset)} real-world phone photos in {india_root}! Evaluating...")
        india_loader = DataLoader(india_dataset, batch_size=16, shuffle=False, num_workers=1)
        i_probs, i_targets, i_paths = [], [], []
        with torch.no_grad():
            for inputs, targets, paths in india_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                i_probs.append(probs)
                i_targets.extend(targets.numpy())
                i_paths.extend(paths)
        i_probs = np.concatenate(i_probs, axis=0)
        i_targets = np.array(i_targets)
        india_results = evaluate_predictions(i_probs, i_targets, i_paths, class_names, reports_dir="reports/india_photos_eval")
    else:
        print(f"\nNotice: {india_root} is currently empty. Real-world performance on Indian household photos is UNTESTED.")
        india_results = {
            "status": "UNTESTED",
            "reason": "Directory ./india_photos/ is currently empty.",
            "recommendation": "Collect 25-50 photos per class under realistic Indian domestic lighting, backgrounds, and angles (e.g. newspaper stacks, milk pouches, steel cans, broken bottles, battery cells)."
        }

    full_evaluation = {
        "checkpoint_used": checkpoint_path,
        "classes": class_names,
        "test_set_evaluation": test_results,
        "india_photos_evaluation": india_results
    }

    with open("reports/test_evaluation.json", "w") as f:
        json.dump(full_evaluation, f, indent=2)

    print("\nEvaluation complete! Report saved to reports/test_evaluation.json")
    return full_evaluation

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    run_evaluation(ckpt)
