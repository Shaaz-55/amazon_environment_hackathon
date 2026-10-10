import os
import json
import random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

def run_phase_2_and_3():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # 1. Define V1 Class Mapping
    V1_CLASSES = [
        "battery",
        "biological",
        "cardboard",
        "clothes",
        "glass",
        "metal",
        "paper",
        "plastic",
        "shoes",
        "trash"
    ]
    class_to_idx = {cls: idx for idx, cls in enumerate(V1_CLASSES)}

    def map_to_v1(native_class):
        if native_class in ["brown-glass", "green-glass", "white-glass"]:
            return "glass"
        return native_class

    # Save labels file
    os.makedirs("manifests", exist_ok=True)
    with open("labels.txt", "w") as f:
        for cls in V1_CLASSES:
            f.write(f"{cls}\n")

    with open("labels.json", "w") as f:
        json.dump({
            "classes": V1_CLASSES,
            "class_to_idx": class_to_idx,
            "idx_to_class": {idx: cls for idx, cls in enumerate(V1_CLASSES)}
        }, f, indent=2)

    print("Phase 2: Saved labels.txt and labels.json")
    print(f"Target classes ({len(V1_CLASSES)}): {V1_CLASSES}")

    # Load image records from inspection
    with open("reports/image_records.json") as f:
        records = json.load(f)

    # Attach v1 class
    for rec in records:
        rec["v1_class"] = map_to_v1(rec["class"])

    # 2. Check for conflicting near-duplicate clusters (instruction 3)
    # Group by cluster_id
    clusters = defaultdict(list)
    for rec in records:
        clusters[rec["cluster_id"]].append(rec)

    conflicting_clusters = []
    conflicting_image_paths = set()

    for cid, members in clusters.items():
        if len(members) > 1:
            classes_in_cluster = set(m["v1_class"] for m in members)
            if len(classes_in_cluster) > 1:
                conflicting_clusters.append({
                    "cluster_id": cid,
                    "classes": list(classes_in_cluster),
                    "members": [{"path": m["path"], "native_class": m["class"], "v1_class": m["v1_class"]} for m in members]
                })
                for m in members:
                    conflicting_image_paths.add(m["path"])

    print(f"\nConflicting near-duplicate clusters found: {len(conflicting_clusters)}")
    print(f"Images excluded due to conflicting near-duplicates: {len(conflicting_image_paths)}")
    for conf in conflicting_clusters:
        print(f"  Cluster {conf['cluster_id']} has conflicting classes {conf['classes']}:")
        for m in conf['members']:
            print(f"    - {m['path']} ({m['native_class']} -> {m['v1_class']})")

    # 3. Filter valid records
    # Exclude: is_bad (tiny images like trash622.jpg) and conflicting near-duplicates
    excluded_tiny = []
    clean_records = []

    for rec in records:
        if rec["is_bad"]:
            excluded_tiny.append(rec["path"])
            continue
        if rec["path"] in conflicting_image_paths:
            continue
        clean_records.append(rec)

    print(f"\nTotal scanned images: {len(records)}")
    print(f"Excluded tiny images: {len(excluded_tiny)} ({excluded_tiny})")
    print(f"Excluded conflicting duplicate images: {len(conflicting_image_paths)}")
    print(f"Remaining valid clean images: {len(clean_records)}")

    # 4. Group clean records by cluster_id for atomic splitting
    clean_clusters_by_class = defaultdict(list)
    # Each cluster belongs to a dominant class (since conflicting clusters are removed, all members in a cluster have identical v1_class)
    clean_clusters = defaultdict(list)
    for rec in clean_records:
        clean_clusters[rec["cluster_id"]].append(rec)

    for cid, members in clean_clusters.items():
        cls = members[0]["v1_class"]
        clean_clusters_by_class[cls].append(members)

    # 5. Perform Stratified 80 / 10 / 10 Split at cluster level
    train_records = []
    val_records = []
    test_records = []

    for cls in V1_CLASSES:
        cls_cluster_list = clean_clusters_by_class[cls]
        # Shuffle clusters with fixed seed
        random.shuffle(cls_cluster_list)

        total_cls_images = sum(len(c) for c in cls_cluster_list)
        target_train_count = int(round(total_cls_images * 0.80))
        target_val_count = int(round(total_cls_images * 0.10))

        cls_train, cls_val, cls_test = [], [], []
        curr_train, curr_val = 0, 0

        for cluster in cls_cluster_list:
            cluster_len = len(cluster)
            if curr_train + cluster_len <= target_train_count or (curr_val >= target_val_count and curr_train < target_train_count):
                cls_train.extend(cluster)
                curr_train += cluster_len
            elif curr_val + cluster_len <= target_val_count or (len(cls_test) >= (total_cls_images - target_train_count - target_val_count) and curr_val < target_val_count):
                cls_val.extend(cluster)
                curr_val += cluster_len
            else:
                cls_test.extend(cluster)

        train_records.extend(cls_train)
        val_records.extend(cls_val)
        test_records.extend(cls_test)

    print(f"\nInitial Split Counts (before capping clothes in train):")
    print(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # 6. Apply clothes capping on TRAINING split only (Instruction 1: cap clothes at ~1,800 images)
    TARGET_CLOTHES_TRAIN = 1800
    clothes_train = [r for r in train_records if r["v1_class"] == "clothes"]
    non_clothes_train = [r for r in train_records if r["v1_class"] != "clothes"]

    print(f"Original clothes count in train: {len(clothes_train)}")
    # Group clothes by cluster to preserve cluster atomicity even when capping!
    clothes_clusters = defaultdict(list)
    for r in clothes_train:
        clothes_clusters[r["cluster_id"]].append(r)

    clothes_cluster_list = list(clothes_clusters.values())
    random.shuffle(clothes_cluster_list)

    sampled_clothes_train = []
    curr_clothes = 0
    for cluster in clothes_cluster_list:
        if curr_clothes + len(cluster) <= TARGET_CLOTHES_TRAIN:
            sampled_clothes_train.extend(cluster)
            curr_clothes += len(cluster)
        elif curr_clothes < TARGET_CLOTHES_TRAIN:
            # take part of cluster if needed, or stop
            needed = TARGET_CLOTHES_TRAIN - curr_clothes
            sampled_clothes_train.extend(cluster[:needed])
            curr_clothes += needed
            break
        else:
            break

    print(f"Capped clothes in train to: {len(sampled_clothes_train)}")
    final_train_records = non_clothes_train + sampled_clothes_train
    random.shuffle(final_train_records)

    # 7. Compute Class Weights for Weighted Loss
    # Formula: weight[c] = total_samples / (n_classes * count[c])
    train_counts = Counter(r["v1_class"] for r in final_train_records)
    n_classes = len(V1_CLASSES)
    total_train = len(final_train_records)
    class_weights = {}
    for cls in V1_CLASSES:
        cnt = train_counts[cls]
        class_weights[cls] = round(total_train / (n_classes * cnt), 4)

    print("\nTraining Class Weights (for loss function):")
    for cls in V1_CLASSES:
        print(f"  {cls:12s}: count={train_counts[cls]:4d}, weight={class_weights[cls]:.4f}")

    # 8. Save Manifests
    def format_manifest(recs):
        return [
            {
                "path": r["path"],
                "v1_class": r["v1_class"],
                "class_idx": class_to_idx[r["v1_class"]],
                "native_class": r["class"],
                "cluster_id": r["cluster_id"]
            }
            for r in recs
        ]

    train_manifest = format_manifest(final_train_records)
    val_manifest = format_manifest(val_records)
    test_manifest = format_manifest(test_records)

    with open("manifests/train_manifest.json", "w") as f:
        json.dump(train_manifest, f, indent=2)
    with open("manifests/val_manifest.json", "w") as f:
        json.dump(val_manifest, f, indent=2)
    with open("manifests/test_manifest.json", "w") as f:
        json.dump(test_manifest, f, indent=2)

    pd.DataFrame(train_manifest).to_csv("manifests/train_manifest.csv", index=False)
    pd.DataFrame(val_manifest).to_csv("manifests/val_manifest.csv", index=False)
    pd.DataFrame(test_manifest).to_csv("manifests/test_manifest.csv", index=False)

    # Summary table
    val_counts = Counter(r["v1_class"] for r in val_records)
    test_counts = Counter(r["v1_class"] for r in test_records)

    summary = {
        "classes": V1_CLASSES,
        "conflicting_clusters": conflicting_clusters,
        "excluded_tiny": excluded_tiny,
        "class_weights": class_weights,
        "splits": {
            "train": {
                "total": len(train_manifest),
                "counts": {cls: train_counts[cls] for cls in V1_CLASSES}
            },
            "val": {
                "total": len(val_manifest),
                "counts": {cls: val_counts[cls] for cls in V1_CLASSES}
            },
            "test": {
                "total": len(test_manifest),
                "counts": {cls: test_counts[cls] for cls in V1_CLASSES}
            }
        }
    }

    with open("reports/splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPhase 2 and Phase 3 completed successfully!")
    print(f"Saved manifests to manifests/ (train: {len(train_manifest)}, val: {len(val_manifest)}, test: {len(test_manifest)})")

    return summary

if __name__ == "__main__":
    run_phase_2_and_3()
