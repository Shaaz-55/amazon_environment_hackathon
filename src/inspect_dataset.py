import os
import glob
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import imagehash
import numpy as np

def find_dataset_version_dir(base_data_path="./data/datasets/huberthamelin/garbage-classification-labels-corrections/versions"):
    if not os.path.exists(base_data_path):
        raise FileNotFoundError(f"Base path not found: {base_data_path}")
    versions = [d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d))]
    if not versions:
        raise FileNotFoundError(f"No version folders found in {base_data_path}")
    # Sort version folders (handle numeric or string versions)
    versions.sort(key=lambda v: int(v) if v.isdigit() else v)
    latest_version = versions[-1]
    return os.path.join(base_data_path, latest_version)

def inspect_dataset():
    version_dir = find_dataset_version_dir()
    enhanced_dir = os.path.join(version_dir, "garbage_classification_enhanced")
    if not os.path.exists(enhanced_dir):
        raise FileNotFoundError(f"Enhanced dataset directory not found at: {enhanced_dir}")
    
    print(f"Inspecting dataset at: {enhanced_dir}")
    classes = sorted([d for d in os.listdir(enhanced_dir) if os.path.isdir(os.path.join(enhanced_dir, d)) and not d.startswith('.')])
    print(f"Found {len(classes)} classes: {classes}")

    class_counts = {}
    corrupt_files = []
    tiny_images = []
    odd_format_images = []
    image_metadata = []
    
    # Track hashes for near-duplicate detection
    # Each entry: (hash_int, file_path, class_name)
    hashes_list = []
    
    reports_sample_dir = Path("reports/samples")
    reports_sample_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for cls in classes:
        cls_dir = os.path.join(enhanced_dir, cls)
        cls_files = sorted([f for f in os.listdir(cls_dir) if not f.startswith('.')])
        class_counts[cls] = len(cls_files)
        
        valid_samples = []
        
        for fname in cls_files:
            fpath = os.path.join(cls_dir, fname)
            try:
                with Image.open(fpath) as img:
                    # Check for corruption by reading data
                    img.load()
                    width, height = img.size
                    mode = img.mode
                    img_format = img.format or os.path.splitext(fname)[1].lstrip('.').upper()
                    
                    is_bad = False
                    
                    # Check tiny
                    if width < 64 or height < 64:
                        tiny_images.append({
                            "path": fpath,
                            "class": cls,
                            "width": width,
                            "height": height
                        })
                        is_bad = True
                    
                    # Check odd formats or grayscale
                    if mode not in ["RGB"]:
                        odd_format_images.append({
                            "path": fpath,
                            "class": cls,
                            "mode": mode,
                            "format": img_format,
                            "size": (width, height)
                        })
                        # Grayscale or RGBA isn't necessarily excluded unless corrupt, but we note it
                    
                    # Compute perceptual hash
                    phash = imagehash.phash(img.convert('RGB'))
                    phash_int = int(str(phash), 16)
                    
                    hashes_list.append({
                        "hash_int": phash_int,
                        "hash_str": str(phash),
                        "path": fpath,
                        "class": cls,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "format": img_format,
                        "is_bad": is_bad
                    })
                    
                    if not is_bad and len(valid_samples) < 3:
                        valid_samples.append((fpath, fname))
                        
            except Exception as e:
                corrupt_files.append({
                    "path": fpath,
                    "class": cls,
                    "error": str(e)
                })
        
        # Save 3 sample images for inspection
        sample_cls_dir = reports_sample_dir / cls
        sample_cls_dir.mkdir(parents=True, exist_ok=True)
        for idx, (sfpath, sfname) in enumerate(valid_samples, start=1):
            dst = sample_cls_dir / f"sample_{idx}_{sfname}"
            shutil.copy2(sfpath, dst)
            
    print(f"Scanned {len(hashes_list)} valid images in {time.time() - start_time:.2f} seconds.")
    print(f"Corrupt files count: {len(corrupt_files)}")
    print(f"Tiny images (<64px) count: {len(tiny_images)}")
    print(f"Non-RGB / Odd format images count: {len(odd_format_images)}")

    # Near-duplicate detection using perceptual hash
    # Hamming distance threshold <= 4 indicates near duplicate
    print("Finding near-duplicates using perceptual hashing (pHash distance <= 4)...")
    hash_dup_start = time.time()
    
    # Bucket by 16-bit blocks for fast candidate retrieval
    # A 64-bit hash has 4 chunks of 16 bits. If hamming dist <= 4, at least one chunk has at most 1 bit difference,
    # or we can index on exact 16-bit chunks (pigeonhole: at distance <= 3, at least one 16-bit chunk matches exactly).
    # Since N is ~15,000, 64-bit uint popcount comparison is also very fast with numpy!
    hash_array = np.array([item["hash_int"] for item in hashes_list], dtype=np.uint64)
    n = len(hash_array)
    
    # We can group near duplicates into connected components (clusters)
    # Using Union-Find
    parent = list(range(n))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    # To be extremely fast and exact: chunk indexing with bucket matching
    # 64-bit hash split into four 16-bit keys
    buckets = [defaultdict(list) for _ in range(4)]
    for i, h in enumerate(hash_array):
        val = int(h)
        for chunk_idx in range(4):
            key = (val >> (chunk_idx * 16)) & 0xFFFF
            buckets[chunk_idx][key].append(i)

    candidate_pairs = set()
    for chunk_idx in range(4):
        for key, indices in buckets[chunk_idx].items():
            if len(indices) > 1:
                for idx_a in range(len(indices)):
                    for idx_b in range(idx_a + 1, len(indices)):
                        i, j = indices[idx_a], indices[idx_b]
                        candidate_pairs.add((min(i, j), max(i, j)))

    near_duplicate_pairs = []
    HAMMING_THRESHOLD = 4

    for i, j in candidate_pairs:
        # Compute hamming distance
        xor_val = int(hash_array[i] ^ hash_array[j])
        dist = bin(xor_val).count('1')
        if dist <= HAMMING_THRESHOLD:
            near_duplicate_pairs.append({
                "img1": hashes_list[i]["path"],
                "class1": hashes_list[i]["class"],
                "img2": hashes_list[j]["path"],
                "class2": hashes_list[j]["class"],
                "distance": dist
            })
            union(i, j)

    # Group into duplicate clusters
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    dup_clusters = [members for members in clusters.values() if len(members) > 1]
    total_images_in_dup_groups = sum(len(m) for m in dup_clusters)

    print(f"Perceptual hash analysis completed in {time.time() - hash_dup_start:.2f} seconds.")
    print(f"Found {len(near_duplicate_pairs)} near-duplicate pairs across {len(dup_clusters)} clusters ({total_images_in_dup_groups} total images involved).")

    # Save summary report
    inspection_report = {
        "version_dir": version_dir,
        "enhanced_dir": enhanced_dir,
        "class_counts": class_counts,
        "total_images": sum(class_counts.values()),
        "corrupt_files_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "tiny_images_count": len(tiny_images),
        "tiny_images": tiny_images,
        "odd_format_count": len(odd_format_images),
        "odd_format_samples": odd_format_images[:20],
        "near_duplicate_pairs_count": len(near_duplicate_pairs),
        "duplicate_clusters_count": len(dup_clusters),
        "total_images_in_dup_groups": total_images_in_dup_groups,
        "sample_clusters": [
            [hashes_list[idx]["path"] for idx in cluster[:5]]
            for cluster in dup_clusters[:10]
        ]
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/dataset_inspection.json", "w") as f:
        json.dump(inspection_report, f, indent=2)

    # Save complete image records with cluster IDs for Phase 3 splits
    records = []
    for i, item in enumerate(hashes_list):
        records.append({
            "path": item["path"],
            "class": item["class"],
            "hash_str": item["hash_str"],
            "cluster_id": find(i),
            "width": item["width"],
            "height": item["height"],
            "mode": item["mode"],
            "format": item["format"],
            "is_bad": item["is_bad"]
        })
    with open("reports/image_records.json", "w") as f:
        json.dump(records, f)

    print("Inspection report saved to reports/dataset_inspection.json")
    print("Full image records saved to reports/image_records.json")

    return inspection_report

if __name__ == "__main__":
    inspect_dataset()
