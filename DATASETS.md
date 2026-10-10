# DATASETS DOCUMENTATION

## 1. Overview & Data Source
- **Dataset Name**: Garbage Classification (Enhanced & Corrected)
- **Kaggle Source**: `huberthamelin/garbage-classification-labels-corrections` (Version 4)
- **Local Directory Location**: `./data/datasets/huberthamelin/garbage-classification-labels-corrections/versions/4/`
- **License**: CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International) / Open Database License (ODbL)
- **Original Source**: Combines the original Gary Thung & Mindy Yang Garbage Classification dataset with additional web-scraped images and crowd-validated label corrections.

## 2. Directory Structure Discovered
Under version `4/`:
- `garbage_classification_corrected/`: Corrected annotations applied to the original 2,527 images.
- `garbage_classification_enhanced/`: The full enhanced dataset (15,870 images) combining corrected originals with supplemental web imagery.
- `labels_corrections.csv`: 65 ground-truth relabeling overrides correcting mistakes from earlier community releases.

> **Zero Leakage Rule**: Training uses **strictly and exclusively** `garbage_classification_enhanced/`. The raw data was never duplicated, moved, or altered.

---

## 3. Native Classes vs. V1 Target Classes

### Native 12 Classes (Enhanced Set Raw: 15,870 images)
| Native Class | Raw Count | Native % |
| :--- | :--- | :--- |
| `battery` | 979 | 6.17% |
| `biological` | 1,035 | 6.52% |
| `brown-glass` | 619 | 3.90% |
| `cardboard` | 936 | 5.90% |
| `clothes` | 5,286 | 33.31% |
| `green-glass` | 641 | 4.04% |
| `metal` | 837 | 5.27% |
| `paper` | 1,108 | 6.98% |
| `plastic` | 901 | 5.68% |
| `shoes` | 2,006 | 12.64% |
| `trash` | 726 | 4.57% |
| `white-glass` | 796 | 5.02% |

### V1 10 Target Classes
Glass subclasses (`brown-glass`, `green-glass`, `white-glass`) were unified into a single `glass` class to align with informal recycling scrap buyers (kabadiwalas):
`['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']`

---

## 4. Manifests & Splits (80 / 10 / 10 Stratified Cluster Splits)

To eliminate data leakage, perceptual hashing (pHash Hamming distance $\le 4$) was used to group near-duplicate web images into atomic clusters:
- **Corrupt images**: 0
- **Tiny images excluded**: 1 (`trash/trash622.jpg`, 51×400 px)
- **Conflicting duplicate clusters excluded**: 12 clusters (63 images) with discordant labels across classes.
- **Clothes Dominance Adjustment**: Capped at 1,800 images in the **training split only** (fixed seed 42) while preserving natural validation and test distributions.

### Final Per-Class Counts per Split

| Class Name | Training Split (80% capped) | Validation Split (10%) | Test Split (10%) | Total Included |
| :--- | :--- | :--- | :--- | :--- |
| `battery` | 780 | 98 | 97 | 975 |
| `biological` | 828 | 104 | 103 | 1,035 |
| `cardboard` | 748 | 94 | 93 | 935 |
| `clothes` | 1,800 (capped) | 529 | 528 | 2,857 |
| `glass` | 1,626 | 203 | 204 | 2,033 |
| `metal` | 666 | 83 | 83 | 832 |
| `paper` | 886 | 111 | 111 | 1,108 |
| `plastic` | 715 | 89 | 90 | 894 |
| `shoes` | 1,605 | 201 | 200 | 2,006 |
| `trash` | 562 | 70 | 70 | 702 |
| **Total** | **10,216** | **1,582** | **1,579** | **13,377** |

---

## 5. India Real-World Photos Staging (`./india_photos/`)
- Directory `./india_photos/` contains 10 subdirectories corresponding to the 10 V1 classes.
- Designed to accept real mobile phone captures of Indian household waste items (e.g. newspaper bundles, UPI milk pouches, blister packaging, rusted tins, plastic bottles).
- Evaluation pipeline automatically detects and scores against this set when images are present.
