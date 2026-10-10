# Household Waste Material Classifier (ResNet50)

An image classification system for Indian household waste powered by a fine-tuned ResNet50 backbone. Designed for mobile phone camera inputs to classify single household items into 10 recyclable material categories and interface with scrap buyer (*kabadiwala*) rate cards.

---

## Target Classes (10 V1 Classes)
1. `battery` (used dry cells, lithium batteries)
2. `biological` (organic food/kitchen waste)
3. `cardboard` (corrugated cartons, boxes)
4. `clothes` (old fabric, garments)
5. `glass` (brown, green, and clear bottles/jars)
6. `metal` (aluminum cans, steel tins, scrap metal)
7. `paper` (newspapers, magazines, office scrap)
8. `plastic` (bottles, containers, household plastics)
9. `shoes` (footwear)
10. `trash` (non-recyclable domestic refuse)

---

## Project Structure
```
├── data/                               # Raw Kaggle hub dataset (unmodified)
├── manifests/                          # Reproducible split manifests (JSON & CSV)
│   ├── train_manifest.json
│   ├── val_manifest.json
│   └── test_manifest.json
├── checkpoints/                        # Model weights & training checkpoints
├── reports/                            # Metrics, confusion matrices, curves & calibration
├── src/                                # Source code
│   ├── inspect_dataset.py              # Data sanity check & pHash duplicate clustering
│   ├── prepare_splits.py               # Leak-free stratified splitting & class weighting
│   ├── dataset.py                      # Dataset class & realistic phone augmentations
│   ├── train.py                        # Stage 1 training (frozen backbone)
│   ├── train_stage2.py                 # Stage 2 fine-tuning (residual blocks unfrozen)
│   ├── evaluate.py                     # Evaluation suite & threshold calibration
│   └── export_onnx.py                  # ONNX export, verification & latency benchmarking
├── predict.py                          # CLI prediction tool with JSON API support
├── labels.txt                          # Canonical list of 10 class labels
├── labels.json                         # Class index mapping
├── requirements.txt                    # Pinned Python package dependencies
├── DATASETS.md                         # Dataset source, licensing, counts documentation
├── DECISIONS.md                        # Running architectural & engineering decision log
└── MODEL_CARD.md                       # Model card with intended use & critical limitations
```

---

## Quickstart & Reproduction Steps

### 1. Environment Setup
```bash
# Create local virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Inspect Dataset & Generate Splits
```bash
# Audit data, verify formats, compute perceptual hashes
python src/inspect_dataset.py

# Generate leak-free stratified 80/10/10 splits & compute class weights
python src/prepare_splits.py
```

### 3. Model Training
```bash
# Stage 1: Train linear head with frozen ResNet50 backbone (5 epochs)
python src/train.py --stage 1 --epochs 5 --batch-size 32

# Stage 2: Fine-tune last two residual blocks with early stopping
python src/train_stage2.py --epochs 15 --batch-size 32 --patience 4
```

### 4. Evaluation
```bash
# Evaluate on test set and ./india_photos/ (if populated)
python src/evaluate.py checkpoints/best_model.pt
```

### 5. ONNX Export & Benchmarking
```bash
# Export to ONNX, verify numerical parity on 50 images, benchmark CPU latency
python src/export_onnx.py checkpoints/best_model.pt
```

### 6. Predict on New Images
```bash
# Human-readable output:
python predict.py /path/to/test_image.jpg

# JSON API output:
python predict.py /path/to/test_image.jpg --json
```

---

## Real-World Domestic Photo Evaluation (`./india_photos/`)
Add real mobile photos to `./india_photos/<class_name>/` (e.g. `./india_photos/plastic/bottle.jpg`). Re-running `python src/evaluate.py` will automatically detect and report domestic accuracy separately from the benchmark dataset.
