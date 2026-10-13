# Household Waste Material Classifier (ResNet50)

An image classifier for Indian household waste. Point a phone camera at a single item and it tells you what the item is made of, so the result can be matched to scrap-buyer (kabadiwala) rates. Built on a fine-tuned ResNet50 and exported to ONNX, it classifies an image in about 12 ms on a laptop CPU.

> **Status:** model training, evaluation and ONNX export are complete. The web app, rate cards and AWS deployment are in progress (see [Roadmap](#roadmap)). Performance on real Indian household photos has **not been measured yet** (see [Limitations](#limitations)).

## Why this exists

Many households throw away items a kabadiwala would pay for: newspaper, cardboard, metal scrap, some plastics and glass. The goal is a simple tool that answers "what is this, and is it worth selling?" from one photo.

## Results

Trained on the Garbage Classification ENHANCED dataset (15,870 images, merged into 10 classes) with leak-free splits (near-duplicate images are always kept in the same split).

**Held-out test set (1,579 images, evaluated once):**

| Metric | Result |
|---|---|
| Accuracy | 97.53% (1,540 / 1,579) |
| Macro-F1 | 0.9673 |
| Top-3 accuracy | 99.68% |
| Accuracy when confidence ≥ 0.90 | 100% (388 / 388 images) |

**Confidence threshold.** Predictions below 0.60 confidence return `unsure: true` (the app should ask the user to try another angle). At this threshold, 95.1% of test images get an answer, and those answers are 99.1% accurate.

| Threshold | Coverage | Accuracy on accepted images |
|---|---|---|
| ≥ 0.50 | 97.78% | 98.25% |
| **≥ 0.60 (default)** | **95.12%** | **99.07%** |
| ≥ 0.70 | 91.32% | 99.38% |
| ≥ 0.75 | 85.50% | 99.63% |
| ≥ 0.90 | 24.57% | 100% |

The model is under-confident (expected calibration error 0.15), so raw confidence numbers should not be shown to users as probabilities.

**Validation F1 per class, before and after fine-tuning:**

| Class | Stage 1 (head only) | Stage 2 (fine-tuned) |
|---|---|---|
| battery | 0.9406 | 0.9744 |
| biological | 0.9904 | 0.9856 |
| cardboard | 0.9091 | 0.9399 |
| clothes | 0.9905 | 0.9981 |
| glass | 0.9160 | 0.9655 |
| metal | 0.8506 | 0.9647 |
| paper | 0.8837 | 0.9315 |
| plastic | 0.8619 | 0.9153 |
| shoes | 0.9752 | 0.9901 |
| trash | 0.8933 | 0.9379 |
| **Overall** | **0.9211 macro-F1 / 94.44% acc** | **0.9603 macro-F1 / 97.35% acc** |

**Most common test errors:** glass vs metal (shiny reflections), glass vs plastic (transparent containers), and cardboard vs paper (thin card stock).

**Export and speed (Apple M4, CPU):** 89.68 MB ONNX model, 12.04 ms mean latency per image, maximum difference from the PyTorch model 3.7 × 10⁻⁵ over 50 images.

## Classes

| Class | Examples |
|---|---|
| `battery` | used dry cells, lithium batteries |
| `biological` | organic food and kitchen waste |
| `cardboard` | corrugated cartons, boxes |
| `clothes` | old fabric, garments |
| `glass` | brown, green and clear bottles and jars (three source classes merged) |
| `metal` | aluminium cans, steel tins, scrap metal |
| `paper` | newspapers, magazines, office scrap |
| `plastic` | bottles, containers, household plastics |
| `shoes` | footwear |
| `trash` | non-recyclable domestic refuse |

## Quickstart: run the trained model

```bash
git clone https://github.com/Shaaz-55/amazon_environment_hackathon.git
cd amazon_environment_hackathon

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download `model.onnx` from the **[Releases page](https://github.com/Shaaz-55/amazon_environment_hackathon/releases)** (the file is ~90 MB, so it is not stored in the git history) and place it in the project root. Then:

```bash
# Human-readable output
python predict.py /path/to/photo.jpg

# JSON output (stable format, intended for a future API)
python predict.py /path/to/photo.jpg --json
```

Example JSON output:

```json
{
  "image": "photo.jpg",
  "top3": [
    {"class": "plastic", "confidence": 0.9592},
    {"class": "trash", "confidence": 0.0112},
    {"class": "battery", "confidence": 0.0051}
  ],
  "top_class": "plastic",
  "confidence": 0.9592,
  "threshold": 0.60,
  "unsure": false,
  "decision": "Accepted (plastic)"
}
```

Model input: one RGB image resized to 224 × 224, normalised with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. The output class order is in `labels.txt`.

## Reproduce the training

Tested on macOS (Apple M4, 16 GB) with Python 3.14 and PyTorch's MPS backend. The full run took about an hour: Stage 1 around 2 minutes per epoch, Stage 2 around 3 minutes per epoch, with early stopping at epoch 12 (best epoch 8).

**1. Environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Download the dataset into the project folder**

```bash
KAGGLEHUB_CACHE=./data python -c "import kagglehub; print(kagglehub.dataset_download('huberthamelin/garbage-classification-labels-corrections'))"
```

Only the `garbage_classification_enhanced/` folder is used for training. The corrected-original folder is a subset of it, and using both would leak duplicate images across splits.

**3. Inspect the data and build splits**

```bash
# Audit data, check formats, compute perceptual hashes for near-duplicate clusters
python src/inspect_dataset.py

# Leak-free stratified 80/10/10 splits and class weights
python src/prepare_splits.py
```

Details that affect results: the three glass colours are merged into `glass`; one tiny image and 12 near-duplicate clusters with conflicting labels are excluded; and in the **training split only**, `clothes` is capped at 1,800 images because it makes up about a third of the raw data. Validation and test sets are untouched. See `DECISIONS.md` for the full log.

**4. Train**

```bash
# Stage 1: new classification head, frozen ResNet50 backbone (5 epochs)
python src/train.py --stage 1 --epochs 5 --batch-size 32

# Stage 2: fine-tune the last two residual blocks, with early stopping
python src/train_stage2.py --epochs 15 --batch-size 32 --patience 4
```

**5. Evaluate**

```bash
# Scores the held-out test set, and ./india_photos/ separately if it has images
python src/evaluate.py checkpoints/best_model.pt
```

**6. Export to ONNX**

```bash
# Exports, checks parity on 50 images, and benchmarks CPU latency
python src/export_onnx.py checkpoints/best_model.pt
```

## Project structure

```
├── src/
│   ├── inspect_dataset.py     # Data audit and perceptual-hash duplicate clustering
│   ├── prepare_splits.py      # Leak-free stratified splits and class weights
│   ├── dataset.py             # Dataset class and phone-style augmentations
│   ├── train.py               # Stage 1: frozen backbone
│   ├── train_stage2.py        # Stage 2: fine-tuning
│   ├── evaluate.py            # Evaluation suite and threshold analysis
│   └── export_onnx.py         # ONNX export, verification, latency benchmark
├── manifests/                 # Reproducible train/val/test split lists (JSON and CSV)
├── reports/                   # Confusion matrix, calibration, threshold trade-off, sample errors
├── india_photos/              # Your own real-world test photos (one subfolder per class)
├── predict.py                 # Command-line prediction tool with JSON output
├── labels.txt / labels.json   # Class order used by the model
├── requirements.txt           # Pinned dependencies
├── DATASETS.md                # Dataset sources, licenses, counts
├── DECISIONS.md               # Engineering decision log
└── MODEL_CARD.md              # Intended use, metrics, limitations
```

Not stored in git: `data/` (raw dataset, re-download with the command above), `checkpoints/` (training weights), `venv/`, and `model.onnx` (available on the Releases page).

## Real-world evaluation

The benchmark images are web-scraped, mostly on clean backgrounds. Phone photos taken in Indian homes will look different, so test-set scores will be higher than real-world accuracy.

To measure it, add your own photos to `india_photos/<class_name>/` (for example `india_photos/plastic/bottle.jpg`), then run `python src/evaluate.py checkpoints/best_model.pt`. Results for this folder are reported separately from the benchmark. A good target is 25-50 photos per class.

## Limitations

- **Real-world accuracy is untested.** No Indian household photos have been scored yet.
- **One item per photo.** The model classifies the whole image and cannot find several items in a pile.
- **Plastic is a single class.** It cannot tell PET bottles from hard plastics or thin wrappers, and wrappers and pouches usually have no scrap value.
- **Indian packaging is under-represented.** Milk pouches, biscuit and chip wrappers, and local brands are not in the training data.
- **Metal is a single class.** It cannot tell aluminium, steel or copper apart, which have very different scrap prices.
- **Confidence is not a probability.** The model is under-confident, so use the `unsure` flag instead of showing raw percentages.
- **Material only, not price.** The classifier predicts a class. Prices come from a separate rate table.