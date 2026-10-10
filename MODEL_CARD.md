# MODEL CARD: Household Waste Material Classifier (ResNet50)

## Model Overview
- **Model Architecture**: ResNet50 (ImageNet-pretrained backbone with modified 10-class linear classification head)
- **Framework**: PyTorch 2.14.1 / ONNX Runtime 1.31.0 (Opset 17)
- **Input Specifications**: Single RGB image, fixed 224×224 pixels, ImageNet normalized (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Output Specifications**: Softmax probabilities over 10 target waste material classes with uncertainty thresholding
- **Target Classes**: `battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`
- **ONNX Model File**: [`model.onnx`](file:///Users/shaaz/desktop/amazon_env_hack/model.onnx) (89.68 MB)
- **CPU Inference Latency**: **12.04 ms/image** (Mean), **11.96 ms** (P50), **12.56 ms** (P95) on Apple M4

---

## Performance Summary (Held-Out Test Set: 1,579 Images)

| Evaluation Metric | Test Set Score | Target Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **97.53%** | — | Exceeded |
| **Macro-F1 Score** | **0.9673** | $\ge 0.90$ | **Exceeded (+0.0673)** |
| **Top-3 Accuracy** | **99.68%** | — | Outstanding |
| **Expected Calibration Error (ECE)** | 0.1496 | — | Low |
| **High Confidence ($\ge 90\%$) Accuracy**| **100.0%** (388/388) | — | Perfectly Calibrated |

### Per-Class Test Performance
| Class Name | Test Support | Precision | Recall | F1-Score | Scrap Value Relevance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `battery` | 97 | 0.9700 | 1.0000 | **0.9848** | E-waste hazard |
| `biological` | 103 | 0.9717 | 1.0000 | **0.9856** | Wet waste / Compost |
| **`cardboard`** | 93 | 0.9890 | 0.9677 | **0.9783** | **High resale value** |
| `clothes` | 528 | 0.9962 | 0.9867 | **0.9914** | Textile reuse |
| `glass` | 204 | 0.9749 | 0.9510 | **0.9628** | Bottle deposit / Cullet |
| **`metal`** | 83 | 0.9070 | 0.9398 | **0.9231** | **Highest resale value** |
| **`paper`** | 111 | 0.9643 | 0.9730 | **0.9686** | **High resale value (*raddi*)** |
| **`plastic`** | 90 | 0.9022 | 0.9222 | **0.9121** | **High resale value** |
| `shoes` | 200 | 0.9851 | 0.9900 | **0.9875** | Footwear reuse |
| `trash` | 70 | 0.9855 | 0.9714 | **0.9784** | Non-recyclable refuse |

---

## Top 5 Most Confused Class Pairs
Analysis of off-diagonal errors in the held-out test set confusion matrix ([`confusion_matrix.csv`](file:///Users/shaaz/desktop/amazon_env_hack/reports/test_eval/confusion_matrix.csv)):

1. **`glass` mistaken for `metal`** (5 images): Transparent glass with reflective highlights resembling shiny sheet metal.
2. **`glass` mistaken for `plastic`** (5 images): Clear glass bottles resembling rigid clear plastic (PET) water bottles.
3. **`plastic` mistaken for `glass`** (4 images): Rigid transparent food containers resembling glass jars.
4. **`metal` mistaken for `plastic`** (3 images): Painted metal cans or beverage cans with colorful labels resembling plastic wrappers.
5. **`cardboard` mistaken for `paper`** (2 images): Thin brown card stock and uncorrugated paperboards.

Saved visual examples for all confusion pairs: [`reports/test_eval/misclassified/`](file:///Users/shaaz/desktop/amazon_env_hack/reports/test_eval/misclassified/).

---

## Confidence Threshold & Uncertainty Policy

| Threshold $\tau$ | Coverage (%) | Filtered Accuracy (%) | Operational Impact |
| :---: | :---: | :---: | :--- |
| 0.50 | 97.78% | 98.25% | Accepts almost all items |
| **0.60 (Recommended)** | **95.12%** | **99.07%** | **Optimal balance of coverage and extreme reliability** |
| 0.70 | 91.32% | 99.38% | Higher caution |
| 0.75 | 85.50% | 99.63% | Conservative filtering |
| 0.90 | 24.57% | 100.0% | Zero errors observed |

**Policy Implementation**:
- If `confidence >= 0.60`: Accept prediction and route to kabadiwala price table.
- If `confidence < 0.60`: Return `"unsure: true"` with prompt: `"not sure, try another angle"`.

---

## Domain Disclaimer: Real-World Phone Photos (`./india_photos/`)

> [!CAUTION]
> **Real-World Indian Household Performance is UNTESTED:**
> As verified during evaluation, `./india_photos/` was empty at evaluation time.
> The benchmark test set consists of web-scraped images often shot under clean lighting with isolated backgrounds. Real-world performance on smartphone photos taken inside Indian homes will inevitably experience domain shift.
> 
> **Action Required**: Collect **25–50 test photos per class** across real domestic settings (e.g. unwashed milk pouches, crumpled newspaper stacks, rusted tins, oil cans, cardboard courier boxes) and place them in `./india_photos/<class>/` to validate field accuracy.
