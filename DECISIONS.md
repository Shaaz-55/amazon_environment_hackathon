# Architectural & Engineering Decisions Log

## DECISION 001: Python Runtime and Environment Setup
- **Date**: 2026-10-10
- **Context**: The host machine runs macOS on Apple Silicon (M4) with Python 3.14.3. User specified ground rule: if PyTorch or key packages fail on 3.14, propose a project-local Python 3.12 venv.
- **Evaluation**: PyPI offers pre-built `macosx_14_0_arm64` wheels for Python 3.14 for PyTorch 2.14.1, Torchvision 0.29.1, Scikit-Learn 1.9.1, ONNX 1.23.2, ONNXRuntime 1.31.0, and ImageHash 4.3.2.
- **Decision**: Use the existing project-local virtual environment (`./venv`) with Python 3.14.3. No system modifications required.
- **Rejected**: Downgrading or installing an additional Python 3.12 interpreter, as native wheels exist and work without issues.

## DECISION 002: Version Control & Project Hygiene
- **Date**: 2026-10-10
- **Context**: Project directory was untracked inside a parent folder git repo.
- **Decision**: Initialized an independent git repository at `/Users/shaaz/desktop/amazon_env_hack`. Configured `.gitignore` to strictly exclude `data/`, `venv/`, checkpoints (`*.pt`, `*.pth`), ONNX artifacts, and system files (`.DS_Store`).

## DECISION 003: India Photos Staging Directory
- **Date**: 2026-10-10
- **Context**: User plans to add real phone photos later for evaluation and domain adaptation.
- **Decision**: Created `./india_photos/` with 10 empty subdirectories corresponding to the 10 target classes (`battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`). The evaluation and data pipeline checks dynamically if images are present and seamlessly evaluates on them when available.

## DECISION 004: V1 Class Mapping and Canonical Class Ordering
- **Date**: 2026-10-10
- **Context**: The enhanced dataset has 12 native classes. For domestic recycling/kabadiwala pricing, glass subtypes (brown, green, white) should be unified into a single class.
- **Decision**: Mapped `brown-glass`, `green-glass`, and `white-glass` into a single `glass` class, producing 10 target classes:
  `['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']`.
  Canonical 0-indexed mapping saved to `labels.txt` and `labels.json` to guarantee identical ordering across training, validation, testing, and ONNX export.

## DECISION 005: Filtering Near-Duplicates with Conflicting Labels
- **Date**: 2026-10-10
- **Context**: Perceptual hashing (pHash Hamming distance <= 4) revealed near-identical images in the enhanced dataset with discordant labels across classes.
- **Evaluation**: 12 near-duplicate clusters containing 63 total images had discordant labels (e.g. green-glass vs trash, metal vs plastic, cardboard vs trash). Training or testing on discordant duplicates introduces label noise and contradictory gradients.
- **Decision**: Completely excluded all 63 images from conflicting clusters across all splits. Detailed list preserved in `reports/splits_summary.json`. Also excluded 1 tiny image (`trash/trash622.jpg`, 51x400 px).

## DECISION 006: Atomic Cluster Splitting for Zero Data Leakage
- **Date**: 2026-10-10
- **Context**: Near-duplicate images ending up in both training and test sets artificially inflate evaluation metrics.
- **Decision**: Grouped all non-conflicting near-duplicate images into atomic clusters. Partitioning was performed at the cluster level (80% train, 10% val, 10% test) with fixed random seed 42, guaranteeing that no near-duplicate pairs or web scrape variants span across splits.

## DECISION 007: Capping Dominant Class (Clothes) in Training Split Only
- **Date**: 2026-10-10
- **Context**: `clothes` accounted for 33.3% of the dataset (4,229 images in train), severely biasing the model.
- **Decision**: Capped `clothes` at exactly 1,800 images in the training manifest using fixed random seed 42. Maintained natural validation (529 images) and test (528 images) distributions to measure real-world performance honestly.
- **Balanced Loss Weighting**: Implemented inverse frequency class weights in CrossEntropyLoss:
  - battery: 1.3097
  - biological: 1.2338
  - cardboard: 1.3658
  - clothes: 0.5676
  - glass: 0.6283
  - metal: 1.5339
  - paper: 1.1530
  - plastic: 1.4288
  - shoes: 0.6365
  - trash: 1.8178

## DECISION 008: RGBA & Palette Image Compositing
- **Date**: 2026-10-10
- **Context**: 113 images in the dataset are in RGBA or Palette ('P') mode. Simply stripping the alpha channel can convert transparent backgrounds to black, creating artificial dark borders/artifacts.
- **Decision**: All non-RGB images are composited onto a solid white background `(255, 255, 255)` using their alpha mask before conversion to RGB.

## DECISION 009: Two-Stage Training & Backbone Fine-Tuning Strategy
- **Date**: 2026-10-10
- **Context**: Directly fine-tuning an entire ResNet50 backbone on a relatively small target dataset risks destroying pretrained ImageNet feature extractors.
- **Decision**:
  - *Stage 1*: 5 epochs with backbone frozen, learning rate $1\times 10^{-3}$, AdamW, Cosine Annealing. Reached 94.44% val accuracy and 0.9211 val macro-F1.
  - *Stage 2*: Unfroze `layer3`, `layer4`, and `fc` with differential learning rates ($2\times 10^{-5}$, $5\times 10^{-5}$, $1\times 10^{-4}$). Early stopping with patience 4 triggered at epoch 12, preserving the best checkpoint from epoch 8 (97.35% val accuracy, 0.9603 val macro-F1).

## DECISION 010: Confidence Thresholding Policy for Domestic Resale
- **Date**: 2026-10-10
- **Context**: In real-world scrap buying (*kabadiwala* app), misclassifying valuable scrap (e.g. metal or paper) as non-recyclable trash or vice-versa destroys user trust.
- **Decision**:
  - Recommended threshold $\tau = 0.60$.
  - At $\tau = 0.60$, coverage is 95.12% and filtered accuracy is **99.07%**.
  - Any prediction with confidence $< 0.60$ is flagged with `unsure: true` and prompts: `"not sure, try another angle"`.
  - On the highest confidence subset ($\ge 0.90$), empirical accuracy is **100.0%**.

## DECISION 011: ONNX Export with Numerical Parity Guarantee
- **Date**: 2026-10-10
- **Context**: Model must run efficiently on CPU for lightweight backend deployment.
- **Decision**: Exported model with fixed shape `(1, 3, 224, 224)` and Opset 17. Tested on 50 unseen images against PyTorch: maximum absolute difference was **$3.74\times 10^{-5}$** ($< 1\times 10^{-4}$ SLA).
- **Latency**: Mean CPU latency on Apple M4 is **12.04 ms/image** (over 40× faster than the 500 ms ceiling), so dynamic quantization was not required.
