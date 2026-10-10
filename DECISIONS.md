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
- **Decision**: Created `./india_photos/` with 10 empty subdirectories corresponding to the 10 target classes (`battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`). The evaluation and data pipeline will check dynamically if images are present and seamlessly evaluate on them when available.
