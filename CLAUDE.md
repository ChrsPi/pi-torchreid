# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Torchreid is a PyTorch library for deep learning person re-identification (re-ID). Person re-ID is the task of identifying people across multiple non-overlapping camera views. The library is based on the ICCV'19 paper "Omni-Scale Feature Learning for Person Re-Identification."

**Note: This library is no longer actively maintained.**

## Modernization Guidelines

This codebase is undergoing modernization. When making changes:

- **No backward compatibility required**: Feel free to remove old implementations and replace them with better designs. We don't need to maintain legacy APIs or support old config formats.
- **Replace rather than wrap**: If a better approach exists (e.g., torchvision.transforms.v2 vs custom augmentations), replace the old code entirely rather than adding compatibility layers.
- **Update configs and tests**: When changing APIs, update all affected config files and tests to use the new approach.

## Common Commands

### Installation

**Using uv (Recommended):**
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Install extras (e.g. for testing)
uv sync --extra dev

# Verify installation
uv run python -c "import torchreid; print(torchreid.__version__)"
```

### Docker
```bash
make build-image  # Build Docker image
make run          # Run container with GPU support
```

### Linting
```bash
# Check for linting issues
uv run ruff check .

# Auto-fix linting issues (where possible)
uv run ruff check . --fix

# Format code
uv run ruff format .
```

### Training
```bash
# Train OSNet on Market1501
uv run python scripts/main.py \
  --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
  --transforms RandomHorizontalFlip random_erase \
  --root $PATH_TO_DATA

# Cross-domain: train on DukeMTMC, test on Market1501
uv run python scripts/main.py \
  --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml \
  -s dukemtmcreid -t market1501 \
  --transforms RandomHorizontalFlip ColorJitter \
  --root $PATH_TO_DATA
```

### Evaluation
```bash
uv run python scripts/main.py \
  --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
  --root $PATH_TO_DATA \
  model.load_weights log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250 \
  test.evaluate True
```

### Testing

**Run the test suite:**
```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=torchreid --cov-report=html

# Run specific test file
uv run pytest tests/test_models.py

# Run with verbose output
uv run pytest -v

# Run only fast tests (exclude slow benchmarks)
uv run pytest -m "not slow"
```

**Test suite structure:**
- `tests/test_models.py` - Model building and forward pass tests (50+ models)
- `tests/test_losses.py` - Loss function tests (CrossEntropyLoss, TripletLoss)
- `tests/test_metrics.py` - Evaluation metrics tests (evaluate_rank, accuracy, distance)
- `tests/test_transforms.py` - Data augmentation transform tests
- `tests/test_optim.py` - Optimizer and scheduler tests
- `tests/test_utils.py` - Utility function tests (rerank, model complexity, etc.)
- `tests/test_data.py` - Data loading and dataset tests
- `tests/test_cython_rank.py` - Cython ranking speed and correctness tests
- `tests/integration/` - Integration tests (Cython equivalence)

**Testing Cython Build (legacy):**
```bash
uv run python torchreid/metrics/rank_cylib/test_cython.py
# Note: This test has been moved to tests/test_cython_rank.py
```

### Multi-Split Results Aggregation
```bash
uv run python tools/parse_test_res.py log/eval_viper
```

### CI/CD

The repository uses GitHub Actions for automated testing and linting (see `.github/workflows/ci.yml`). All code changes are automatically validated on push/PR. The repository also uses Greptile for AI-assisted code reviews - PRs are automatically reviewed, or trigger manually with `@greptileai`.

## Architecture

### Core Package Structure (`torchreid/`)

- **`data/`** - Data loading and augmentation
  - `datamanager.py`: `ImageDataManager` and `VideoDataManager` classes
  - `sampler.py`: Custom samplers (`RandomIdentitySampler`, `RandomDomainSampler`, `RandomDatasetSampler`)
  - `data/transforms/__init__.py`: Augmentations (`RandomHorizontalFlip`, `random_erase`, `ColorJitter`, etc.)
  - `datasets/image/`: 13+ image re-ID datasets (market1501, cuhk03, dukemtmcreid, msmt17, etc.)
  - `datasets/video/`: 4 video re-ID datasets (mars, ilids, prid2011, dukemtmc-videoreid)

- **`models/`** - 50+ model architectures
  - `osnet.py`, `osnet_ain.py`: OSNet variants (state-of-the-art for re-ID)
  - `resnet.py`, `senet.py`, `densenet.py`: ImageNet backbones
  - `mobilenetv2.py`, `shufflenetv2.py`: Lightweight models
  - `pcb.py`, `hacnn.py`, `mlfn.py`: Re-ID specific architectures

- **`engine/`** - Training/evaluation logic
  - `engine.py`: Base `Engine` class with checkpoint saving, TensorBoard logging
  - `image/softmax.py`, `image/triplet.py`: Image-based training engines
  - `video/softmax.py`, `video/triplet.py`: Video-based training engines

- **`losses/`** - Loss functions
  - `cross_entropy_loss.py`: Softmax with optional label smoothing
  - `hard_mine_triplet_loss.py`: Hard-mining triplet loss

- **`metrics/`** - Evaluation (CMC, mAP)
  - `rank_cylib/`: Cython-optimized ranking for faster evaluation

- **`utils/`** - Utilities
  - `feature_extractor.py`: Simple API for inference
  - `rerank.py`: Re-ranking for improved retrieval

### Entry Points

- **`scripts/main.py`**: Unified CLI for training/testing with YACS config
- **`scripts/default_config.py`**: Configuration schema with all hyperparameters
- **`configs/`**: Example YAML configuration files

### Configuration System

Uses YACS for hierarchical YAML configs. Override via command-line:
```bash
uv run python scripts/main.py --config-file config.yaml train.lr 0.001 train.max_epoch 100
```

Key config groups: `model`, `data`, `train`, `loss`, `test`, `sampler`

### Data Tuple Format

Dataset items are 4-tuples: `(impath, pid, camid, dsetid)` where `dsetid` identifies the source dataset for multi-dataset training.

### Model Factory Pattern

```python
import torchreid
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    loss='softmax',  # or 'triplet'
    pretrained=True
)
```

### Feature Extraction API

```python
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor(model_name='osnet_x1_0', model_path='checkpoint.pth')
features = extractor(image_list)
```

## Key Resources

- Documentation: https://kaiyangzhou.github.io/deep-person-reid/
- Model Zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- Pretrained weights: https://huggingface.co/kaiyangzhou/osnet
- Tech report: https://arxiv.org/abs/1910.10093
