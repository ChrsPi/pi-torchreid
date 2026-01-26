# Torchreid Maintenance Plan

## Overview

Modernize the torchreid library by migrating to uv for dependency management, replacing legacy linting tools with ruff, fixing compatibility issues with modern Python/NumPy/PyTorch, and adding a proper test suite.

**Status:** Planning

---

## Phase 1: Migrate to uv and Ruff

### 1.1 Create pyproject.toml

Replace setup.py + requirements.txt with pyproject.toml:

```toml
[project]
name = "torchreid"
version = "1.4.0"
description = "A library for deep learning person re-ID in PyTorch"
readme = "README.rst"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Kaiyang Zhou" }]
keywords = ["Person Re-Identification", "Deep Learning", "Computer Vision"]

dependencies = [
    "numpy>=1.21,<2.0",
    "cython>=0.29",
    "h5py>=3.0",
    "pillow>=9.0",
    "scipy>=1.7",
    "opencv-python>=4.5",
    "matplotlib>=3.5",
    "tensorboard>=2.10",
    "yacs>=0.1.8",
    "gdown>=4.0",
    "imageio>=2.9",
    "chardet>=4.0",
    "torch>=2.0",
    "torchvision>=0.15",
]

[project.optional-dependencies]
dev = ["ruff", "pytest", "pytest-cov"]
export = ["onnx", "onnx-simplifier", "openvino-dev"]

[project.urls]
Homepage = "https://github.com/KaiyangZhou/deep-person-reid"
Documentation = "https://kaiyangzhou.github.io/deep-person-reid/"

[build-system]
requires = ["setuptools>=61", "wheel", "cython>=0.29", "numpy>=1.21"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["torchreid*"]

[tool.ruff]
line-length = 79
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]  # line length handled separately

[tool.ruff.lint.isort]
known-first-party = ["torchreid"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

### 1.2 Simplify setup.py (keep for Cython extension)

Reduce to just the Cython extension build:

```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        'torchreid.metrics.rank_cylib.rank_cy',
        ['torchreid/metrics/rank_cylib/rank_cy.pyx'],
        include_dirs=[np.get_include()],
    )
]

setup(ext_modules=cythonize(ext_modules))
```

### 1.3 Remove old linting files and dependencies

- Delete `requirements.txt` (merged into pyproject.toml)
- Delete `.flake8`
- Delete `.isort.cfg`
- Delete `.style.yapf`
- Delete `linter.sh`
- Remove `six` and `future` from dependencies (Python 2 compat)
- Remove `tb-nightly` (use `tensorboard` instead)

### 1.4 Create uv.lock

```bash
uv lock
uv sync --all-extras
```

### 1.5 Update CLAUDE.md and README

Update installation and linting instructions:

```bash
# Installation with uv (recommended)
uv sync
uv run python -c "import torchreid"

# Or with pip
pip install -e .

# Linting (replaces linter.sh)
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest
```

**Files modified/deleted:**
- Create `pyproject.toml`
- Simplify `setup.py`
- Delete `requirements.txt`
- Delete `.flake8`, `.isort.cfg`, `.style.yapf`, `linter.sh`
- Update `README.rst`
- Update `CLAUDE.md`

---

## Phase 2: Fix NumPy Compatibility

Replace deprecated type aliases (removed in NumPy 1.24):

| Old | New |
|-----|-----|
| `np.int` | `int` |
| `np.float` | `float` |
| `np.bool` | `bool` |

**Files to fix:**
- `torchreid/metrics/rank.py`
- `torchreid/utils/rerank.py`
- `torchreid/utils/GPU-Re-Ranking/utils.py`
- `torchreid/data/datasets/dataset.py`
- `projects/attribute_recognition/datasets/pa100k.py`

---

## Phase 3: Fix PyTorch Compatibility

Replace `torch.utils.model_zoo.load_url()` with `torch.hub.load_state_dict_from_url()` (removed in PyTorch 2.2):

**Files to fix:**
- `torchreid/models/resnet.py`
- `torchreid/models/mobilenetv2.py`
- `torchreid/models/densenet.py`
- `torchreid/models/squeezenet.py`
- `torchreid/models/resnet_ibn_a.py`
- `torchreid/models/resnet_ibn_b.py`
- `torchreid/models/mlfn.py`
- `torchreid/models/senet.py`
- `torchreid/models/shufflenet.py`
- `torchreid/models/shufflenetv2.py`
- `torchreid/models/inceptionv4.py`
- `torchreid/models/inceptionresnetv2.py`
- `torchreid/models/xception.py`
- `torchreid/models/nasnet.py`

---

## Phase 4: Cleanup Python 2 Compatibility

Remove from ~30 files:

```python
from __future__ import absolute_import, division, print_function
```

**Directories to process:**
- `torchreid/`
- `scripts/`
- `tools/`
- `projects/`

---

## Phase 5: Add Test Suite

### 5.1 Create test directory structure

```
tests/
├── __init__.py
├── conftest.py              # pytest fixtures
├── test_models.py           # model building tests
├── test_data.py             # data loading tests
├── test_losses.py           # loss function tests
├── test_metrics.py          # evaluation metrics tests
├── test_transforms.py       # data augmentation tests
├── test_optim.py            # optimizer/scheduler tests
└── test_utils.py            # utility function tests
```

### 5.2 Test categories

**Unit tests (no GPU/data required):**
- Model instantiation and forward pass shape checks
- Loss function computation
- Transform correctness
- Optimizer/scheduler creation
- Utility functions (feature extraction, reranking math)

**Integration tests (require small synthetic data):**
- DataManager with mock dataset
- Engine training loop (1-2 epochs on tiny data)
- Cython vs Python ranking equivalence

### 5.3 Key test files to create

**tests/conftest.py** - Shared fixtures:
```python
import pytest
import torch

@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing."""
    return {
        'img': torch.randn(4, 3, 256, 128),
        'pid': torch.tensor([0, 0, 1, 1]),
        'camid': torch.tensor([0, 1, 0, 1]),
    }

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**tests/test_models.py** - Model tests:
```python
import pytest
import torchreid

MODEL_NAMES = ['resnet50', 'osnet_x1_0', 'mobilenetv2_x1_0']

@pytest.mark.parametrize('name', MODEL_NAMES)
def test_model_build(name):
    model = torchreid.models.build_model(name=name, num_classes=10, loss='softmax')
    assert model is not None

@pytest.mark.parametrize('name', MODEL_NAMES)
def test_model_forward(name, dummy_batch, device):
    model = torchreid.models.build_model(name=name, num_classes=10, loss='softmax')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(dummy_batch['img'].to(device))
    assert out.shape[0] == 4
```

**tests/test_losses.py** - Loss tests:
```python
import torch
from torchreid.losses import CrossEntropyLoss, TripletLoss

def test_cross_entropy_loss():
    loss_fn = CrossEntropyLoss(num_classes=10)
    inputs = torch.randn(4, 10)
    targets = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(inputs, targets)
    assert loss.item() > 0

def test_triplet_loss():
    loss_fn = TripletLoss(margin=0.3)
    features = torch.randn(4, 512)
    targets = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(features, targets)
    assert loss.item() >= 0
```

**tests/test_metrics.py** - Metrics tests:
```python
import numpy as np
from torchreid.metrics.rank import evaluate_rank

def test_evaluate_rank():
    # Simple synthetic test
    distmat = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32)
    q_pids = np.array([0, 1, 2])
    g_pids = np.array([0, 1, 2])
    q_camids = np.array([0, 0, 0])
    g_camids = np.array([1, 1, 1])

    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    assert cmc[0] == 1.0  # Rank-1 should be perfect for diagonal
    assert mAP == 1.0
```

### 5.4 Move existing test

- Move `torchreid/metrics/rank_cylib/test_cython.py` to `tests/test_cython_rank.py`

---

## Verification

After each phase:

1. **Phase 1 (uv + ruff migration):**
   ```bash
   uv sync --all-extras
   uv run python -c "import torchreid; print(torchreid.__version__)"
   uv run ruff check .
   uv run python torchreid/metrics/rank_cylib/test_cython.py
   ```

2. **Phase 2-4 (compatibility fixes):**
   ```bash
   uv run ruff check .
   uv run python scripts/main.py --help
   ```

3. **Phase 5 (tests):**
   ```bash
   uv run pytest
   uv run pytest --cov=torchreid
   ```

4. **Full integration test (if datasets available):**
   ```bash
   uv run python scripts/main.py \
     --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
     --root $PATH_TO_DATA \
     test.evaluate True
   ```

---

## Background: Issues Found

### Critical Breaking Issues

| Issue | Status | Impact |
|-------|--------|--------|
| NumPy type aliases (`np.int`, `np.float`, `np.bool`) | Removed in NumPy 1.24+ | Evaluation metrics broken |
| `torch.utils.model_zoo` | Removed in PyTorch 2.2+ | Can't load pretrained weights |
| `distutils` | Removed in Python 3.12 | Can't build Cython extensions |

### Other Issues

- Python version: Currently requires 3.7 (EOL June 2023)
- Unpinned dependencies in requirements.txt
- Python 2 compatibility cruft in 30+ files
- Old `.cuda()/.cpu()` device handling pattern
- Legacy linting tools (flake8, yapf, isort) instead of modern ruff
- No test suite beyond one Cython benchmark script
