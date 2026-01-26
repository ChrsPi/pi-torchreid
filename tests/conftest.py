"""Pytest fixtures and shared utilities for torchreid tests."""

import pytest
import torch
import numpy as np
import random
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def device():
    """Get available device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing."""
    return {
        "img": torch.randn(4, 3, 256, 128),
        "pid": torch.tensor([0, 0, 1, 1]),
        "camid": torch.tensor([0, 1, 0, 1]),
    }


@pytest.fixture
def dummy_model():
    """Create a simple test model."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10),
    )


@pytest.fixture
def tmp_data_dir():
    """Create a temporary directory for test data."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir)
