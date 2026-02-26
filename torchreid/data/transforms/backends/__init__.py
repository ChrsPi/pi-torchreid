"""Augmentation backends registry."""

from torchreid.data.transforms.backends.base import AugmentationBackend
from torchreid.data.transforms.backends.torchvision_v2 import TorchvisionV2Backend

BACKENDS: dict[str, type[AugmentationBackend]] = {
    "torchvision_v2": TorchvisionV2Backend,
}


def get_backend(name: str) -> AugmentationBackend:
    """Return an augmentation backend by name."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown augmentation backend: {name}. Available: {list(BACKENDS)}")
    return BACKENDS[name]()
