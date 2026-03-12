"""Augmentation pipeline: build_transforms and backends."""

from collections.abc import Callable, Sequence
from typing import Any

from yacs.config import CfgNode as CN  # noqa: N817

from torchreid.data.transforms.augmentations import (
    DeterministicBrightness,
    DeterministicContrast,
    DeterministicRotation,
    RandomPatch,
    ResolutionDegradation,
)
from torchreid.data.transforms.backends import get_backend
from torchreid.data.transforms.names import canonicalize_transform_list


def _build_effective_config(
    height: int,
    width: int,
    transforms: str | Sequence[str] | None,
    norm_mean: Sequence[float] | None,
    norm_std: Sequence[float] | None,
    cfg: Any | None,
) -> Any:
    """Build a config object the backend can read. Merge cfg with explicit args."""
    canonical_transforms = canonicalize_transform_list(transforms)

    if cfg is not None and hasattr(cfg, "data"):
        # Use existing cfg; ensure data has required fields from args
        if not getattr(cfg.data, "height", None):
            cfg.data.height = height
        if not getattr(cfg.data, "width", None):
            cfg.data.width = width
        if not hasattr(cfg.data, "transforms") or cfg.data.transforms is None:
            cfg.data.transforms = canonical_transforms
        existing_norm_mean = getattr(cfg.data, "norm_mean", None)
        if norm_mean is not None and not existing_norm_mean:
            cfg.data.norm_mean = list(norm_mean)
        existing_norm_std = getattr(cfg.data, "norm_std", None)
        if norm_std is not None and not existing_norm_std:
            cfg.data.norm_std = list(norm_std)
        # Ensure aug exists (backend reads from it)
        if not hasattr(cfg, "aug"):
            cfg.aug = CN()
            cfg.aug.backend = "torchvision_v2"
            cfg.aug.seed = None
            cfg.aug.disable_stochastic = False
            cfg.aug.normalize = CN()
            cfg.aug.normalize.mean = list(getattr(cfg.data, "norm_mean", [0.485, 0.456, 0.406]))
            cfg.aug.normalize.std = list(getattr(cfg.data, "norm_std", [0.229, 0.224, 0.225]))
            cfg.aug.train = CN()
            cfg.aug.test = CN()
            cfg.aug.test.center_crop = False
        return cfg

    # Build minimal config from args only
    effective = CN()
    effective.data = CN()
    effective.data.height = height
    effective.data.width = width
    effective.data.transforms = canonical_transforms
    effective.data.norm_mean = list(norm_mean) if norm_mean is not None else [0.485, 0.456, 0.406]
    effective.data.norm_std = list(norm_std) if norm_std is not None else [0.229, 0.224, 0.225]
    effective.aug = CN()
    effective.aug.backend = "torchvision_v2"
    effective.aug.seed = None
    effective.aug.disable_stochastic = False
    effective.aug.normalize = CN()
    effective.aug.normalize.mean = effective.data.norm_mean
    effective.aug.normalize.std = effective.data.norm_std
    effective.aug.train = CN()
    effective.aug.train.random_crop = CN()
    effective.aug.train.random_crop.enabled = "random_crop" in effective.data.transforms
    effective.aug.train.random_crop.scale_factor = 1.125
    effective.aug.train.random_erase = CN()
    effective.aug.train.random_erase.enabled = "random_erase" in effective.data.transforms
    effective.aug.train.random_erase.p = 0.5
    effective.aug.train.random_erase.scale = [0.02, 0.4]
    effective.aug.train.random_erase.ratio = [0.3, 3.3]
    effective.aug.train.random_patch = CN()
    effective.aug.train.random_patch.enabled = "random_patch" in effective.data.transforms
    effective.aug.train.random_patch.prob_happen = 0.5
    effective.aug.train.rand_augment = CN()
    effective.aug.train.rand_augment.enabled = "rand_augment" in effective.data.transforms
    effective.aug.train.rand_augment.num_ops = 2
    effective.aug.train.rand_augment.magnitude = 9
    # Allow arbitrary keys for v2 passthrough transform params
    effective.aug.train.set_new_allowed(True)
    effective.aug.test = CN()
    effective.aug.test.center_crop = False
    effective.aug.test.gaussian_noise = CN({"enabled": False, "std": 0.1})
    effective.aug.test.gaussian_blur = CN({"enabled": False, "sigma": 2.0, "kernel_size": 0})
    effective.aug.test.grayscale = CN({"enabled": False})
    effective.aug.test.rotation = CN({"enabled": False, "angle": 10.0})
    effective.aug.test.resolution = CN({"enabled": False, "scale": 0.5})
    effective.aug.test.jpeg = CN({"enabled": False, "quality": 50})
    effective.aug.test.brightness = CN({"enabled": False, "factor": 0.7})
    effective.aug.test.contrast = CN({"enabled": False, "factor": 0.5})
    return effective


def build_transforms(
    height: int,
    width: int,
    transforms: str | Sequence[str] | None = None,
    norm_mean: Sequence[float] | None = None,
    norm_std: Sequence[float] | None = None,
    cfg: Any | None = None,
    **kwargs: Any,
) -> tuple[Callable, Callable]:
    """Build train and test transform pipelines.

    Args:
        height: Target image height.
        width: Target image width.
        transforms: Transform names for training. Use v2 class names (PascalCase,
            e.g. 'RandomHorizontalFlip') or shortcut tokens (random_crop, random_patch,
            rand_augment, random_erase). Legacy aliases 'random_flip' and
            'color_jitter' are also accepted. Default None (no augmentation).
        norm_mean: Normalization mean (default ImageNet).
        norm_std: Normalization std (default ImageNet).
        cfg: Optional full config with aug.* and data.*. If provided, backend and
            detailed params are read from cfg.aug; otherwise built from the args above.
        **kwargs: Ignored (for compatibility).

    Returns:
        Tuple of (train_transform, test_transform), both callables PIL/tensor -> tensor.
    """
    if norm_std is None:
        existing_norm_std = getattr(getattr(cfg, "data", None), "norm_std", None) if cfg is not None else None
        if not existing_norm_std:
            norm_std = [0.229, 0.224, 0.225]
    if norm_mean is None:
        existing_norm_mean = getattr(getattr(cfg, "data", None), "norm_mean", None) if cfg is not None else None
        if not existing_norm_mean:
            norm_mean = [0.485, 0.456, 0.406]
    if transforms is not None and not isinstance(transforms, (str, list, tuple)):
        raise ValueError(f"transforms must be a list of strings, got {type(transforms)}")

    effective_cfg = _build_effective_config(height, width, transforms, norm_mean, norm_std, cfg)
    backend_name = getattr(getattr(effective_cfg, "aug", None), "backend", "torchvision_v2")
    backend = get_backend(backend_name)
    return (
        backend.build_train_transforms(effective_cfg),
        backend.build_test_transforms(effective_cfg),
    )


__all__ = [
    "build_transforms",
    "DeterministicBrightness",
    "DeterministicContrast",
    "DeterministicRotation",
    "RandomPatch",
    "ResolutionDegradation",
    "get_backend",
]
