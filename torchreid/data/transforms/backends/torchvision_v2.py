"""Torchvision transforms v2 augmentation backend."""

from collections.abc import Callable
import inspect
import math
import random
from typing import Any

import numpy as np
import torch
from torchvision.transforms import v2

from torchreid.data.transforms.augmentations import (
    DeterministicBrightness,
    DeterministicContrast,
    DeterministicRotation,
    RandomPatch,
    ResolutionDegradation,
)
from torchreid.utils import logger

# Shortcut tokens: names with custom build logic that can't be expressed
# as a simple v2.ClassName(**kwargs) passthrough.
SHORTCUT_TOKENS = frozenset(
    {
        "random_crop",  # replaces Resize with RandomResizedCrop + scale_factor math
        "random_patch",  # custom class, not a v2 transform
        "rand_augment",  # must go at uint8 stage (after ToImage, before ToDtype)
        "random_erase",  # must go after Normalize, fills with per-channel mean
    }
)


def _get_transform_names(cfg: Any) -> list[str]:
    """Return list of transform names from config."""
    names = getattr(cfg.data, "transforms", None)
    if names is None:
        names = []
    if isinstance(names, str):
        names = [names]
    return [str(x) for x in names]


def _get_img_size(cfg: Any) -> tuple[int, int]:
    """Return (height, width) from config."""
    height = getattr(cfg.data, "height", 256)
    width = getattr(cfg.data, "width", 128)
    return int(height), int(width)


def _get_norm(cfg: Any) -> tuple[list[float], list[float]]:
    """Return (mean, std) from config."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Base values may come from aug.normalize, but data.norm_* has final precedence.
    if hasattr(cfg, "aug") and hasattr(cfg.aug, "normalize"):
        if hasattr(cfg.aug.normalize, "mean"):
            mean = list(cfg.aug.normalize.mean)
        if hasattr(cfg.aug.normalize, "std"):
            std = list(cfg.aug.normalize.std)

    data_norm_mean = getattr(getattr(cfg, "data", None), "norm_mean", None)
    if data_norm_mean is not None:
        mean = list(data_norm_mean)

    data_norm_std = getattr(getattr(cfg, "data", None), "norm_std", None)
    if data_norm_std is not None:
        std = list(data_norm_std)

    return mean, std


def _is_enabled(cfg: Any, transform_name: str, default: bool = False) -> bool:
    """Check if a transform is enabled via cfg.aug.train or data.transforms list."""
    disable_stochastic = bool(getattr(getattr(cfg, "aug", None), "disable_stochastic", False))
    if disable_stochastic:
        return False

    names = _get_transform_names(cfg)
    if transform_name in names:
        return True

    if not hasattr(cfg, "aug") or not hasattr(cfg.aug, "train"):
        return default
    train = cfg.aug.train

    # Check aug.train.<name>.enabled
    if hasattr(train, transform_name):
        sub = getattr(train, transform_name)
        if hasattr(sub, "enabled"):
            return bool(sub.enabled)
    return default


def _get_train_param(cfg: Any, transform_name: str, param: str, default: Any) -> Any:
    """Get a train param from cfg.aug.train.<name>.<param>."""
    if not hasattr(cfg, "aug") or not hasattr(cfg.aug, "train"):
        return default
    train = cfg.aug.train
    if not hasattr(train, transform_name):
        return default
    sub = getattr(train, transform_name)
    return getattr(sub, param, default)


def _get_random_crop_scale_params(cfg: Any) -> tuple[float, float, float]:
    """Return validated RandomResizedCrop scale params."""
    scale_factor = float(_get_train_param(cfg, "random_crop", "scale_factor", 1.125))
    if not math.isfinite(scale_factor) or scale_factor <= 1.0:
        raise ValueError(
            "cfg.aug.train.random_crop.scale_factor must be > 1.0 "
            f"for RandomResizedCrop, got {scale_factor}"
        )

    scale_min = 1.0 / (scale_factor * scale_factor)
    scale_max = 1.0
    return scale_factor, scale_min, scale_max


def _is_test_degradation_enabled(cfg: Any, name: str) -> bool:
    """Check if a test degradation is enabled via cfg.aug.test.<name>.enabled."""
    aug_test = getattr(getattr(cfg, "aug", None), "test", None)
    if aug_test is None:
        return False
    sub = getattr(aug_test, name, None)
    if sub is None:
        return False
    return bool(getattr(sub, "enabled", False))


def _get_test_param(cfg: Any, name: str, param: str, default: Any) -> Any:
    """Get a test degradation param from cfg.aug.test.<name>.<param>."""
    aug_test = getattr(getattr(cfg, "aug", None), "test", None)
    if aug_test is None:
        return default
    sub = getattr(aug_test, name, None)
    if sub is None:
        return default
    return getattr(sub, param, default)


def _build_v2_passthrough(name: str, cfg: Any, size: tuple[int, int]) -> Any | None:
    """Instantiate a torchvision v2 transform by class name.

    Returns None if *name* is not found in ``torchvision.transforms.v2``.
    """
    cls = getattr(v2, name, None)
    if cls is None:
        return None
    kwargs: dict[str, Any] = {}
    if hasattr(cfg, "aug") and hasattr(cfg.aug, "train"):
        sub = getattr(cfg.aug.train, name, None)
        if sub is not None and hasattr(sub, "items"):
            kwargs = {k: val for k, val in sub.items() if k != "enabled"}
    # Auto-inject size if the transform expects it and user didn't provide it
    if "size" not in kwargs:
        sig = inspect.signature(cls)
        if "size" in sig.parameters:
            kwargs["size"] = size
    return cls(**kwargs)


def _validate_transform_names(names: list[str]) -> None:
    """Validate transform names, raising ValueError for unknown ones."""
    for name in names:
        if name in SHORTCUT_TOKENS:
            continue
        if getattr(v2, name, None) is not None:
            continue
        shortcuts = ", ".join(sorted(SHORTCUT_TOKENS))
        raise ValueError(
            f"Unknown transform: {name!r}. "
            f"Shortcut tokens: {shortcuts}. "
            f"You can also use any torchvision.transforms.v2 class name (PascalCase)."
        )


class TorchvisionV2Backend:
    """Augmentation backend using torchvision.transforms.v2."""

    def build_train_transforms(self, cfg: Any) -> Callable:
        """Build training augmentation pipeline (PIL/tensor in -> tensor out)."""
        height, width = _get_img_size(cfg)
        mean, std = _get_norm(cfg)
        size = (height, width)
        transforms_list: list[Any] = []

        # Validate all transform names
        names = _get_transform_names(cfg)
        _validate_transform_names(names)

        # Collect passthrough names (v2 class names, not shortcut tokens)
        disable_stochastic = bool(getattr(getattr(cfg, "aug", None), "disable_stochastic", False))
        passthrough_names = [] if disable_stochastic else [n for n in names if n not in SHORTCUT_TOKENS]

        # Optional seed for deterministic augmentation behavior.
        seed = getattr(cfg.aug, "seed", None) if hasattr(cfg, "aug") else None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            logger.info("+ set augmentation seed=%s", seed)

        # 1. Resize or RandomResizedCrop (random_crop)
        if _is_enabled(cfg, "random_crop", default=False):
            scale_factor, scale_min, scale_max = _get_random_crop_scale_params(cfg)
            # RandomResizedCrop with scale that mimics 1.125x resize then crop
            transforms_list.append(v2.RandomResizedCrop(size=size, scale=(scale_min, scale_max), antialias=True))
            logger.info(
                "+ random resized crop %s (scale ~1/%.2f to 1)",
                size,
                scale_factor * scale_factor,
            )
        else:
            transforms_list.append(v2.Resize(size=size, antialias=True))
            logger.info("+ resize to %sx%s", height, width)

        # 2. Random patch (custom, PIL-only)
        if _is_enabled(cfg, "random_patch", default=False):
            prob = _get_train_param(cfg, "random_patch", "prob_happen", 0.5)
            transforms_list.append(RandomPatch(prob_happen=prob))
            logger.info("+ random patch")

        # 3. v2 passthrough transforms (in data.transforms list order)
        for name in passthrough_names:
            t = _build_v2_passthrough(name, cfg, size)
            transforms_list.append(t)
            logger.info("+ %s (v2 passthrough)", name)

        # 4. PIL -> uint8 TVTensor
        transforms_list.append(v2.ToImage())

        # 5. RandAugment (optional, expects uint8 input)
        if _is_enabled(cfg, "rand_augment", default=False):
            num_ops = _get_train_param(cfg, "rand_augment", "num_ops", 2)
            magnitude = _get_train_param(cfg, "rand_augment", "magnitude", 9)
            transforms_list.append(v2.RandAugment(num_ops=num_ops, magnitude=magnitude))
            logger.info("+ RandAugment")

        # 6. uint8 -> float32 [0, 1]
        transforms_list.append(v2.ToDtype(torch.float32, scale=True))

        # 7. Normalize
        transforms_list.append(v2.Normalize(mean=mean, std=std))
        logger.info("+ normalize (mean=%s, std=%s)", mean, std)

        # 8. Random erasing (tensor-only, after normalize; fill with mean)
        if _is_enabled(cfg, "random_erase", default=False):
            p = _get_train_param(cfg, "random_erase", "p", 0.5)
            scale = _get_train_param(cfg, "random_erase", "scale", (0.02, 0.4))
            ratio = _get_train_param(cfg, "random_erase", "ratio", (0.3, 3.3))
            scale = tuple(scale) if isinstance(scale, (list, tuple)) and len(scale) == 2 else (0.02, 0.4)
            ratio = tuple(ratio) if isinstance(ratio, (list, tuple)) and len(ratio) == 2 else (0.3, 3.3)
            transforms_list.append(v2.RandomErasing(p=p, scale=scale, ratio=ratio, value=mean))
            logger.info("+ random erase (p=%s)", p)

        return v2.Compose(transforms_list)

    def build_test_transforms(self, cfg: Any) -> Callable:
        """Build test/validation augmentation pipeline (deterministic).

        Pipeline order:
          Resize -> [CenterCrop] -> ToImage (uint8)
          -> [JPEG] -> [Resolution] -> [Grayscale] -> [Rotation]
          -> ToDtype(float32) -> [GaussianBlur] -> [GaussianNoise]
          -> [Brightness] -> [Contrast] -> Normalize
        """
        height, width = _get_img_size(cfg)
        mean, std = _get_norm(cfg)
        size = (height, width)

        transforms_list: list[Any] = []

        # 1. Resize
        transforms_list.append(v2.Resize(size=size, antialias=True))

        # 2. Optional center crop
        aug_test = getattr(getattr(cfg, "aug", None), "test", None)
        if aug_test is not None and getattr(aug_test, "center_crop", False):
            transforms_list.append(v2.CenterCrop(size=size))

        # 3. PIL -> uint8 TVTensor
        transforms_list.append(v2.ToImage())

        # -- uint8 degradations --

        # JPEG compression
        if _is_test_degradation_enabled(cfg, "jpeg"):
            quality = int(_get_test_param(cfg, "jpeg", "quality", 50))
            transforms_list.append(v2.JPEG(quality=(quality, quality)))
            logger.info("+ test degradation: JPEG (quality=%d)", quality)

        # Resolution degradation
        if _is_test_degradation_enabled(cfg, "resolution"):
            scale = float(_get_test_param(cfg, "resolution", "scale", 0.5))
            transforms_list.append(ResolutionDegradation(scale=scale, target_size=size))
            logger.info("+ test degradation: resolution (scale=%.2f)", scale)

        # Grayscale
        if _is_test_degradation_enabled(cfg, "grayscale"):
            transforms_list.append(v2.Grayscale(num_output_channels=3))
            logger.info("+ test degradation: grayscale")

        # Rotation
        if _is_test_degradation_enabled(cfg, "rotation"):
            angle = float(_get_test_param(cfg, "rotation", "angle", 10.0))
            transforms_list.append(DeterministicRotation(angle=angle))
            logger.info("+ test degradation: rotation (angle=%.1f)", angle)

        # 4. uint8 -> float32 [0, 1]
        transforms_list.append(v2.ToDtype(torch.float32, scale=True))

        # -- float32 degradations --

        # Gaussian blur
        if _is_test_degradation_enabled(cfg, "gaussian_blur"):
            sigma = float(_get_test_param(cfg, "gaussian_blur", "sigma", 2.0))
            kernel_size = int(_get_test_param(cfg, "gaussian_blur", "kernel_size", 0))
            if kernel_size == 0:
                # Auto kernel size: must be odd, >= 6*sigma+1
                kernel_size = int(math.ceil(6 * sigma + 1))
                kernel_size |= 1  # bitwise OR 1 ensures odd
            transforms_list.append(v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
            logger.info("+ test degradation: gaussian_blur (sigma=%.1f, kernel=%d)", sigma, kernel_size)

        # Gaussian noise
        if _is_test_degradation_enabled(cfg, "gaussian_noise"):
            noise_std = float(_get_test_param(cfg, "gaussian_noise", "std", 0.1))
            transforms_list.append(v2.GaussianNoise(mean=0.0, sigma=noise_std))
            logger.info("+ test degradation: gaussian_noise (std=%.2f)", noise_std)

        # Brightness
        if _is_test_degradation_enabled(cfg, "brightness"):
            factor = float(_get_test_param(cfg, "brightness", "factor", 0.7))
            transforms_list.append(DeterministicBrightness(factor=factor))
            logger.info("+ test degradation: brightness (factor=%.2f)", factor)

        # Contrast
        if _is_test_degradation_enabled(cfg, "contrast"):
            factor = float(_get_test_param(cfg, "contrast", "factor", 0.5))
            transforms_list.append(DeterministicContrast(factor=factor))
            logger.info("+ test degradation: contrast (factor=%.2f)", factor)

        # 5. Normalize
        transforms_list.append(v2.Normalize(mean=mean, std=std))

        logger.info("Building test transforms: resize %sx%s, normalize", height, width)
        return v2.Compose(transforms_list)
