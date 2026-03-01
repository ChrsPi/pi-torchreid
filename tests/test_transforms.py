"""Tests for data transforms (torchvision v2 backend)."""

import random

from PIL import Image
import pytest
import torch
from yacs.config import CfgNode as CN  # noqa: N817

from torchreid.data.transforms import (
    DeterministicBrightness,
    DeterministicContrast,
    DeterministicRotation,
    RandomPatch,
    ResolutionDegradation,
    build_transforms,
)


class TestRandomPatch:
    """Test RandomPatch transform (custom augmentation)."""

    def test_random_patch_basic(self):
        """Test basic random patch functionality."""
        transform = RandomPatch(prob_happen=1.0, min_sample_size=1)
        img = Image.new("RGB", (128, 256))
        for _ in range(transform.min_sample_size):
            result = transform(img)
            assert result.size == img.size

    def test_random_patch_probability(self):
        """Test probability parameter."""
        transform = RandomPatch(prob_happen=0.0, min_sample_size=1)
        img = Image.new("RGB", (128, 256))
        for _ in range(transform.min_sample_size):
            transform(img)
        random.seed(42)
        result = transform(img)
        assert result.size == img.size


class TestBuildTransforms:
    """Test build_transforms function (torchvision v2 backend)."""

    def test_build_transforms_returns_tuple_of_callables(self):
        """Test that build_transforms returns (train, test) callables."""
        transform_tr, transform_te = build_transforms(256, 128, transforms="random_flip")
        assert transform_tr is not None
        assert transform_te is not None
        assert callable(transform_tr)
        assert callable(transform_te)

    def test_build_transforms_string_input(self):
        """Test with string input."""
        transform_tr, transform_te = build_transforms(256, 128, transforms="random_flip")
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_list_input(self):
        """Test with list input."""
        transform_tr, transform_te = build_transforms(
            256, 128, transforms=["random_flip", "random_erase", "color_jitter"]
        )
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_none(self):
        """Test with None transforms (defaults to random_flip)."""
        transform_tr, transform_te = build_transforms(256, 128, transforms=None)
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_empty_list(self):
        """Test with empty list (no stochastic augmentations)."""
        transform_tr, transform_te = build_transforms(256, 128, transforms=[])
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_custom_norm(self):
        """Test with custom normalization."""
        transform_tr, transform_te = build_transforms(256, 128, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5])
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_all_transforms(self):
        """Test with all supported transform names."""
        transform_tr, transform_te = build_transforms(
            256,
            128,
            transforms=[
                "random_flip",
                "random_crop",
                "random_patch",
                "color_jitter",
                "random_erase",
            ],
        )
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_invalid_type(self):
        """Test error handling for invalid transform type."""
        with pytest.raises(ValueError, match="transforms must be a list"):
            build_transforms(256, 128, transforms=123)

    def test_build_transforms_train_vs_test_different_objects(self):
        """Test that train and test transforms are different objects."""
        transform_tr, transform_te = build_transforms(256, 128, transforms=["random_flip", "random_erase"])
        assert transform_tr is not transform_te

    def test_build_transforms_output_shape_and_dtype(self):
        """Test that transforms produce tensors of expected shape and dtype."""
        transform_tr, transform_te = build_transforms(256, 128, transforms="random_flip")
        img = Image.new("RGB", (100, 200))

        result_te = transform_te(img)
        assert hasattr(result_te, "shape")
        assert result_te.shape == (3, 256, 128)
        assert result_te.dtype == torch.float32

        random.seed(42)
        torch.manual_seed(42)
        result_tr = transform_tr(img)
        assert hasattr(result_tr, "shape")
        assert result_tr.shape == (3, 256, 128)
        assert result_tr.dtype == torch.float32


def _make_test_cfg(**degradation_overrides: dict) -> CN:
    """Build a minimal cfg with test degradations for testing."""
    from scripts.default_config import get_default_config

    cfg = get_default_config()
    cfg.data.height = 256
    cfg.data.width = 128
    cfg.data.transforms = ["random_flip"]
    for name, params in degradation_overrides.items():
        sub = getattr(cfg.aug.test, name)
        for k, v in params.items():
            setattr(sub, k, v)
    return cfg


class TestTestDegradations:
    """Integration tests for evaluation-time degradation transforms."""

    def _apply_test_transform(self, cfg: CN) -> torch.Tensor:
        _, transform_te = build_transforms(256, 128, cfg=cfg)
        img = Image.new("RGB", (200, 400), color=(128, 100, 80))
        return transform_te(img)

    def test_no_degradation_default(self):
        """No degradations enabled produces standard output."""
        cfg = _make_test_cfg()
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_gaussian_noise(self):
        """Gaussian noise changes pixel values."""
        baseline_cfg = _make_test_cfg(gaussian_noise={"enabled": False})
        baseline_result = self._apply_test_transform(baseline_cfg)

        cfg = _make_test_cfg(gaussian_noise={"enabled": True, "std": 0.1})
        result = self._apply_test_transform(cfg)

        assert result.shape == baseline_result.shape
        assert result.dtype == baseline_result.dtype
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32
        diff = (result - baseline_result).abs()
        assert torch.count_nonzero(diff).item() > 0
        assert diff.mean().item() > 0

    def test_gaussian_blur(self):
        """Gaussian blur produces valid output."""
        cfg = _make_test_cfg(gaussian_blur={"enabled": True, "sigma": 2.0})
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_gaussian_blur_custom_kernel(self):
        """Gaussian blur with explicit kernel size."""
        cfg = _make_test_cfg(gaussian_blur={"enabled": True, "sigma": 1.0, "kernel_size": 5})
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)

    def test_grayscale(self):
        """Grayscale produces 3-channel output."""
        cfg = _make_test_cfg(grayscale={"enabled": True})
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_rotation(self):
        """Rotation produces valid output."""
        cfg = _make_test_cfg(rotation={"enabled": True, "angle": 15.0})
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_resolution(self):
        """Resolution degradation produces valid output."""
        cfg = _make_test_cfg(resolution={"enabled": True, "scale": 0.25})
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_jpeg(self):
        """JPEG compression produces valid output."""
        cfg = _make_test_cfg(jpeg={"enabled": True, "quality": 10})
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_brightness(self):
        """Brightness adjustment produces valid output."""
        cfg = _make_test_cfg(brightness={"enabled": True, "factor": 0.5})
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_contrast(self):
        """Contrast adjustment produces valid output."""
        cfg = _make_test_cfg(contrast={"enabled": True, "factor": 0.3})
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_multiple_degradations_combined(self):
        """Multiple degradations can be combined."""
        cfg = _make_test_cfg(
            gaussian_blur={"enabled": True, "sigma": 1.5},
            brightness={"enabled": True, "factor": 0.8},
            jpeg={"enabled": True, "quality": 30},
            grayscale={"enabled": True},
        )
        result = self._apply_test_transform(cfg)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.float32

    def test_determinism_rotation(self):
        """Rotation is deterministic across calls."""
        cfg = _make_test_cfg(rotation={"enabled": True, "angle": 10.0})
        r1 = self._apply_test_transform(cfg)
        r2 = self._apply_test_transform(cfg)
        assert torch.equal(r1, r2)

    def test_determinism_brightness(self):
        """Brightness is deterministic across calls."""
        cfg = _make_test_cfg(brightness={"enabled": True, "factor": 0.6})
        r1 = self._apply_test_transform(cfg)
        r2 = self._apply_test_transform(cfg)
        assert torch.equal(r1, r2)

    def test_determinism_contrast(self):
        """Contrast is deterministic across calls."""
        cfg = _make_test_cfg(contrast={"enabled": True, "factor": 0.4})
        r1 = self._apply_test_transform(cfg)
        r2 = self._apply_test_transform(cfg)
        assert torch.equal(r1, r2)

    def test_determinism_resolution(self):
        """Resolution degradation is deterministic."""
        cfg = _make_test_cfg(resolution={"enabled": True, "scale": 0.5})
        r1 = self._apply_test_transform(cfg)
        r2 = self._apply_test_transform(cfg)
        assert torch.equal(r1, r2)


class TestCustomDegradationTransforms:
    """Unit tests for custom degradation transform classes."""

    def test_resolution_degradation_shape(self):
        img = torch.randint(0, 255, (3, 256, 128), dtype=torch.uint8)
        t = ResolutionDegradation(scale=0.5, target_size=(256, 128))
        result = t(img)
        assert result.shape == (3, 256, 128)
        assert result.dtype == torch.uint8

    def test_resolution_degradation_invalid_scale(self):
        with pytest.raises(ValueError, match="scale must be in"):
            ResolutionDegradation(scale=1.5, target_size=(256, 128))

    def test_resolution_degradation_very_small_scale(self):
        img = torch.randint(0, 255, (3, 256, 128), dtype=torch.uint8)
        t = ResolutionDegradation(scale=0.05, target_size=(256, 128))
        result = t(img)
        assert result.shape == (3, 256, 128)

    def test_deterministic_rotation(self):
        img = torch.randint(0, 255, (3, 64, 32), dtype=torch.uint8)
        t = DeterministicRotation(angle=45.0)
        r1 = t(img)
        r2 = t(img)
        assert r1.shape == img.shape
        assert torch.equal(r1, r2)

    def test_deterministic_brightness(self):
        img = torch.rand(3, 64, 32, dtype=torch.float32)
        t = DeterministicBrightness(factor=0.5)
        r1 = t(img)
        r2 = t(img)
        assert r1.shape == img.shape
        assert torch.equal(r1, r2)

    def test_deterministic_brightness_darkens(self):
        img = torch.ones(3, 64, 32, dtype=torch.float32) * 0.8
        t = DeterministicBrightness(factor=0.5)
        result = t(img)
        assert result.max() < img.max()

    def test_deterministic_contrast(self):
        img = torch.rand(3, 64, 32, dtype=torch.float32)
        t = DeterministicContrast(factor=0.5)
        r1 = t(img)
        r2 = t(img)
        assert r1.shape == img.shape
        assert torch.equal(r1, r2)

    def test_deterministic_contrast_reduces(self):
        img = torch.rand(3, 64, 32, dtype=torch.float32)
        t = DeterministicContrast(factor=0.5)
        result = t(img)
        # Reduced contrast means values closer to the mean
        assert result.std() < img.std()
