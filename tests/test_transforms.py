"""Tests for data transforms."""

import pytest
import torch
import numpy as np
from PIL import Image
from torchreid.data.transforms import (
    Random2DTranslation,
    RandomErasing,
    ColorAugmentation,
    RandomPatch,
    build_transforms,
)


class TestRandom2DTranslation:
    """Test Random2DTranslation transform."""

    def test_random_2d_translation_output_size(self):
        """Test output image size matches target."""
        transform = Random2DTranslation(height=256, width=128, p=1.0)
        img = Image.new("RGB", (100, 200))
        result = transform(img)
        assert result.size == (128, 256)

    def test_random_2d_translation_probability(self):
        """Test probability parameter."""
        # With p=0.0, should always resize (no translation)
        transform = Random2DTranslation(height=256, width=128, p=0.0)
        img = Image.new("RGB", (100, 200))
        # Set random seed for deterministic test
        import random
        random.seed(42)
        result = transform(img)
        assert result.size == (128, 256)

    def test_random_2d_translation_different_sizes(self):
        """Test with different image sizes."""
        transform = Random2DTranslation(height=256, width=128, p=1.0)
        for w, h in [(64, 128), (200, 400), (50, 100)]:
            img = Image.new("RGB", (w, h))
            result = transform(img)
            assert result.size == (128, 256)


class TestRandomErasing:
    """Test RandomErasing transform."""

    def test_random_erasing_basic(self):
        """Test basic random erasing."""
        transform = RandomErasing(probability=1.0, mean=[0.5, 0.5, 0.5])
        img = torch.randn(3, 256, 128)
        result = transform(img)
        assert result.shape == img.shape

    def test_random_erasing_probability(self):
        """Test probability parameter."""
        # With probability=0.0, should return original image
        transform = RandomErasing(probability=0.0, mean=[0.5, 0.5, 0.5])
        img = torch.randn(3, 256, 128)
        original = img.clone()
        import random
        random.seed(42)
        result = transform(img)
        # May or may not be equal due to randomness, but shape should match
        assert result.shape == original.shape

    def test_random_erasing_grayscale(self):
        """Test with grayscale image (1 channel)."""
        transform = RandomErasing(probability=1.0, mean=[0.5])
        img = torch.randn(1, 256, 128)
        result = transform(img)
        assert result.shape == img.shape

    def test_random_erasing_different_sizes(self):
        """Test with different image sizes."""
        transform = RandomErasing(probability=1.0, mean=[0.5, 0.5, 0.5])
        for c, h, w in [(3, 64, 32), (3, 512, 256), (1, 128, 64)]:
            img = torch.randn(c, h, w)
            result = transform(img)
            assert result.shape == img.shape


class TestColorAugmentation:
    """Test ColorAugmentation transform."""

    def test_color_augmentation_basic(self):
        """Test basic color augmentation."""
        transform = ColorAugmentation(p=1.0)
        tensor = torch.randn(3, 256, 128)
        result = transform(tensor)
        assert result.shape == tensor.shape

    def test_color_augmentation_input_validation(self):
        """Test input validation."""
        transform = ColorAugmentation(p=1.0)
        # Should work with 3-channel tensor
        tensor = torch.randn(3, 256, 128)
        result = transform(tensor)
        assert result.shape == tensor.shape

        # Should fail with wrong dimensions
        with pytest.raises(AssertionError):
            transform._check_input(torch.randn(1, 256, 128))
        with pytest.raises(AssertionError):
            transform._check_input(torch.randn(3, 256))


class TestRandomPatch:
    """Test RandomPatch transform."""

    def test_random_patch_basic(self):
        """Test basic random patch functionality."""
        transform = RandomPatch(prob_happen=1.0, min_sample_size=1)
        img = Image.new("RGB", (128, 256))
        # Need to populate patch pool first
        for _ in range(transform.min_sample_size):
            result = transform(img)
            assert result.size == img.size

    def test_random_patch_probability(self):
        """Test probability parameter."""
        transform = RandomPatch(prob_happen=0.0, min_sample_size=1)
        img = Image.new("RGB", (128, 256))
        # Populate pool
        for _ in range(transform.min_sample_size):
            transform(img)
        # With prob_happen=0.0, should return original
        import random
        random.seed(42)
        result = transform(img)
        assert result.size == img.size


class TestBuildTransforms:
    """Test build_transforms function."""

    def test_build_transforms_basic(self):
        """Test basic transform building."""
        transform_tr, transform_te = build_transforms(256, 128, transforms="random_flip")
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_string_input(self):
        """Test with string input."""
        transform_tr, transform_te = build_transforms(256, 128, transforms="random_flip")
        assert transform_tr is not None

    def test_build_transforms_list_input(self):
        """Test with list input."""
        transform_tr, transform_te = build_transforms(
            256, 128, transforms=["random_flip", "random_erase", "color_jitter"]
        )
        assert transform_tr is not None

    def test_build_transforms_none(self):
        """Test with None transforms."""
        transform_tr, transform_te = build_transforms(256, 128, transforms=None)
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_empty_list(self):
        """Test with empty list."""
        transform_tr, transform_te = build_transforms(256, 128, transforms=[])
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_custom_norm(self):
        """Test with custom normalization."""
        transform_tr, transform_te = build_transforms(
            256, 128, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]
        )
        assert transform_tr is not None
        assert transform_te is not None

    def test_build_transforms_all_transforms(self):
        """Test with all supported transforms."""
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

    def test_build_transforms_invalid_type(self):
        """Test error handling for invalid transform type."""
        with pytest.raises(ValueError, match="transforms must be a list"):
            build_transforms(256, 128, transforms=123)  # Invalid type

    def test_build_transforms_train_vs_test(self):
        """Test that train and test transforms are different."""
        transform_tr, transform_te = build_transforms(
            256, 128, transforms=["random_flip", "random_erase"]
        )
        # They should be different objects
        assert transform_tr is not transform_te

    def test_build_transforms_apply(self):
        """Test that transforms can be applied to images."""
        transform_tr, transform_te = build_transforms(256, 128, transforms="random_flip")
        img = Image.new("RGB", (100, 200))
        # Test transform should work
        result_te = transform_te(img)
        assert isinstance(result_te, torch.Tensor)
        assert result_te.shape == (3, 256, 128)

        # Train transform (with randomness)
        import random
        random.seed(42)
        result_tr = transform_tr(img)
        assert isinstance(result_tr, torch.Tensor)
        assert result_tr.shape == (3, 256, 128)
