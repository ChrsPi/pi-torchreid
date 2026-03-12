"""Regression tests for torchvision v2 backend config handling."""

from torchvision.transforms import v2
from yacs.config import CfgNode as CN  # noqa: N817

from scripts.default_config import get_default_config
from torchreid.data.transforms import RandomPatch, build_transforms


def test_empty_transform_list_is_noop_not_random_flip():
    """An explicit empty transform list should not fallback to random_flip."""
    transform_tr, _ = build_transforms(256, 128, transforms=[])
    assert not any(isinstance(t, v2.RandomHorizontalFlip) for t in transform_tr.transforms)


def test_cfg_empty_transform_list_beats_transforms_arg():
    """cfg.data.transforms=[] must not be overwritten by transforms arg."""
    cfg = get_default_config()
    cfg.data.transforms = []

    transform_tr, _ = build_transforms(256, 128, transforms=["RandomHorizontalFlip"], cfg=cfg)

    assert not any(isinstance(t, v2.RandomHorizontalFlip) for t in transform_tr.transforms)


def test_cfg_data_norm_values_are_not_overwritten_by_explicit_norm_args():
    """Existing cfg.data.norm_* must take precedence over explicit norm args."""
    cfg = get_default_config()
    cfg.aug.normalize.mean = [0.9, 0.8, 0.7]
    cfg.aug.normalize.std = [0.3, 0.2, 0.1]

    transform_tr, transform_te = build_transforms(
        256,
        128,
        norm_mean=[0.1, 0.2, 0.3],
        norm_std=[0.7, 0.8, 0.9],
        cfg=cfg,
    )
    norm_tr = next(t for t in transform_tr.transforms if isinstance(t, v2.Normalize))
    norm_te = next(t for t in transform_te.transforms if isinstance(t, v2.Normalize))

    assert norm_tr.mean == [0.485, 0.456, 0.406]
    assert norm_tr.std == [0.229, 0.224, 0.225]
    assert norm_te.mean == [0.485, 0.456, 0.406]
    assert norm_te.std == [0.229, 0.224, 0.225]


def test_explicit_norm_used_when_cfg_data_norm_values_are_missing():
    """Explicit norm args should populate cfg.data.norm_* when values are missing."""
    cfg = get_default_config()
    cfg.data.norm_mean = []
    cfg.data.norm_std = None
    cfg.aug.normalize.mean = [0.9, 0.8, 0.7]
    cfg.aug.normalize.std = [0.3, 0.2, 0.1]

    transform_tr, transform_te = build_transforms(
        256,
        128,
        norm_mean=[0.1, 0.2, 0.3],
        norm_std=[0.7, 0.8, 0.9],
        cfg=cfg,
    )
    norm_tr = next(t for t in transform_tr.transforms if isinstance(t, v2.Normalize))
    norm_te = next(t for t in transform_te.transforms if isinstance(t, v2.Normalize))

    assert norm_tr.mean == [0.1, 0.2, 0.3]
    assert norm_tr.std == [0.7, 0.8, 0.9]
    assert norm_te.mean == [0.1, 0.2, 0.3]
    assert norm_te.std == [0.7, 0.8, 0.9]


def test_disable_stochastic_blocks_data_transform_shortcut():
    """disable_stochastic must disable random transforms even if listed in data.transforms."""
    cfg = get_default_config()
    cfg.data.transforms = [
        "RandomHorizontalFlip",
        "random_crop",
        "ColorJitter",
        "random_patch",
        "rand_augment",
        "random_erase",
    ]
    cfg.aug.disable_stochastic = True

    transform_tr, _ = build_transforms(256, 128, cfg=cfg)

    assert not any(isinstance(t, v2.RandomHorizontalFlip) for t in transform_tr.transforms)
    assert not any(isinstance(t, v2.RandomResizedCrop) for t in transform_tr.transforms)
    assert not any(isinstance(t, v2.ColorJitter) for t in transform_tr.transforms)
    assert not any(isinstance(t, RandomPatch) for t in transform_tr.transforms)
    assert not any(isinstance(t, v2.RandAugment) for t in transform_tr.transforms)
    assert not any(isinstance(t, v2.RandomErasing) for t in transform_tr.transforms)


def test_legacy_color_jitter_subtree_applies_to_passthrough_transform():
    """Legacy snake_case config keys should still configure canonical passthrough transforms."""
    cfg = get_default_config()
    cfg.data.transforms = ["color_jitter"]
    cfg.aug.train.color_jitter = CN()
    cfg.aug.train.color_jitter.brightness = (0.1, 0.2)
    cfg.aug.train.color_jitter.contrast = (0.3, 0.4)

    transform_tr, _ = build_transforms(256, 128, cfg=cfg)
    color_jitter = next(t for t in transform_tr.transforms if isinstance(t, v2.ColorJitter))

    assert color_jitter.brightness == (0.1, 0.2)
    assert color_jitter.contrast == (0.3, 0.4)
