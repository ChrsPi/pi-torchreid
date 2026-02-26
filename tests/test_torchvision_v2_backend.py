"""Regression tests for torchvision v2 backend config handling."""

from torchvision.transforms import v2

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
    cfg.aug.train.random_flip.enabled = False

    transform_tr, _ = build_transforms(256, 128, transforms=["random_flip"], cfg=cfg)

    assert not any(isinstance(t, v2.RandomHorizontalFlip) for t in transform_tr.transforms)


def test_explicit_norm_overrides_aug_normalize():
    """Explicit norm_mean/norm_std must override aug.normalize.* when both exist."""
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

    assert norm_tr.mean == [0.1, 0.2, 0.3]
    assert norm_tr.std == [0.7, 0.8, 0.9]
    assert norm_te.mean == [0.1, 0.2, 0.3]
    assert norm_te.std == [0.7, 0.8, 0.9]


def test_disable_stochastic_blocks_data_transform_shortcut():
    """disable_stochastic must disable random transforms even if listed in data.transforms."""
    cfg = get_default_config()
    cfg.data.transforms = [
        "random_flip",
        "random_crop",
        "color_jitter",
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
