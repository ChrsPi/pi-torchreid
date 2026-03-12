"""Regression tests for FeatureExtractor preprocessing parity."""

import numpy as np
from PIL import Image
import torch

from pi_torchreid.data.datamanager import DataManager
from pi_torchreid.data.transforms import build_transforms
import pi_torchreid.utils.feature_extractor as feature_extractor_module
from scripts.default_config import get_default_config


def _make_cfg() -> object:
    cfg = get_default_config()
    cfg.data.height = 32
    cfg.data.width = 16
    cfg.data.transforms = ["RandomHorizontalFlip"]
    cfg.data.norm_mean = [0.25, 0.35, 0.45]
    cfg.data.norm_std = [0.5, 0.4, 0.3]
    cfg.aug.test.brightness.enabled = True
    cfg.aug.test.brightness.factor = 0.6
    return cfg


def _stub_model(*args, **kwargs) -> torch.nn.Module:
    return torch.nn.Identity()


def test_feature_extractor_cfg_uses_shared_eval_transform(monkeypatch):
    monkeypatch.setattr(feature_extractor_module, "build_model", _stub_model)
    cfg = _make_cfg()
    image = np.full((24, 12, 3), 180, dtype=np.uint8)

    extractor = feature_extractor_module.FeatureExtractor(
        model_name="ignored",
        cfg=cfg,
        device="cpu",
        verbose=False,
    )
    _, transform_te = build_transforms(
        cfg.data.height,
        cfg.data.width,
        norm_mean=[0.1, 0.2, 0.3],
        norm_std=[0.9, 0.8, 0.7],
        cfg=cfg,
    )

    features = extractor(image)
    expected = transform_te(Image.fromarray(image)).unsqueeze(0)

    assert torch.allclose(features, expected)


def test_feature_extractor_matches_datamanager_test_pipeline(monkeypatch):
    monkeypatch.setattr(feature_extractor_module, "build_model", _stub_model)
    cfg = _make_cfg()
    image = Image.new("RGB", (20, 40), color=(120, 90, 60))

    extractor = feature_extractor_module.FeatureExtractor(
        model_name="ignored",
        cfg=cfg,
        device="cpu",
        verbose=False,
    )
    datamanager = DataManager(sources="dummy", use_gpu=False, cfg=cfg)

    assert torch.allclose(extractor.preprocess(image), datamanager.preprocess_pil_img(image))


def test_feature_extractor_preprocess_override_beats_cfg(monkeypatch):
    monkeypatch.setattr(feature_extractor_module, "build_model", _stub_model)
    cfg = _make_cfg()
    expected = torch.ones(3, 4, 5, dtype=torch.float32)

    extractor = feature_extractor_module.FeatureExtractor(
        model_name="ignored",
        cfg=cfg,
        preprocess=lambda _img: expected,
        device="cpu",
        verbose=False,
    )

    features = extractor(np.zeros((10, 6, 3), dtype=np.uint8))

    assert torch.equal(features, expected.unsqueeze(0))


def test_feature_extractor_verbose_flops_uses_cfg_image_size(monkeypatch):
    monkeypatch.setattr(feature_extractor_module, "build_model", _stub_model)
    cfg = _make_cfg()
    complexity_calls = []

    def _stub_complexity(model, input_size):
        complexity_calls.append(input_size)
        return 1, 2

    monkeypatch.setattr(feature_extractor_module, "compute_model_complexity", _stub_complexity)

    feature_extractor_module.FeatureExtractor(
        model_name="ignored",
        image_size=(256, 128),
        cfg=cfg,
        device="cpu",
        verbose=True,
    )

    assert complexity_calls == [(1, 3, cfg.data.height, cfg.data.width)]
