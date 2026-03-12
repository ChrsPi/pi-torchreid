"""Tests for utility functions."""

import numpy as np
import pytest
import torch

from pi_torchreid.models import build_model
from pi_torchreid.utils import (
    check_isfile,
    compute_model_complexity,
    count_num_param,
    load_checkpoint,
    mkdir_if_missing,
    re_ranking,
    save_checkpoint,
)


class _PickleOnlyPayload:
    """Simple custom object used to exercise pickle-based checkpoint loads."""

    def __init__(self, value):
        self.value = value


class TestReRanking:
    """Test re_ranking function."""

    def test_re_ranking_basic(self):
        """Test basic re-ranking."""
        q_g_dist = np.random.rand(10, 100).astype(np.float32)
        q_q_dist = np.random.rand(10, 10).astype(np.float32)
        g_g_dist = np.random.rand(100, 100).astype(np.float32)

        reranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        assert reranked_dist.shape == q_g_dist.shape
        assert reranked_dist.dtype == np.float32

    def test_re_ranking_parameters(self):
        """Test re-ranking with different parameters."""
        q_g_dist = np.random.rand(5, 50).astype(np.float32)
        q_q_dist = np.random.rand(5, 5).astype(np.float32)
        g_g_dist = np.random.rand(50, 50).astype(np.float32)

        for k1, k2, lambda_val in [(10, 3, 0.2), (20, 6, 0.3), (30, 10, 0.5)]:
            reranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lambda_val)
            assert reranked_dist.shape == q_g_dist.shape

    def test_re_ranking_small_matrices(self):
        """Test re-ranking with small matrices."""
        q_g_dist = np.random.rand(3, 10).astype(np.float32)
        q_q_dist = np.random.rand(3, 3).astype(np.float32)
        g_g_dist = np.random.rand(10, 10).astype(np.float32)

        reranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=5, k2=2)
        assert reranked_dist.shape == q_g_dist.shape


class TestModelComplexity:
    """Test compute_model_complexity function."""

    def test_compute_model_complexity_basic(self):
        """Test basic model complexity computation."""
        model = build_model("resnet50", num_classes=10, loss="softmax", pretrained=False)
        num_params, flops = compute_model_complexity(model, (1, 3, 256, 128))
        assert num_params > 0
        assert flops > 0

    def test_compute_model_complexity_different_models(self):
        """Test with different models."""
        for model_name in ["resnet18", "osnet_x1_0", "mobilenetv2_x1_0"]:
            model = build_model(model_name, num_classes=10, loss="softmax", pretrained=False)
            num_params, flops = compute_model_complexity(model, (1, 3, 256, 128))
            assert num_params > 0
            assert flops > 0

    def test_compute_model_complexity_different_input_sizes(self):
        """Test with different input sizes."""
        model = build_model("resnet50", num_classes=10, loss="softmax", pretrained=False)
        for input_size in [(1, 3, 128, 64), (1, 3, 256, 128), (1, 3, 384, 192)]:
            num_params, flops = compute_model_complexity(model, input_size)
            assert num_params > 0
            assert flops > 0

    def test_compute_model_complexity_only_conv_linear(self):
        """Test with only_conv_linear parameter."""
        model = build_model("resnet50", num_classes=10, loss="softmax", pretrained=False)
        num_params1, flops1 = compute_model_complexity(model, (1, 3, 256, 128), only_conv_linear=True)
        num_params2, flops2 = compute_model_complexity(model, (1, 3, 256, 128), only_conv_linear=False)
        # Parameters should be the same
        assert num_params1 == num_params2
        # FLOPs might differ
        assert flops1 > 0
        assert flops2 > 0


class TestCheckpointUtilities:
    """Test checkpoint save/load utilities."""

    def test_save_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoints."""
        model = build_model("resnet50", num_classes=10, loss="softmax", pretrained=False)
        optimizer = torch.optim.Adam(model.parameters())

        state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 1,
            "rank1": 0.5,
        }

        save_dir = tmp_path / "checkpoints"
        save_checkpoint(state, str(save_dir), is_best=False)

        # Check that file was created
        checkpoint_file = save_dir / "model.pth.tar-1"
        assert checkpoint_file.exists()

        # Load checkpoint
        loaded = load_checkpoint(str(checkpoint_file))
        assert "state_dict" in loaded
        assert "epoch" in loaded
        assert loaded["epoch"] == 1
        assert loaded["rank1"] == 0.5

    def test_save_checkpoint_is_best(self, tmp_path):
        """Test saving best checkpoint."""
        model = build_model("resnet50", num_classes=10, loss="softmax", pretrained=False)
        state = {
            "state_dict": model.state_dict(),
            "epoch": 1,
        }

        save_dir = tmp_path / "checkpoints"
        save_checkpoint(state, str(save_dir), is_best=True)

        # Check that best file was created
        best_file = save_dir / "model-best.pth.tar"
        assert best_file.exists()

    def test_load_checkpoint_nonexistent(self):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent_file.pth.tar")

    def test_load_checkpoint_safe_mode_rejects_pickled_objects(self, tmp_path):
        """Safe mode should reject checkpoint payloads that require pickle code execution."""
        checkpoint_file = tmp_path / "unsafe_payload.pth.tar"
        torch.save({"epoch": 1, "payload": _PickleOnlyPayload("unsafe")}, checkpoint_file)

        with pytest.raises(RuntimeError, match="safe=False"):
            load_checkpoint(str(checkpoint_file))

    def test_load_checkpoint_trusted_mode_allows_pickled_objects(self, tmp_path):
        """Trusted mode should load legacy pickle payloads."""
        checkpoint_file = tmp_path / "trusted_payload.pth.tar"
        torch.save({"epoch": 1, "payload": _PickleOnlyPayload("trusted")}, checkpoint_file)

        loaded = load_checkpoint(str(checkpoint_file), safe=False)
        assert isinstance(loaded["payload"], _PickleOnlyPayload)
        assert loaded["payload"].value == "trusted"


class TestCountNumParam:
    """Test count_num_param function."""

    def test_count_num_param_basic(self):
        """Test basic parameter counting."""
        model = build_model("resnet50", num_classes=10, loss="softmax", pretrained=False)
        with pytest.warns(UserWarning, match="deprecated"):
            num_params = count_num_param(model)
        assert num_params > 0
        assert isinstance(num_params, (int, float))

    def test_count_num_param_different_models(self):
        """Test with different models."""
        for model_name in ["resnet18", "osnet_x1_0", "mobilenetv2_x1_0"]:
            model = build_model(model_name, num_classes=10, loss="softmax", pretrained=False)
            with pytest.warns(UserWarning, match="deprecated"):
                num_params = count_num_param(model)
            assert num_params > 0


class TestTools:
    """Test basic tool functions."""

    def test_mkdir_if_missing(self, tmp_path):
        """Test mkdir_if_missing function."""
        new_dir = tmp_path / "new_directory"
        mkdir_if_missing(str(new_dir))
        assert new_dir.exists()
        assert new_dir.is_dir()

        # Should not raise error if directory already exists
        mkdir_if_missing(str(new_dir))

    def test_check_isfile(self, tmp_path):
        """Test check_isfile function."""
        # Test with existing file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        assert check_isfile(str(test_file)) is True

        # Test with non-existent file
        with pytest.warns(UserWarning, match="No file found"):
            assert check_isfile(str(tmp_path / "nonexistent.txt")) is False

        # Test with directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        with pytest.warns(UserWarning, match="No file found"):
            assert check_isfile(str(test_dir)) is False
