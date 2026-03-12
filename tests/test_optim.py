"""Tests for optimizers and learning rate schedulers."""

import pytest
import torch
import torch.nn as nn

from pi_torchreid.optim import build_lr_scheduler, build_optimizer


class TestBuildOptimizer:
    """Test build_optimizer function."""

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        return model

    @pytest.mark.parametrize("optim_name", ["adam", "amsgrad", "sgd", "rmsprop", "radam"])
    def test_build_optimizer_all_types(self, optim_name, dummy_model):
        """Test building all available optimizer types."""
        optimizer = build_optimizer(dummy_model, optim=optim_name, lr=0.001)
        assert optimizer is not None
        assert len(list(optimizer.param_groups)) > 0

    def test_build_optimizer_invalid_name(self, dummy_model):
        """Test error handling for invalid optimizer names."""
        with pytest.raises(ValueError, match="Unsupported optim"):
            build_optimizer(dummy_model, optim="invalid_optim")

    def test_build_optimizer_invalid_model(self):
        """Test error handling for invalid model type."""
        with pytest.raises(TypeError, match="must be an instance of nn.Module"):
            build_optimizer("not_a_model", optim="adam")

    def test_build_optimizer_staged_lr(self, dummy_model):
        """Test staged learning rates."""

        # Create model with named layers
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(10, 20)
                self.classifier = nn.Linear(20, 10)

            def forward(self, x):
                x = self.base(x)
                return self.classifier(x)

        model = TestModel()
        optimizer = build_optimizer(
            model,
            optim="sgd",
            lr=0.01,
            staged_lr=True,
            new_layers="classifier",
            base_lr_mult=0.1,
        )
        # Check that different parameter groups have different learning rates
        lrs = [group["lr"] for group in optimizer.param_groups]
        assert len(set(lrs)) > 1  # Should have different LRs
        # Base layers should have lr * base_lr_mult
        assert any(lr == 0.01 * 0.1 for lr in lrs)
        # New layers should have full lr
        assert any(lr == 0.01 for lr in lrs)

    def test_build_optimizer_staged_lr_multiple_layers(self, dummy_model):
        """Test staged learning rates with multiple new layers."""

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(10, 20)
                self.fc = nn.Linear(20, 15)
                self.classifier = nn.Linear(15, 10)

            def forward(self, x):
                x = self.base(x)
                x = self.fc(x)
                return self.classifier(x)

        model = TestModel()
        optimizer = build_optimizer(
            model,
            optim="sgd",
            lr=0.01,
            staged_lr=True,
            new_layers=["fc", "classifier"],
            base_lr_mult=0.1,
        )
        lrs = [group["lr"] for group in optimizer.param_groups]
        assert len(set(lrs)) > 1

    def test_build_optimizer_parameters(self, dummy_model):
        """Test optimizer parameters."""
        optimizer = build_optimizer(dummy_model, optim="adam", lr=0.001, weight_decay=0.0001)
        param_group = optimizer.param_groups[0]
        assert param_group["lr"] == 0.001
        assert param_group["weight_decay"] == 0.0001

    def test_build_optimizer_sgd_momentum(self, dummy_model):
        """Test SGD with momentum."""
        optimizer = build_optimizer(dummy_model, optim="sgd", lr=0.01, momentum=0.9, sgd_nesterov=True)
        param_group = optimizer.param_groups[0]
        assert param_group["momentum"] == 0.9
        assert param_group["nesterov"] is True


class TestBuildLRScheduler:
    """Test build_lr_scheduler function."""

    @pytest.fixture
    def dummy_optimizer(self):
        """Create a dummy optimizer."""
        params = [torch.tensor(1.0, requires_grad=True)]
        return torch.optim.SGD(params, lr=0.1)

    def test_build_lr_scheduler_single_step(self, dummy_optimizer):
        """Test single step scheduler."""
        scheduler = build_lr_scheduler(dummy_optimizer, lr_scheduler="single_step", stepsize=10, gamma=0.1)
        assert scheduler is not None
        initial_lr = dummy_optimizer.param_groups[0]["lr"]
        # Step 10 times (should trigger decay)
        for _ in range(11):
            dummy_optimizer.step()
            scheduler.step()
        assert dummy_optimizer.param_groups[0]["lr"] < initial_lr

    def test_build_lr_scheduler_multi_step(self, dummy_optimizer):
        """Test multi step scheduler."""
        scheduler = build_lr_scheduler(dummy_optimizer, lr_scheduler="multi_step", stepsize=[5, 10, 15], gamma=0.1)
        assert scheduler is not None
        initial_lr = dummy_optimizer.param_groups[0]["lr"]
        # Step past milestones
        for _ in range(16):
            dummy_optimizer.step()
            scheduler.step()
        assert dummy_optimizer.param_groups[0]["lr"] < initial_lr

    def test_build_lr_scheduler_cosine(self, dummy_optimizer):
        """Test cosine annealing scheduler."""
        scheduler = build_lr_scheduler(dummy_optimizer, lr_scheduler="cosine", max_epoch=10)
        assert scheduler is not None
        initial_lr = dummy_optimizer.param_groups[0]["lr"]
        # Step a few times
        for _ in range(5):
            dummy_optimizer.step()
            scheduler.step()
        # LR should change (cosine annealing)
        current_lr = dummy_optimizer.param_groups[0]["lr"]
        assert current_lr != initial_lr

    def test_build_lr_scheduler_invalid_name(self, dummy_optimizer):
        """Test error handling for invalid scheduler names."""
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            build_lr_scheduler(dummy_optimizer, lr_scheduler="invalid")

    def test_build_lr_scheduler_single_step_type_error(self, dummy_optimizer):
        """Test type error for single_step with list."""
        # Should convert list to last element
        scheduler = build_lr_scheduler(dummy_optimizer, lr_scheduler="single_step", stepsize=[10, 20])
        assert scheduler is not None

    def test_build_lr_scheduler_multi_step_type_error(self, dummy_optimizer):
        """Test type error for multi_step with non-list."""
        with pytest.raises(TypeError, match="must be a list"):
            build_lr_scheduler(dummy_optimizer, lr_scheduler="multi_step", stepsize=10)

    def test_build_lr_scheduler_gamma(self, dummy_optimizer):
        """Test gamma parameter."""
        scheduler = build_lr_scheduler(dummy_optimizer, lr_scheduler="single_step", stepsize=5, gamma=0.5)
        initial_lr = dummy_optimizer.param_groups[0]["lr"]
        # Step past stepsize
        for _ in range(6):
            dummy_optimizer.step()
            scheduler.step()
        # Should decay by gamma
        expected_lr = initial_lr * 0.5
        assert abs(dummy_optimizer.param_groups[0]["lr"] - expected_lr) < 1e-6
