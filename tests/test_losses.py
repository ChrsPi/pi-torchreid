"""Tests for loss functions."""

import pytest
import torch

from pi_torchreid.losses import CrossEntropyLoss, DeepSupervision, TripletLoss


class TestCrossEntropyLoss:
    """Test CrossEntropyLoss."""

    def test_cross_entropy_loss_basic(self, device):
        """Test basic forward pass with valid inputs."""
        loss_fn = CrossEntropyLoss(num_classes=11, label_smooth=True, use_gpu=device.type == "cuda")
        inputs = torch.randn(4, 10).to(device)
        targets = torch.tensor([0, 1, 2, 3]).to(device)
        loss = loss_fn(inputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_cross_entropy_loss_no_smoothing(self, device):
        """Test with label_smooth=False."""
        loss_fn = CrossEntropyLoss(num_classes=10, label_smooth=False, use_gpu=device.type == "cuda")
        inputs = torch.randn(4, 10).to(device)
        targets = torch.tensor([0, 1, 2, 3]).to(device)
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0

    @pytest.mark.parametrize("num_classes", [2, 10, 100, 751])
    def test_cross_entropy_loss_num_classes(self, num_classes, device):
        """Test with different num_classes values."""
        loss_fn = CrossEntropyLoss(num_classes=num_classes, label_smooth=True, use_gpu=device.type == "cuda")
        inputs = torch.randn(4, num_classes).to(device)
        targets = (torch.tensor([0, 1, 2, 3]) % num_classes).to(device)
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0

    @pytest.mark.parametrize("eps", [0.0, 0.1, 0.5])
    def test_cross_entropy_loss_eps(self, eps, device):
        """Test with different eps values."""
        loss_fn = CrossEntropyLoss(num_classes=10, eps=eps, label_smooth=True, use_gpu=device.type == "cuda")
        inputs = torch.randn(4, 10).to(device)
        targets = torch.tensor([0, 1, 2, 3]).to(device)
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0

    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_cross_entropy_loss_use_gpu(self, use_gpu):
        """Test with use_gpu parameter."""
        if use_gpu and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # For use_gpu=False, use CPU device
        test_device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        loss_fn = CrossEntropyLoss(num_classes=10, use_gpu=use_gpu)
        inputs = torch.randn(4, 10).to(test_device)
        targets = torch.tensor([0, 1, 2, 3]).to(test_device)
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0

    def test_cross_entropy_loss_single_class(self, device):
        """Test edge case with single class."""
        loss_fn = CrossEntropyLoss(num_classes=1, label_smooth=True, use_gpu=device.type == "cuda")
        inputs = torch.randn(4, 1).to(device)
        targets = torch.tensor([0, 0, 0, 0]).to(device)
        loss = loss_fn(inputs, targets)
        assert loss.item() >= 0

    def test_cross_entropy_loss_large_batch(self, device):
        """Test with large batch size."""
        loss_fn = CrossEntropyLoss(num_classes=10, label_smooth=True, use_gpu=device.type == "cuda")
        inputs = torch.randn(128, 10).to(device)
        targets = torch.randint(0, 10, (128,)).to(device)
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0

    def test_cross_entropy_loss_gradient(self, device):
        """Test gradient computation."""
        loss_fn = CrossEntropyLoss(num_classes=10, label_smooth=True, use_gpu=device.type == "cuda")
        inputs = torch.randn(4, 10, device=device, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3]).to(device)
        loss = loss_fn(inputs, targets)
        loss.backward()
        # Check that gradient was computed (should exist and not be None)
        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()


class TestTripletLoss:
    """Test TripletLoss."""

    def test_triplet_loss_basic(self):
        """Test basic forward pass with valid inputs."""
        loss_fn = TripletLoss(margin=0.3)
        features = torch.randn(4, 512)
        targets = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(features, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    @pytest.mark.parametrize("margin", [0.1, 0.3, 0.5, 1.0])
    def test_triplet_loss_margin(self, margin):
        """Test with different margin values."""
        loss_fn = TripletLoss(margin=margin)
        features = torch.randn(4, 512)
        targets = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(features, targets)
        assert loss.item() >= 0

    def test_triplet_loss_same_identity(self):
        """Test with all same identity (edge case)."""
        loss_fn = TripletLoss(margin=0.3)
        features = torch.randn(4, 512)
        targets = torch.tensor([0, 0, 0, 0])  # All same identity
        # This should still compute, but may have different behavior
        # since there are no negatives
        try:
            loss = loss_fn(features, targets)
            assert isinstance(loss, torch.Tensor)
        except (ValueError, RuntimeError):
            # Some implementations may raise error for no negatives
            pass
        else:
            assert torch.isfinite(loss)
            assert loss.item() >= 0

    def test_triplet_loss_different_identities(self):
        """Test with all different identities."""
        loss_fn = TripletLoss(margin=0.3)
        features = torch.randn(4, 512)
        targets = torch.tensor([0, 1, 2, 3])  # All different
        # This should still compute, but may have different behavior
        # since there are no positives
        try:
            loss = loss_fn(features, targets)
            assert isinstance(loss, torch.Tensor)
        except (ValueError, RuntimeError):
            # Some implementations may raise error for no positives
            pass
        else:
            assert torch.isfinite(loss)
            assert loss.item() >= 0

    def test_triplet_loss_valid_pairs(self):
        """Test with valid positive and negative pairs."""
        loss_fn = TripletLoss(margin=0.3)
        features = torch.randn(8, 512)
        targets = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])  # Multiple identities
        loss = loss_fn(features, targets)
        assert loss.item() >= 0

    def test_triplet_loss_different_feature_dims(self):
        """Test with different feature dimensions."""
        for feat_dim in [128, 256, 512, 1024]:
            loss_fn = TripletLoss(margin=0.3)
            features = torch.randn(4, feat_dim)
            targets = torch.tensor([0, 0, 1, 1])
            loss = loss_fn(features, targets)
            assert loss.item() >= 0

    def test_triplet_loss_gradient(self):
        """Test gradient computation."""
        loss_fn = TripletLoss(margin=0.3)
        features = torch.randn(4, 512, requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(features, targets)
        loss.backward()
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_triplet_loss_large_batch(self):
        """Test with large batch size."""
        loss_fn = TripletLoss(margin=0.3)
        features = torch.randn(64, 512)
        targets = torch.randint(0, 10, (64,))
        loss = loss_fn(features, targets)
        assert loss.item() >= 0


class TestDeepSupervision:
    """Test DeepSupervision helper function."""

    def test_deep_supervision_basic(self, device):
        """Test basic DeepSupervision functionality."""
        criterion = CrossEntropyLoss(num_classes=10, label_smooth=True, use_gpu=device.type == "cuda")
        xs = (torch.randn(4, 10).to(device), torch.randn(4, 10).to(device), torch.randn(4, 10).to(device))
        y = torch.tensor([0, 1, 2, 3]).to(device)
        loss = DeepSupervision(criterion, xs, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_deep_supervision_averaging(self, device):
        """Test that DeepSupervision averages losses correctly."""
        criterion = CrossEntropyLoss(num_classes=10, label_smooth=True, use_gpu=device.type == "cuda")
        xs = (torch.randn(4, 10).to(device), torch.randn(4, 10).to(device))
        y = torch.tensor([0, 1, 2, 3]).to(device)
        loss = DeepSupervision(criterion, xs, y)
        # Should be average of individual losses
        loss1 = criterion(xs[0], y)
        loss2 = criterion(xs[1], y)
        expected_loss = (loss1 + loss2) / 2
        assert torch.allclose(loss, expected_loss, rtol=1e-5)

    def test_deep_supervision_single_input(self, device):
        """Test DeepSupervision with single input."""
        criterion = CrossEntropyLoss(num_classes=10, label_smooth=True, use_gpu=device.type == "cuda")
        xs = (torch.randn(4, 10).to(device),)
        y = torch.tensor([0, 1, 2, 3]).to(device)
        loss = DeepSupervision(criterion, xs, y)
        expected_loss = criterion(xs[0], y)
        assert torch.allclose(loss, expected_loss, rtol=1e-5)

    def test_deep_supervision_multiple_inputs(self, device):
        """Test DeepSupervision with multiple inputs."""
        criterion = CrossEntropyLoss(num_classes=10, label_smooth=True, use_gpu=device.type == "cuda")
        xs = (
            torch.randn(4, 10).to(device),
            torch.randn(4, 10).to(device),
            torch.randn(4, 10).to(device),
            torch.randn(4, 10).to(device),
        )
        y = torch.tensor([0, 1, 2, 3]).to(device)
        loss = DeepSupervision(criterion, xs, y)
        # Should be average of all losses
        individual_losses = [criterion(x, y) for x in xs]
        expected_loss = sum(individual_losses) / len(individual_losses)
        assert torch.allclose(loss, expected_loss, rtol=1e-5)
