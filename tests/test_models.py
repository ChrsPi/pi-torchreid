"""Tests for model building and forward pass."""

import pytest
import torch
from torchreid import models

# Get all available models
ALL_MODELS = list(models.__model_factory.keys())

# Representative models for detailed testing
REPRESENTATIVE_MODELS = [
    "resnet18",
    "resnet50",
    "resnet101",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_ain_x1_0",
    "mobilenetv2_x1_0",
    "shufflenet_v2_x1_0",
    "pcb_p6",
    "hacnn",
    "mudeep",
]

# Models that support both softmax and triplet
DUAL_LOSS_MODELS = [
    "resnet50",
    "osnet_x1_0",
    "mobilenetv2_x1_0",
]


class TestModelBuilding:
    """Test model building functionality."""

    def test_show_avai_models(self, capsys):
        """Test show_avai_models() function."""
        models.show_avai_models()
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        # Check that some model names are in the output
        assert "resnet50" in captured.out or "osnet" in captured.out

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_build_model_valid_names(self, model_name):
        """Test building all available models."""
        model = models.build_model(name=model_name, num_classes=10, loss="softmax")
        assert model is not None
        assert hasattr(model, "feat_dim")
        assert isinstance(model.feat_dim, int)
        assert model.feat_dim > 0

    def test_build_model_invalid_name(self):
        """Test error handling for invalid model names."""
        with pytest.raises(KeyError, match="Unknown model"):
            models.build_model(name="invalid_model_name", num_classes=10, loss="softmax")

    @pytest.mark.parametrize("loss", ["softmax", "triplet"])
    def test_build_model_loss_modes(self, loss):
        """Test building models with different loss modes."""
        model = models.build_model(name="resnet50", num_classes=10, loss=loss)
        assert model is not None
        assert hasattr(model, "loss")
        assert model.loss == loss

    @pytest.mark.parametrize("pretrained", [True, False])
    def test_build_model_pretrained(self, pretrained):
        """Test building models with and without pretrained weights."""
        model = models.build_model(
            name="resnet50", num_classes=10, loss="softmax", pretrained=pretrained
        )
        assert model is not None

    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_build_model_use_gpu(self, use_gpu):
        """Test building models with use_gpu parameter."""
        model = models.build_model(
            name="resnet50", num_classes=10, loss="softmax", use_gpu=use_gpu
        )
        assert model is not None


class TestModelForward:
    """Test model forward pass."""

    @pytest.mark.parametrize("model_name", REPRESENTATIVE_MODELS)
    def test_model_forward_softmax(self, model_name, dummy_batch, device):
        """Test forward pass for representative models with softmax loss."""
        model = models.build_model(
            name=model_name, num_classes=10, loss="softmax", pretrained=False
        )
        model = model.to(device).eval()
        with torch.no_grad():
            out = model(dummy_batch["img"].to(device))
        assert out.shape[0] == dummy_batch["img"].shape[0]
        assert out.shape[1] == 10  # num_classes

    @pytest.mark.parametrize("model_name", DUAL_LOSS_MODELS)
    def test_model_forward_triplet(self, model_name, dummy_batch, device):
        """Test forward pass for models with triplet loss."""
        model = models.build_model(
            name=model_name, num_classes=10, loss="triplet", pretrained=False
        )
        model = model.to(device).eval()
        with torch.no_grad():
            outputs = model(dummy_batch["img"].to(device))
            # Triplet loss mode returns (logits, features)
            assert isinstance(outputs, tuple)
            assert len(outputs) == 2
            logits, features = outputs
            assert logits.shape[0] == dummy_batch["img"].shape[0]
            assert logits.shape[1] == 10  # num_classes
            assert features.shape[0] == dummy_batch["img"].shape[0]
            assert features.shape[1] == model.feat_dim

    @pytest.mark.parametrize("model_name", REPRESENTATIVE_MODELS)
    def test_model_forward_training_mode(self, model_name, dummy_batch, device):
        """Test forward pass in training mode."""
        model = models.build_model(
            name=model_name, num_classes=10, loss="softmax", pretrained=False
        )
        model = model.to(device).train()
        out = model(dummy_batch["img"].to(device))
        assert out.shape[0] == dummy_batch["img"].shape[0]
        assert out.shape[1] == 10  # num_classes

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_model_forward_all_models(self, model_name, dummy_batch, device):
        """Test forward pass for all models (quick shape check)."""
        try:
            model = models.build_model(
                name=model_name, num_classes=10, loss="softmax", pretrained=False
            )
            model = model.to(device).eval()
            with torch.no_grad():
                out = model(dummy_batch["img"].to(device))
            # Some models might return tuple, handle both cases
            if isinstance(out, tuple):
                out = out[0]
            assert out.shape[0] == dummy_batch["img"].shape[0]
        except Exception as e:
            pytest.fail(f"Model {model_name} failed forward pass: {e}")


class TestModelProperties:
    """Test model properties and utilities."""

    @pytest.mark.parametrize("model_name", REPRESENTATIVE_MODELS)
    def test_model_feat_dim(self, model_name):
        """Test that models have feat_dim attribute."""
        model = models.build_model(
            name=model_name, num_classes=10, loss="softmax", pretrained=False
        )
        assert hasattr(model, "feat_dim")
        assert isinstance(model.feat_dim, int)
        assert model.feat_dim > 0

    @pytest.mark.parametrize("model_name", REPRESENTATIVE_MODELS)
    def test_model_device_move(self, model_name, device):
        """Test that models can be moved to different devices."""
        model = models.build_model(
            name=model_name, num_classes=10, loss="softmax", pretrained=False
        )
        model = model.to(device)
        # Check that model parameters are on the correct device
        for param in model.parameters():
            assert param.device == device

    @pytest.mark.parametrize("model_name", REPRESENTATIVE_MODELS)
    def test_model_state_dict(self, model_name, tmp_path):
        """Test that model state dict can be saved and loaded."""
        model1 = models.build_model(
            name=model_name, num_classes=10, loss="softmax", pretrained=False
        )
        state_dict = model1.state_dict()
        assert len(state_dict) > 0

        # Save and load
        checkpoint_path = tmp_path / "model.pth"
        torch.save(state_dict, checkpoint_path)
        assert checkpoint_path.exists()

        # Load into new model
        model2 = models.build_model(
            name=model_name, num_classes=10, loss="softmax", pretrained=False
        )
        model2.load_state_dict(torch.load(checkpoint_path))
        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)

    @pytest.mark.parametrize("model_name", REPRESENTATIVE_MODELS)
    def test_model_num_classes(self, model_name):
        """Test that models respect num_classes parameter."""
        for num_classes in [10, 100, 751]:
            model = models.build_model(
                name=model_name, num_classes=num_classes, loss="softmax", pretrained=False
            )
            # Test forward pass to verify output shape
            dummy_input = torch.randn(2, 3, 256, 128)
            with torch.no_grad():
                out = model(dummy_input)
                if isinstance(out, tuple):
                    out = out[0]
                assert out.shape[1] == num_classes
