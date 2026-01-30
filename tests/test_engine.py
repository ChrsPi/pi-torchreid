import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torchreid


class DummyDataset(Dataset):
    def __init__(self, num_samples=4, num_classes=3, shape=(3, 8, 4)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.shape = shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img = torch.randn(*self.shape)
        pid = torch.tensor(index % self.num_classes, dtype=torch.int64)
        return {"img": img, "pid": pid, "camid": torch.tensor(0, dtype=torch.int64)}


class DummyDataManager:
    def __init__(self, num_classes=3):
        self.num_train_pids = num_classes
        self.num_train_cams = 1
        self.sources = ["dummy"]
        self.targets = ["dummy"]
        self.data_type = "image"
        self.height = 8
        self.width = 4
        dataset = DummyDataset(num_samples=6, num_classes=num_classes)
        self.train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        self.test_loader = {}

    def fetch_test_loaders(self, name):
        return None, None


class DummyEngine(torchreid.engine.Engine):
    def forward_backward(self, data):
        return {"loss": 0.0}


class DummyTripletModel(nn.Module):
    def __init__(self, num_classes=3, feat_dim=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 4, feat_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.features(x)
        outputs = self.classifier(feats)
        return outputs, feats


class DummySoftmaxModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 4, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def test_register_model_and_get_names():
    datamanager = DummyDataManager()
    engine = DummyEngine(datamanager)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    engine.register_model("model", model, optimizer, scheduler)

    assert engine.get_model_names() == ["model"]
    assert engine.get_model_names("model") == ["model"]


def test_set_model_mode():
    datamanager = DummyDataManager()
    engine = DummyEngine(datamanager)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    engine.register_model("model", model, optimizer, scheduler)

    engine.set_model_mode("eval")
    assert not model.training
    engine.set_model_mode("train")
    assert model.training


def test_save_model(tmp_path):
    datamanager = DummyDataManager()
    engine = DummyEngine(datamanager)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    engine.register_model("model", model, optimizer, scheduler)

    save_dir = tmp_path / "ckpt"
    engine.save_model(epoch=0, rank1=0.5, save_dir=str(save_dir))
    assert (save_dir / "model" / "model.pth.tar-1").exists()


def test_image_softmax_engine_forward_backward():
    datamanager = DummyDataManager(num_classes=3)
    model = DummySoftmaxModel(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        use_gpu=False,
    )

    batch = next(iter(datamanager.train_loader))
    summary = engine.forward_backward(batch)
    assert "loss" in summary
    assert "acc" in summary


def test_image_triplet_engine_forward_backward():
    datamanager = DummyDataManager(num_classes=3)
    model = DummyTripletModel(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        use_gpu=False,
        weight_t=1.0,
        weight_x=1.0,
    )

    batch = next(iter(datamanager.train_loader))
    summary = engine.forward_backward(batch)
    assert "loss_t" in summary
    assert "loss_x" in summary
    assert "acc" in summary


def test_image_triplet_engine_weight_combinations():
    datamanager = DummyDataManager(num_classes=3)
    model = DummyTripletModel(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        use_gpu=False,
        weight_t=0.0,
        weight_x=1.0,
    )

    batch = next(iter(datamanager.train_loader))
    summary = engine.forward_backward(batch)
    assert "loss_x" in summary
    assert "loss_t" not in summary


def test_video_softmax_pooling():
    datamanager = DummyDataManager(num_classes=3)
    model = DummySoftmaxModel(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    engine = torchreid.engine.VideoSoftmaxEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        use_gpu=False,
        pooling_method="avg",
    )

    batch_size, seq_len = 2, 3
    imgs = torch.randn(batch_size, seq_len, 3, 8, 4)
    feats = engine.extract_features(imgs)
    assert feats.shape == (batch_size, datamanager.num_train_pids)
