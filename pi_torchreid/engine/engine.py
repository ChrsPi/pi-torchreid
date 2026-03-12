from collections import OrderedDict
from collections.abc import Sequence
import datetime
import os.path as osp
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from pi_torchreid import metrics
from pi_torchreid.losses import DeepSupervision
from pi_torchreid.utils import (
    AverageMeter,
    MetricMeter,
    logger,
    open_all_layers,
    open_specified_layers,
    re_ranking,
    save_checkpoint,
    visualize_ranked_results,
)


class Engine:
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``pi_torchreid.data.ImageDataManager``
            or ``pi_torchreid.data.VideoDataManager``.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager: Any, use_gpu: bool = True) -> None:
        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = torch.cuda.is_available() and use_gpu
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.writer = None
        self.epoch = 0

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def register_model(
        self,
        name: str = "model",
        model: torch.nn.Module | None = None,
        optim: torch.optim.Optimizer | None = None,
        sched: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        if self.__dict__.get("_models") is None:
            raise AttributeError("Cannot assign model before super().__init__() call")

        if self.__dict__.get("_optims") is None:
            raise AttributeError("Cannot assign optim before super().__init__() call")

        if self.__dict__.get("_scheds") is None:
            raise AttributeError("Cannot assign sched before super().__init__() call")

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names: str | list[str] | None = None) -> list[str]:
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                if name not in names_real:
                    raise ValueError(f"Unknown model name: {name}. Must be one of {names_real}")
            return names
        else:
            return names_real

    def save_model(self, epoch: int, rank1: float, save_dir: str, is_best: bool = False) -> None:
        names = self.get_model_names()

        for name in names:
            save_checkpoint(
                {
                    "state_dict": self._models[name].state_dict(),
                    "epoch": epoch + 1,
                    "rank1": rank1,
                    "optimizer": self._optims[name].state_dict(),
                    "scheduler": self._scheds[name].state_dict(),
                },
                osp.join(save_dir, name),
                is_best=is_best,
            )

    def set_model_mode(self, mode: str = "train", names: str | list[str] | None = None) -> None:
        if mode not in ["train", "eval", "test"]:
            raise ValueError(f"mode must be one of ['train', 'eval', 'test'], but got {mode}")
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            else:
                self._models[name].eval()

    def get_current_lr(self, names: str | list[str] | None = None) -> float:
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[-1]["lr"]

    def update_lr(self, names: str | list[str] | None = None) -> None:
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def run(
        self,
        save_dir: str = "log",
        max_epoch: int = 0,
        start_epoch: int = 0,
        print_freq: int = 10,
        fixbase_epoch: int = 0,
        open_layers: str | Sequence[str] | None = None,
        start_eval: int = 0,
        eval_freq: int = -1,
        test_only: bool = False,
        dist_metric: str = "euclidean",
        normalize_feature: bool = False,
        visrank: bool = False,
        visrank_topk: int = 10,
        use_metric_cuhk03: bool = False,
        ranks: Sequence[int] | None = None,
        rerank: bool = False,
    ) -> None:
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """

        if ranks is None:
            ranks = [1, 5, 10, 20]
        if visrank and not test_only:
            raise ValueError("visrank can be set to True only if test_only=True")

        if test_only:
            self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        logger.info("=> Start training")

        for epoch in range(self.start_epoch, self.max_epoch):
            self.epoch = epoch
            self.train(print_freq=print_freq, fixbase_epoch=fixbase_epoch, open_layers=open_layers)

            if (
                (self.epoch + 1) >= start_eval
                and eval_freq > 0
                and (self.epoch + 1) % eval_freq == 0
                and (self.epoch + 1) != self.max_epoch
            ):
                rank1 = self.test(
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                )
                self.save_model(self.epoch, rank1, save_dir)

        if self.max_epoch > 0:
            logger.info("=> Final test")
            rank1 = self.test(
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
            )
            self.save_model(self.epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info("Elapsed %s", elapsed)
        if self.writer is not None:
            self.writer.close()

    def train(
        self,
        print_freq: int = 10,
        fixbase_epoch: int = 0,
        open_layers: str | Sequence[str] | None = None,
    ) -> None:
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode("train")

        self.two_stepped_transfer_learning(self.epoch, fixbase_epoch, open_layers)

        self.num_batches = len(self.train_loader)
        end = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            self.batch_idx = batch_idx
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (self.max_epoch - (self.epoch + 1)) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    f"epoch: [{self.epoch + 1}/{self.max_epoch}][{self.batch_idx + 1}/{self.num_batches}]\t"
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"eta {eta_str}\t"
                    f"{losses}\t"
                    f"lr {self.get_current_lr():.6f}"
                )

            if self.writer is not None:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar("Train/time", batch_time.avg, n_iter)
                self.writer.add_scalar("Train/data", data_time.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar("Train/" + name, meter.avg, n_iter)
                self.writer.add_scalar("Train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        self.update_lr()

    def forward_backward(self, data: Any) -> dict[str, float]:
        raise NotImplementedError

    def test(
        self,
        dist_metric: str = "euclidean",
        normalize_feature: bool = False,
        visrank: bool = False,
        visrank_topk: int = 10,
        save_dir: str = "",
        use_metric_cuhk03: bool = False,
        ranks: Sequence[int] | None = None,
        rerank: bool = False,
    ) -> float:
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        if ranks is None:
            ranks = [1, 5, 10, 20]
        self.set_model_mode("eval")
        targets = list(self.test_loader.keys())

        for name in targets:
            domain = "source" if name in self.datamanager.sources else "target"
            logger.info("##### Evaluating %s (%s) #####", name, domain)
            query_loader = self.test_loader[name]["query"]
            gallery_loader = self.test_loader[name]["gallery"]
            rank1, mAP = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank,
            )

            if self.writer is not None:
                self.writer.add_scalar(f"Test/{name}/rank1", rank1, self.epoch)
                self.writer.add_scalar(f"Test/{name}/mAP", mAP, self.epoch)

        return rank1

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name: str = "",
        query_loader: torch.utils.data.DataLoader | None = None,
        gallery_loader: torch.utils.data.DataLoader | None = None,
        dist_metric: str = "euclidean",
        normalize_feature: bool = False,
        visrank: bool = False,
        visrank_topk: int = 10,
        save_dir: str = "",
        use_metric_cuhk03: bool = False,
        ranks: Sequence[int] | None = None,
        rerank: bool = False,
    ) -> tuple[float, float]:
        if ranks is None:
            ranks = [1, 5, 10, 20]
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for _batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.to(self.device)
                end = time.time()
                features = self.extract_features(imgs)
                batch_time.update(time.time() - end)
                features = features.to("cpu")
                f_.append(features)
                pids_.extend(pids.tolist())
                camids_.extend(camids.tolist())
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        logger.info("Extracting features from query set ...")
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        logger.info("Done, obtained %s-by-%s matrix", qf.size(0), qf.size(1))

        logger.info("Extracting features from gallery set ...")
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        logger.info("Done, obtained %s-by-%s matrix", gf.size(0), gf.size(1))

        logger.info("Speed: %.4f sec/batch", batch_time.avg)

        if normalize_feature:
            logger.info("Normalzing features with L2 norm ...")
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        logger.info("Computing distance matrix with metric=%s ...", dist_metric)
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            logger.info("Applying person re-ranking ...")
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        logger.info("Computing CMC and mAP ...")
        cmc, mAP = metrics.evaluate_rank(
            distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=use_metric_cuhk03
        )

        logger.info("** Results **")
        logger.info("mAP: %.1f%%", mAP * 100)
        logger.info("CMC curve")
        for r in ranks:
            logger.info("Rank-%-3d: %.1f%%", r, cmc[r - 1] * 100)

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, "visrank_" + dataset_name),
                topk=visrank_topk,
            )

        return cmc[0], mAP

    def compute_loss(
        self, criterion: Any, outputs: torch.Tensor | Sequence[torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def extract_features(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)

    def parse_data_for_train(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        imgs = data["img"]
        pids = data["pid"]
        return imgs, pids

    def parse_data_for_eval(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = data["img"]
        pids = data["pid"]
        camids = data["camid"]
        return imgs, pids, camids

    def two_stepped_transfer_learning(
        self,
        epoch: int,
        fixbase_epoch: int,
        open_layers: str | Sequence[str] | None,
        model: torch.nn.Module | None = None,
    ) -> None:
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """
        model = self.model if model is None else model
        if model is None:
            return

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            logger.info("* Only train %s (epoch: %s/%s)", open_layers, epoch + 1, fixbase_epoch)
            open_specified_layers(model, open_layers)
        else:
            open_all_layers(model)
