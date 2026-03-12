from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from torchreid.data.transforms import build_transforms
from torchreid.models import build_model
from torchreid.utils import check_isfile, compute_model_complexity, load_pretrained_weights
from torchreid.utils.logging_config import logger


class FeatureExtractor:
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        cfg: optional config object used to build the shared evaluation transform.
        preprocess: optional prebuilt preprocessing callable. If provided, it
            overrides ``cfg`` and the simple built-in resize/normalize path.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        model_name: str = "",
        model_path: str = "",
        image_size: Sequence[int] = (256, 128),
        pixel_mean: Sequence[float] | None = None,
        pixel_std: Sequence[float] | None = None,
        pixel_norm: bool = True,
        cfg: object | None = None,
        preprocess: Callable | None = None,
        device: str = "cuda",
        verbose: bool = True,
    ) -> None:
        # Build model
        if pixel_std is None:
            pixel_std = [0.229, 0.224, 0.225]
        if pixel_mean is None:
            pixel_mean = [0.485, 0.456, 0.406]
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (model_path and check_isfile(model_path)),
            use_gpu=device.startswith("cuda"),
        )
        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(model, (1, 3, image_size[0], image_size[1]))
            logger.info("Model: %s", model_name)
            logger.info("- params: %s", f"{num_params:,}")
            logger.info("- flops: %s", f"{flops:,}")

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        # Reuse the shared evaluation pipeline when cfg is provided; otherwise keep
        # the historical simple preprocessing behavior.
        if preprocess is None and cfg is not None:
            _, preprocess = build_transforms(
                image_size[0],
                image_size[1],
                norm_mean=pixel_mean,
                norm_std=pixel_std,
                cfg=cfg,
            )
        if preprocess is None:
            transforms = []
            transforms += [T.Resize(image_size)]
            transforms += [T.ToTensor()]
            if pixel_norm:
                transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
            preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(
        self,
        input: str | np.ndarray | torch.Tensor | Sequence[str | np.ndarray],
    ) -> torch.Tensor:
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert("RGB")

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError("Type of each element must belong to [str | numpy.ndarray]")

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert("RGB")
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features
