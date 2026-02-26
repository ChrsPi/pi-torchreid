"""Custom augmentation transforms (not in torchvision)."""

from collections import deque
import math
import random

from PIL import Image
import torch
from torchvision.transforms.v2 import functional as tv2f


class RandomPatch:
    """Random patch data augmentation (PIL-only).

    Maintains a patch pool; for each image, may paste a random patch from
    the pool to simulate occlusion. Works on PIL images before ToTensor.

    Reference:
        Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
    """

    def __init__(
        self,
        prob_happen: float = 0.5,
        pool_capacity: int = 50000,
        min_sample_size: int = 100,
        patch_min_area: float = 0.01,
        patch_max_area: float = 0.5,
        patch_min_ratio: float = 0.1,
        prob_rotate: float = 0.5,
        prob_flip_leftright: float = 0.5,
    ) -> None:
        self.prob_happen = prob_happen
        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio
        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright
        self.patchpool: deque = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def _generate_wh(self, width: int, height: int) -> tuple[int | None, int | None]:
        area = width * height
        for _ in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1.0 / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < width and h < height:
                return w, h
        return None, None

    def _transform_patch(self, patch: Image.Image) -> Image.Image:
        if random.uniform(0, 1) < self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) < self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        w, h = self._generate_wh(width, height)
        if w is not None and h is not None:
            x1 = random.randint(0, width - w)
            y1 = random.randint(0, height - h)
            self.patchpool.append(img.crop((x1, y1, x1 + w, y1 + h)))

        if len(self.patchpool) < self.min_sample_size:
            return img
        if random.uniform(0, 1) > self.prob_happen:
            return img

        patch = random.sample(self.patchpool, 1)[0]
        patch_w, patch_h = patch.size
        x1 = random.randint(0, width - patch_w)
        y1 = random.randint(0, height - patch_h)
        patch = self._transform_patch(patch)
        img.paste(patch, (x1, y1))
        return img


class ResolutionDegradation:
    """Downscale then upscale to simulate low-resolution capture.

    Operates on uint8 tensors. Downscales by ``scale`` factor then upscales
    back to ``target_size`` using bilinear interpolation.
    """

    def __init__(self, scale: float, target_size: tuple[int, int]) -> None:
        if not 0.0 < scale < 1.0:
            raise ValueError(f"scale must be in (0, 1), got {scale}")
        self.scale = scale
        self.target_size = target_size  # (H, W)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = self.target_size
        small_h = max(1, int(h * self.scale))
        small_w = max(1, int(w * self.scale))
        img = tv2f.resize(img, [small_h, small_w], antialias=True)
        return tv2f.resize(img, [h, w], antialias=True)


class DeterministicRotation:
    """Fixed-angle rotation for evaluation-time degradation.

    Operates on uint8 tensors.
    """

    def __init__(self, angle: float) -> None:
        self.angle = angle

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return tv2f.rotate(img, self.angle)


class DeterministicBrightness:
    """Fixed brightness adjustment for evaluation-time degradation.

    Operates on float32 tensors. ``factor`` < 1 darkens, > 1 brightens.
    """

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return tv2f.adjust_brightness(img, self.factor)


class DeterministicContrast:
    """Fixed contrast adjustment for evaluation-time degradation.

    Operates on float32 tensors. ``factor`` < 1 reduces contrast.
    """

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return tv2f.adjust_contrast(img, self.factor)
