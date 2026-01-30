from typing import Literal

import torch
from torch import nn

from torchreid.utils.logging_config import logger

from .densenet import densenet121, densenet121_fc512, densenet161, densenet169, densenet201
from .hacnn import HACNN
from .inceptionresnetv2 import inceptionresnetv2
from .inceptionv4 import inceptionv4
from .mlfn import mlfn
from .mobilenetv2 import mobilenetv2_x1_0, mobilenetv2_x1_4
from .mudeep import MuDeep
from .nasnet import nasnetamobile
from .osnet import osnet_ibn_x1_0, osnet_x0_5, osnet_x0_25, osnet_x0_75, osnet_x1_0
from .osnet_ain import osnet_ain_x0_5, osnet_ain_x0_25, osnet_ain_x0_75, osnet_ain_x1_0
from .pcb import pcb_p4, pcb_p6
from .resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet50_fc512,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
)
from .resnet_ibn_a import resnet50_ibn_a
from .resnet_ibn_b import resnet50_ibn_b
from .resnetmid import resnet50mid
from .senet import se_resnet50, se_resnet50_fc512, se_resnet101, se_resnext50_32x4d, se_resnext101_32x4d
from .shufflenet import shufflenet
from .shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from .squeezenet import squeezenet1_0, squeezenet1_0_fc512, squeezenet1_1
from .xception import xception

__model_factory = {
    # image classification models
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "resnet50_fc512": resnet50_fc512,
    "se_resnet50": se_resnet50,
    "se_resnet50_fc512": se_resnet50_fc512,
    "se_resnet101": se_resnet101,
    "se_resnext50_32x4d": se_resnext50_32x4d,
    "se_resnext101_32x4d": se_resnext101_32x4d,
    "densenet121": densenet121,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "densenet161": densenet161,
    "densenet121_fc512": densenet121_fc512,
    "inceptionresnetv2": inceptionresnetv2,
    "inceptionv4": inceptionv4,
    "xception": xception,
    "resnet50_ibn_a": resnet50_ibn_a,
    "resnet50_ibn_b": resnet50_ibn_b,
    # lightweight models
    "nasnsetmobile": nasnetamobile,
    "mobilenetv2_x1_0": mobilenetv2_x1_0,
    "mobilenetv2_x1_4": mobilenetv2_x1_4,
    "shufflenet": shufflenet,
    "squeezenet1_0": squeezenet1_0,
    "squeezenet1_0_fc512": squeezenet1_0_fc512,
    "squeezenet1_1": squeezenet1_1,
    "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": shufflenet_v2_x2_0,
    # reid-specific models
    "mudeep": MuDeep,
    "resnet50mid": resnet50mid,
    "hacnn": HACNN,
    "pcb_p6": pcb_p6,
    "pcb_p4": pcb_p4,
    "mlfn": mlfn,
    "osnet_x1_0": osnet_x1_0,
    "osnet_x0_75": osnet_x0_75,
    "osnet_x0_5": osnet_x0_5,
    "osnet_x0_25": osnet_x0_25,
    "osnet_ibn_x1_0": osnet_ibn_x1_0,
    "osnet_ain_x1_0": osnet_ain_x1_0,
    "osnet_ain_x0_75": osnet_ain_x0_75,
    "osnet_ain_x0_5": osnet_ain_x0_5,
    "osnet_ain_x0_25": osnet_ain_x0_25,
}


def show_avai_models() -> None:
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    logger.info("%s", list(__model_factory.keys()))


def build_model(
    name: str,
    num_classes: int,
    loss: Literal["softmax", "triplet"] = "softmax",
    pretrained: bool = True,
    use_gpu: bool = True,
) -> nn.Module:
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(f"Unknown model: {name}. Must be one of {avai_models}")
    return __model_factory[name](num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu)
