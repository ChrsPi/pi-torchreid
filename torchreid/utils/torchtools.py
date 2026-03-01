from collections import OrderedDict
from functools import partial
import os.path as osp
import pickle
import shutil
import warnings

import torch
import torch.nn as nn

from .logging_config import logger
from .tools import mkdir_if_missing

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "resume_from_checkpoint",
    "open_all_layers",
    "open_specified_layers",
    "count_num_param",
    "load_pretrained_weights",
]


def _torch_load_compat(fpath, map_location, weights_only, pickle_module=None):
    """Call torch.load with weights_only when supported."""
    kwargs = {"map_location": map_location}
    if pickle_module is not None:
        kwargs["pickle_module"] = pickle_module

    try:
        return torch.load(fpath, weights_only=weights_only, **kwargs)
    except TypeError as error:
        error_msg = str(error).lower()
        unsupported_weights_only = "keyword argument" in error_msg and "weights_only" in error_msg
        if unsupported_weights_only:
            if weights_only:
                raise RuntimeError(
                    "Installed PyTorch does not support safe weights_only checkpoint loading. "
                    "Upgrade PyTorch or retry with load_checkpoint(..., safe=False) only for trusted files."
                ) from error
            return torch.load(fpath, **kwargs)
        raise


def _is_safe_load_rejection(error):
    """Best-effort match for weights_only safety rejections."""
    msg = str(error).lower()
    return (
        "weights only load failed" in msg or "unsupported global" in msg or ("weights_only" in msg and "unsafe" in msg)
    )


def save_checkpoint(state, save_dir, is_best=False, remove_module_from_keys=False):
    r"""Saves checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is False.

    Examples::
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'rank1': 0.5,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)
    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict
    # save
    epoch = state["epoch"]
    fpath = osp.join(save_dir, "model.pth.tar-" + str(epoch))
    torch.save(state, fpath)
    logger.info('Checkpoint saved to "%s"', fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), "model-best.pth.tar"))


def load_checkpoint(fpath, safe=True):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.
        safe (bool, optional): if True, use ``torch.load(..., weights_only=True)``
            to avoid pickle code execution from untrusted checkpoints. Set to
            False only for trusted legacy checkpoints that require full pickle
            deserialization.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError(f'File is not found at "{fpath}"')
    map_location = None if torch.cuda.is_available() else "cpu"
    safe_load_error = (
        "Safe checkpoint loading blocked pickle deserialization. "
        "Retry with load_checkpoint(..., safe=False) only for trusted files."
    )
    try:
        checkpoint = _torch_load_compat(fpath, map_location=map_location, weights_only=safe)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        try:
            checkpoint = _torch_load_compat(
                fpath,
                map_location=map_location,
                weights_only=safe,
                pickle_module=pickle,
            )
        except Exception as error:
            if safe and _is_safe_load_rejection(error):
                raise RuntimeError(safe_load_error) from error
            logger.warning('Unable to load checkpoint from "%s"', fpath)
            raise
    except Exception as error:
        if safe and _is_safe_load_rejection(error):
            raise RuntimeError(safe_load_error) from error
        logger.warning('Unable to load checkpoint from "%s"', fpath)
        raise
    return checkpoint


def resume_from_checkpoint(fpath, model, optimizer=None, scheduler=None, safe=True):
    r"""Resumes training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fpath (str): path to checkpoint.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (LRScheduler, optional): an LRScheduler.
        safe (bool, optional): passed to :func:`load_checkpoint`.

    Returns:
        int: start_epoch.

    Examples::
        >>> from torchreid.utils import resume_from_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> start_epoch = resume_from_checkpoint(
        >>>     fpath, model, optimizer, scheduler
        >>> )
    """
    logger.info('Loading checkpoint from "%s"', fpath)
    checkpoint = load_checkpoint(fpath, safe=safe)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info("Loaded model weights")
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Loaded optimizer")
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("Loaded scheduler")
    start_epoch = checkpoint["epoch"]
    logger.info("Last epoch = %s", start_epoch)
    if "rank1" in checkpoint:
        logger.info("Last rank1 = %.1f%%", checkpoint["rank1"] * 100)
    return start_epoch


def adjust_learning_rate(
    optimizer, base_lr, epoch, stepsize=20, gamma=0.1, linear_decay=False, final_lr=0, max_epoch=100
):
    r"""Adjusts learning rate.

    Deprecated.
    """
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done * final_lr + (1.0 - frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma ** (epoch // stepsize))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_bn_to_eval(m):
    r"""Sets BatchNorm layers to eval mode."""
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def open_all_layers(model):
    r"""Opens all layers in model for training.

    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.

    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        if not hasattr(model, layer):
            raise ValueError(f'"{layer}" is not an attribute of the model, please provide the correct name')

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    r"""Counts number of parameters in a model while ignoring ``self.classifier``.

    Args:
        model (nn.Module): network model.

    Examples::
        >>> from torchreid.utils import count_num_param
        >>> model_size = count_num_param(model)

    .. warning::

        This method is deprecated in favor of
        ``torchreid.utils.compute_model_complexity``.
    """
    warnings.warn("This method is deprecated and will be removed in the future.", stacklevel=2)

    num_param = sum(p.numel() for p in model.parameters())

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters())

    return num_param


def load_pretrained_weights(model, weight_path, safe=True):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.
        safe (bool, optional): passed to :func:`load_checkpoint`.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path, safe=safe)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f'The pretrained weights "{weight_path}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)",
            stacklevel=2,
        )
    else:
        logger.info('Successfully loaded pretrained weights from "%s"', weight_path)
        if len(discarded_layers) > 0:
            logger.info(
                "** The following layers are discarded due to unmatched keys or layer size: %s",
                discarded_layers,
            )
