from collections.abc import Sequence

import torch

AVAI_SCH = ["single_step", "multi_step", "cosine"]


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: str = "single_step",
    stepsize: int | Sequence[int] = 1,
    gamma: float = 0.1,
    max_epoch: int = 1,
) -> torch.optim.lr_scheduler._LRScheduler:
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is single_step.
        stepsize (int or list, optional): step size to decay learning rate. When ``lr_scheduler``
            is "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.

    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = pi_torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = pi_torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError(f"Unsupported scheduler: {lr_scheduler}. Must be one of {AVAI_SCH}")

    if lr_scheduler == "single_step":
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(f"For single_step lr_scheduler, stepsize must be an integer, but got {type(stepsize)}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, list):
            raise TypeError(f"For multi_step lr_scheduler, stepsize must be a list, but got {type(stepsize)}")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(max_epoch))

    return scheduler
