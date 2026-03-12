"""Abstract base class for augmentation backends."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class AugmentationBackend(ABC):
    """Abstract base class for augmentation backends.

    Backends build train and test transform pipelines from a unified config.
    """

    @abstractmethod
    def build_train_transforms(self, cfg: Any) -> Callable:
        """Build training augmentation pipeline.

        Args:
            cfg: Configuration with aug.* and data.* settings.

        Returns:
            Callable transform (PIL/tensor in -> tensor out).
        """
        pass

    @abstractmethod
    def build_test_transforms(self, cfg: Any) -> Callable:
        """Build test/validation augmentation pipeline.

        Args:
            cfg: Configuration with aug.* and data.* settings.

        Returns:
            Callable transform (PIL/tensor in -> tensor out).
        """
        pass
