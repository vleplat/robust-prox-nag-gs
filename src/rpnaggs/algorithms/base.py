from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TrainingAlgorithm(ABC):
    """Unified training interface for baselines and research algorithms."""

    @abstractmethod
    def train_batch(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion,
    ) -> tuple[float, float]:
        """Run one training batch and return (loss, accuracy)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()
