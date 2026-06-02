from __future__ import annotations

import torch.nn as nn

from rpnaggs.config import ExperimentConfig
from rpnaggs.models.small_cnn import SmallCIFARCNN


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.dataset.lower() != "cifar10":
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    if config.model_name.lower() != "small_cifar_cnn":
        raise ValueError(f"Unsupported model: {config.model_name}")
    return SmallCIFARCNN()
