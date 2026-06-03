from __future__ import annotations

from typing import Callable

import torch.nn as nn

from rpnaggs.config import ExperimentConfig
from rpnaggs.models.small_cnn import SmallCIFARCNN
from rpnaggs.models.vgg7_mini import VGG7MiniMNIST

ModelBuilder = Callable[[ExperimentConfig], nn.Module]

_MODEL_BUILDERS: dict[tuple[str, str], ModelBuilder] = {
    ("cifar10", "small_cifar_cnn"): lambda _config: SmallCIFARCNN(),
    ("mnist", "vgg7_mini_mnist"): lambda _config: VGG7MiniMNIST(),
}


def available_model_pairs() -> list[tuple[str, str]]:
    return sorted(_MODEL_BUILDERS.keys())


def build_model(config: ExperimentConfig) -> nn.Module:
    key = (config.dataset.lower(), config.model_name.lower())
    builder = _MODEL_BUILDERS.get(key)
    if builder is None:
        available = ", ".join(f"{dataset}/{model}" for dataset, model in available_model_pairs())
        raise ValueError(
            f"Unsupported model '{config.model_name}' for dataset '{config.dataset}'. "
            f"Available pairs: {available}"
        )
    return builder(config)
