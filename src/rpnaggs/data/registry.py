from __future__ import annotations

from typing import Callable, Tuple

from torch.utils.data import DataLoader

from rpnaggs.config import ExperimentConfig
from rpnaggs.data.cifar10 import get_cifar10_loaders
from rpnaggs.data.mnist import get_mnist_loaders

LoaderFn = Callable[[ExperimentConfig], Tuple[DataLoader, DataLoader, DataLoader]]

_LOADER_REGISTRY: dict[str, LoaderFn] = {
    "cifar10": get_cifar10_loaders,
    "mnist": get_mnist_loaders,
}


def available_datasets() -> list[str]:
    return sorted(_LOADER_REGISTRY)


def get_data_loaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = config.dataset.lower()
    loader_fn = _LOADER_REGISTRY.get(dataset)
    if loader_fn is None:
        raise ValueError(
            f"Unsupported dataset: {config.dataset}. Available datasets: {available_datasets()}"
        )
    return loader_fn(config)
