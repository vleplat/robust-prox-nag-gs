from __future__ import annotations

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from rpnaggs.config import ExperimentConfig


def add_symmetric_label_noise_cifar10(dataset, noise_rate: float, seed: int = 123):
    if noise_rate <= 0:
        return dataset

    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets, dtype=np.int64)
    noisy_targets = targets.copy()

    mask = rng.random(len(targets)) < noise_rate
    noisy_indices = np.where(mask)[0]

    for idx in noisy_indices:
        true_label = targets[idx]
        choices = list(range(10))
        choices.remove(int(true_label))
        noisy_targets[idx] = rng.choice(choices)

    dataset.targets = noisy_targets.tolist()
    print(
        f"Injected symmetric label noise: {len(noisy_indices)}/{len(targets)} labels changed "
        f"({100 * len(noisy_indices) / len(targets):.1f}%)."
    )
    return dataset


def get_cifar10_loaders(config: ExperimentConfig):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    trainset = add_symmetric_label_noise_cifar10(trainset, config.label_noise_rate, seed=config.seed)

    trainset_eval_transform = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=True,
        download=True,
        transform=transform_eval,
    )
    if config.label_noise_rate > 0:
        trainset_eval_transform.targets = list(trainset.targets)

    testset = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=False,
        download=True,
        transform=transform_eval,
    )

    if config.use_train_subset:
        rng = np.random.default_rng(config.seed)
        subset_size = min(config.train_subset_size, len(trainset))
        indices = rng.choice(len(trainset), size=subset_size, replace=False).tolist()
        trainset = Subset(trainset, indices)
        trainset_eval_transform = Subset(trainset_eval_transform, indices)
        print(f"Using train subset of size {len(trainset)}.")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    train_eval_loader = DataLoader(
        trainset_eval_transform,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, train_eval_loader, test_loader
