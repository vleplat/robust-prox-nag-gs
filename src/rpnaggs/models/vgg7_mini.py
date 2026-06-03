from __future__ import annotations

import torch
import torch.nn as nn


class VGG7MiniMNIST(nn.Module):
    """VGG-7-Mini architecture from References/11_VGG_7_Mini.ipynb on 28x28 grayscale input.

    Conv stack uses single-channel filters with ReLU and 2x2 average pooling.
    Two parallel branches after the first pool are concatenated (98 features) before
    tanh MLP heads. The notebook's arctan/softmax output is replaced by a linear
    classifier for use with CrossEntropyLoss in the training pipeline.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.branch_a = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(98, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.stem(x)
        branch_a = self.branch_a(shared)
        branch_b = self.branch_b(shared)
        fused = torch.cat([branch_a, branch_b], dim=1)
        return self.classifier(fused)
