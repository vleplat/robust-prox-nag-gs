from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rpnaggs.optim.prox import ProxOperator, make_prox_operator
from rpnaggs.optim.transforms import robust_transform_vector


@dataclass
class VectorSGD:
    dim: int
    lr: float
    robust_map: str = "identity"
    threshold: Optional[float] = None
    prox_operator: Optional[ProxOperator] = None
    name: str = "sgd"

    def __post_init__(self) -> None:
        self.x = torch.zeros(self.dim)
        self.v = self.x.clone()
        self.prox_operator = self.prox_operator or make_prox_operator("none")

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32):
        self.x = self.x.to(device=device, dtype=dtype)
        self.v = self.v.to(device=device, dtype=dtype)
        return self

    @property
    def step_size(self) -> float:
        return self.lr

    def step(self, batch_grad: torch.Tensor) -> dict:
        transformed = robust_transform_vector(batch_grad, mode=self.robust_map, threshold=self.threshold)
        candidate = self.x - self.lr * transformed
        updated = self.prox_operator.prox([candidate], self.lr)[0]
        self.x = updated
        self.v = updated.clone()
        return {
            "batch_grad": batch_grad.detach().clone(),
            "transformed_grad": transformed.detach().clone(),
            "x_state": self.x.detach().clone(),
            "v_state": self.v.detach().clone(),
            "step_size": self.lr,
            "a": float("nan"),
            "mu_hat": float("nan"),
            "x_v_distance": 0.0,
            "prox_name": self.prox_operator.name,
            "threshold": self.threshold if self.threshold is not None else float("nan"),
        }
