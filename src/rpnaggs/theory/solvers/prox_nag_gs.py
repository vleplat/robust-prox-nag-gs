from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rpnaggs.optim.prox import ProxOperator, make_prox_operator
from rpnaggs.optim.transforms import robust_transform_vector


@dataclass
class VectorProxNAGGS:
    dim: int
    a: float
    mu_hat: float
    robust_map: str = "identity"
    threshold: Optional[float] = None
    prox_operator: Optional[ProxOperator] = None
    name: str = "proxnaggs"

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
        return self.a / self.mu_hat

    def prediction_point(self) -> torch.Tensor:
        return (1.0 - self.a) * self.x + self.a * self.v

    def step(self, batch_grad: torch.Tensor) -> dict:
        x_next = self.prediction_point()
        z_next = (1.0 - self.a) * self.v + self.a * x_next
        transformed = robust_transform_vector(batch_grad, mode=self.robust_map, threshold=self.threshold)
        candidate = z_next - self.step_size * transformed
        v_next = self.prox_operator.prox([candidate], self.step_size)[0]
        self.x = x_next
        self.v = v_next
        return {
            "batch_grad": batch_grad.detach().clone(),
            "transformed_grad": transformed.detach().clone(),
            "x_state": self.x.detach().clone(),
            "v_state": self.v.detach().clone(),
            "step_size": self.step_size,
            "a": self.a,
            "mu_hat": self.mu_hat,
            "x_v_distance": float(torch.linalg.norm(self.x - self.v).item()),
            "prox_name": self.prox_operator.name,
            "threshold": self.threshold if self.threshold is not None else float("nan"),
        }
