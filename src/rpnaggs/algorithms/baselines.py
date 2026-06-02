from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from rpnaggs.algorithms.base import TrainingAlgorithm
from rpnaggs.optim.transforms import flatten_grads_from_params, get_trainable_params


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


class OptimizerAlgorithm(TrainingAlgorithm):
    def __init__(
        self,
        name: str,
        optimizer: torch.optim.Optimizer,
        grad_clip_mode: Optional[str] = None,
        grad_clip_threshold: Optional[float] = None,
    ):
        self._name = name
        self.optimizer = optimizer
        self.grad_clip_mode = grad_clip_mode
        self.grad_clip_threshold = grad_clip_threshold
        self.last_step_stats = {
            "gradient_norm": 0.0,
            "transformed_gradient_norm": 0.0,
            "clipping_ratio": 0.0,
            "coordinate_clipping_ratio": float("nan"),
            "x_v_distance": float("nan"),
            "step_size": float("nan"),
            "base_step_size": float("nan"),
            "warmup_factor": float("nan"),
            "mu_hat": float("nan"),
            "nonfinite": False,
        }

    @property
    def name(self) -> str:
        return self._name

    def train_batch(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion,
    ) -> tuple[float, float]:
        self.optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        params = get_trainable_params(model)
        raw_grad = flatten_grads_from_params(params)
        raw_grad_norm = torch.linalg.norm(raw_grad).item()
        transformed_grad = raw_grad.clone()
        clipping_ratio = 0.0
        coordinate_clipping_ratio = float("nan")

        if self.grad_clip_mode == "norm_clip":
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_threshold)
            transformed_grad = flatten_grads_from_params(params)
        elif self.grad_clip_mode == "coord_clip":
            torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip_threshold)
            transformed_grad = flatten_grads_from_params(params)
            coordinate_clipping_ratio = float((torch.abs(raw_grad) > self.grad_clip_threshold).float().mean().item())

        transformed_grad_norm = torch.linalg.norm(transformed_grad).item()
        if raw_grad_norm > 0:
            clipping_ratio = torch.linalg.norm(raw_grad - transformed_grad).item() / (raw_grad_norm + 1e-12)

        self.optimizer.step()
        acc = accuracy_from_logits(logits.detach(), y)
        self.last_step_stats = {
            "gradient_norm": raw_grad_norm,
            "transformed_gradient_norm": transformed_grad_norm,
            "clipping_ratio": clipping_ratio,
            "coordinate_clipping_ratio": coordinate_clipping_ratio,
            "x_v_distance": float("nan"),
            "step_size": float("nan"),
            "base_step_size": float("nan"),
            "warmup_factor": float("nan"),
            "mu_hat": float("nan"),
            "nonfinite": bool(
                not torch.isfinite(raw_grad).all().item()
                or not torch.isfinite(transformed_grad).all().item()
                or not torch.isfinite(loss.detach()).all().item()
            ),
        }
        return loss.item(), acc
