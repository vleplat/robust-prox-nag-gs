from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from rpnaggs.algorithms.base import TrainingAlgorithm
from rpnaggs.algorithms.baselines import accuracy_from_logits
from rpnaggs.optim.prox import ProxOperator, make_prox_operator
from rpnaggs.optim.transforms import flatten_grads_from_params, get_trainable_params, transform_grad_list


def _find_target_param_indices(model: nn.Module, params: List[torch.nn.Parameter], prox_target: str) -> list[int]:
    prox_target = prox_target.lower()
    if prox_target == "all":
        return list(range(len(params)))
    if prox_target == "weights_only":
        return [idx for idx, param in enumerate(params) if param.ndim >= 2]
    if prox_target == "classifier":
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            return []
        target_ids = {
            id(param)
            for param in last_linear.parameters(recurse=False)
            if param.requires_grad
        }
        return [idx for idx, param in enumerate(params) if id(param) in target_ids]
    if prox_target == "conv":
        target_ids = set()
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                for param in module.parameters(recurse=False):
                    if param.requires_grad and param.ndim >= 2:
                        target_ids.add(id(param))
        return [idx for idx, param in enumerate(params) if id(param) in target_ids]
    raise ValueError(f"Unknown prox target: {prox_target}")


class RobustProxNAGGSAlgorithm(TrainingAlgorithm):
    """Robust Prox-NAG-GS algorithm following the x/v coupled update."""

    def __init__(
        self,
        model: nn.Module,
        a: float = 0.2,
        mu_hat: Optional[float] = None,
        eta: Optional[float] = None,
        robust_map: str = "norm_clip",
        threshold: Optional[float] = 1.0,
        prox_operator: Optional[ProxOperator] = None,
        prox_target: str = "all",
        use_warmup: bool = False,
        warmup_fraction: float = 0.05,
        warmup_steps: Optional[int] = None,
        total_training_steps: Optional[int] = None,
    ):
        self.params = get_trainable_params(model)
        self.prox_target = prox_target
        self.prox_target_indices = _find_target_param_indices(model, self.params, prox_target)
        self.a = float(a)
        if mu_hat is None:
            if eta is None:
                eta = 5e-3
            mu_hat = self.a / float(eta)
        self.mu_hat = float(mu_hat)
        self.step_size = self.a / self.mu_hat
        self.base_step_size = self.step_size
        self.robust_map = robust_map
        self.threshold = float(threshold) if threshold is not None else None
        self.prox_operator = prox_operator if prox_operator is not None else make_prox_operator("none")
        self.prox_name = self.prox_operator.name
        self.use_warmup = bool(use_warmup)
        self.warmup_fraction = float(warmup_fraction)
        self.total_training_steps = total_training_steps
        if self.use_warmup:
            if warmup_steps is not None:
                self.warmup_steps = max(int(warmup_steps), 1)
            elif total_training_steps is not None:
                self.warmup_steps = max(int(total_training_steps * self.warmup_fraction), 1)
            else:
                self.warmup_steps = 1
        else:
            self.warmup_steps = 0
        self.global_step = 0
        self.v_buffers = [p.detach().clone() for p in self.params]
        self.last_step_stats = {
            "gradient_norm": 0.0,
            "transformed_gradient_norm": 0.0,
            "clipping_ratio": 0.0,
            "coordinate_clipping_ratio": float("nan"),
            "x_v_distance": 0.0,
            "step_size": self.base_step_size,
            "base_step_size": self.base_step_size,
            "warmup_factor": 1.0,
            "mu_hat": self.mu_hat,
            "a": self.a,
            "robust_map": self.robust_map,
            "threshold": self.threshold,
            "prox_name": self.prox_name,
            "prox_target": self.prox_target,
            "data_loss": 0.0,
            "regularization_penalty": 0.0,
            "total_objective": 0.0,
            "nonfinite": False,
        }

    @property
    def name(self) -> str:
        return "robust_proxnaggs"

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def _set_x_next(self) -> None:
        for p, v in zip(self.params, self.v_buffers):
            p.copy_((1.0 - self.a) * p.detach() + self.a * v)

    def _current_step_size(self) -> tuple[float, float]:
        if not self.use_warmup or self.warmup_steps <= 0:
            return self.base_step_size, 1.0
        self.global_step += 1
        rho = min(1.0, self.global_step / self.warmup_steps)
        return rho * self.base_step_size, rho

    @torch.no_grad()
    def _update_v(self) -> None:
        raw_grad = flatten_grads_from_params(self.params)
        d_list = transform_grad_list(self.params, self.robust_map, self.threshold)
        transformed_grad = torch.cat([d.reshape(-1) for d in d_list])
        raw_grad_norm = torch.linalg.norm(raw_grad).item()
        transformed_grad_norm = torch.linalg.norm(transformed_grad).item()
        step_size, warmup_factor = self._current_step_size()
        clipping_ratio = 0.0
        coordinate_clipping_ratio = float("nan")
        if raw_grad_norm > 0:
            clipping_ratio = (
                torch.linalg.norm(raw_grad - transformed_grad).item() / (raw_grad_norm + 1e-12)
            )
        if self.threshold is not None and self.robust_map in {"coord_clip", "tanh"}:
            coordinate_clipping_ratio = float((torch.abs(raw_grad) > self.threshold).float().mean().item())
        candidates = []
        for p, v, d in zip(self.params, self.v_buffers, d_list):
            z = (1.0 - self.a) * v + self.a * p.detach()
            u = z - step_size * d
            candidates.append(u)
        selected_candidates = [candidates[idx] for idx in self.prox_target_indices]
        prox_selected = self.prox_operator.prox(selected_candidates, step_size)
        for idx, u in zip(self.prox_target_indices, prox_selected):
            candidates[idx] = u
        for v, u in zip(self.v_buffers, candidates):
            v.copy_(u)
        x_v_sq = torch.zeros([], device=self.params[0].device)
        for p, v in zip(self.params, self.v_buffers):
            x_v_sq = x_v_sq + torch.sum((p.detach() - v) ** 2)
        self.last_step_stats = {
            "gradient_norm": raw_grad_norm,
            "transformed_gradient_norm": transformed_grad_norm,
            "clipping_ratio": clipping_ratio,
            "coordinate_clipping_ratio": coordinate_clipping_ratio,
            "x_v_distance": torch.sqrt(x_v_sq).item(),
            "step_size": step_size,
            "base_step_size": self.base_step_size,
            "warmup_factor": warmup_factor,
            "mu_hat": self.mu_hat,
            "a": self.a,
            "robust_map": self.robust_map,
            "threshold": self.threshold,
            "prox_name": self.prox_name,
            "prox_target": self.prox_target,
            "data_loss": self.last_step_stats.get("data_loss", 0.0),
            "regularization_penalty": self.last_step_stats.get("regularization_penalty", 0.0),
            "total_objective": self.last_step_stats.get("total_objective", 0.0),
            "nonfinite": bool(
                not torch.isfinite(raw_grad).all().item()
                or not torch.isfinite(transformed_grad).all().item()
            ),
        }

    def train_batch(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion,
    ) -> tuple[float, float]:
        self.zero_grad()
        self._set_x_next()
        logits = model(x)
        data_loss = criterion(logits, y)
        targeted_params = [self.params[idx].detach() for idx in self.prox_target_indices]
        regularization_penalty = self.prox_operator.penalty(targeted_params)
        total_objective = data_loss.item() + regularization_penalty
        data_loss.backward()
        self.last_step_stats.update(
            {
                "data_loss": data_loss.item(),
                "regularization_penalty": regularization_penalty,
                "total_objective": total_objective,
                "prox_target": self.prox_target,
            }
        )
        self._update_v()
        self.last_step_stats["nonfinite"] = bool(
            self.last_step_stats["nonfinite"] or not torch.isfinite(data_loss.detach()).all().item()
        )
        with torch.no_grad():
            acc = accuracy_from_logits(model(x), y)
        return data_loss.item(), acc
