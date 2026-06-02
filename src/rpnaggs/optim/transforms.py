from __future__ import annotations

import torch
from typing import Optional


def get_trainable_params(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten_grads_from_params(params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros_like(p, memory_format=torch.preserve_format).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def robust_transform_vector(
    g: torch.Tensor,
    mode: str = "identity",
    threshold: Optional[float] = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    if mode == "identity" or threshold is None:
        return g
    if mode == "coord_clip":
        return torch.clamp(g, min=-threshold, max=threshold)
    if mode == "norm_clip":
        nrm = torch.linalg.norm(g)
        scale = torch.clamp(torch.tensor(threshold, device=g.device) / (nrm + eps), max=1.0)
        return g * scale
    if mode == "tanh":
        return threshold * torch.tanh(g / threshold)
    raise ValueError(f"Unknown robust map: {mode}")


def transform_grad_list(
    params: list[torch.nn.Parameter],
    mode: str = "identity",
    threshold: Optional[float] = 1.0,
    eps: float = 1e-12,
) -> list[torch.Tensor]:
    if mode == "identity" or threshold is None:
        return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    if mode == "coord_clip":
        return [
            torch.clamp(p.grad.detach(), -threshold, threshold) if p.grad is not None else torch.zeros_like(p)
            for p in params
        ]

    if mode == "tanh":
        return [
            threshold * torch.tanh(p.grad.detach() / threshold) if p.grad is not None else torch.zeros_like(p)
            for p in params
        ]

    if mode == "norm_clip":
        total_sq = torch.zeros([], device=params[0].device)
        for p in params:
            if p.grad is not None:
                total_sq = total_sq + torch.sum(p.grad.detach() ** 2)
        total_norm = torch.sqrt(total_sq)
        scale = torch.clamp(torch.tensor(threshold, device=params[0].device) / (total_norm + eps), max=1.0)
        return [scale * p.grad.detach() if p.grad is not None else torch.zeros_like(p) for p in params]

    raise ValueError(f"Unknown robust map: {mode}")


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)
