from __future__ import annotations

from typing import Iterable, List

import torch

from rpnaggs.optim.transforms import soft_threshold


def _group_norms(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return torch.abs(tensor)
    return torch.linalg.vector_norm(tensor.reshape(tensor.shape[0], -1), ord=2, dim=1)


def _group_shrink(tensor: torch.Tensor, tau: float, eps: float = 1e-12) -> torch.Tensor:
    if tau <= 0:
        return tensor
    norms = _group_norms(tensor)
    scale = torch.clamp(1.0 - tau / (norms + eps), min=0.0)
    if tensor.ndim <= 1:
        return tensor * scale
    shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
    return tensor * scale.reshape(shape)


def _group_penalty(tensor: torch.Tensor) -> torch.Tensor:
    return _group_norms(tensor).sum()


class ProxOperator:
    name = "none"

    @torch.no_grad()
    def prox(self, tensors: List[torch.Tensor], step_size: float) -> List[torch.Tensor]:
        return tensors

    def penalty(self, tensors: Iterable[torch.Tensor]) -> float:
        return 0.0


class NoProx(ProxOperator):
    name = "none"


class L1Prox(ProxOperator):
    name = "l1"

    def __init__(self, lam: float):
        self.lam = float(lam)

    @torch.no_grad()
    def prox(self, tensors: List[torch.Tensor], step_size: float) -> List[torch.Tensor]:
        tau = step_size * self.lam
        return [soft_threshold(tensor, tau) for tensor in tensors]

    def penalty(self, tensors: Iterable[torch.Tensor]) -> float:
        total = 0.0
        for tensor in tensors:
            total += self.lam * torch.sum(torch.abs(tensor)).item()
        return total


class L2Prox(ProxOperator):
    name = "l2"

    def __init__(self, lam: float):
        self.lam = float(lam)

    @torch.no_grad()
    def prox(self, tensors: List[torch.Tensor], step_size: float) -> List[torch.Tensor]:
        scale = 1.0 / (1.0 + step_size * self.lam)
        return [scale * tensor for tensor in tensors]

    def penalty(self, tensors: Iterable[torch.Tensor]) -> float:
        total = 0.0
        for tensor in tensors:
            total += 0.5 * self.lam * torch.sum(tensor * tensor).item()
        return total


class ElasticNetProx(ProxOperator):
    name = "elastic_net"

    def __init__(self, l1: float, l2: float):
        self.l1 = float(l1)
        self.l2 = float(l2)

    @torch.no_grad()
    def prox(self, tensors: List[torch.Tensor], step_size: float) -> List[torch.Tensor]:
        scale = 1.0 / (1.0 + step_size * self.l2)
        tau = step_size * self.l1
        return [scale * soft_threshold(tensor, tau) for tensor in tensors]

    def penalty(self, tensors: Iterable[torch.Tensor]) -> float:
        total = 0.0
        for tensor in tensors:
            total += self.l1 * torch.sum(torch.abs(tensor)).item()
            total += 0.5 * self.l2 * torch.sum(tensor * tensor).item()
        return total


class GroupLassoProx(ProxOperator):
    name = "group_lasso"

    def __init__(self, group: float):
        self.group = float(group)

    @torch.no_grad()
    def prox(self, tensors: List[torch.Tensor], step_size: float) -> List[torch.Tensor]:
        tau = step_size * self.group
        return [_group_shrink(tensor, tau) for tensor in tensors]

    def penalty(self, tensors: Iterable[torch.Tensor]) -> float:
        total = 0.0
        for tensor in tensors:
            total += self.group * _group_penalty(tensor).item()
        return total


class SparseGroupLassoProx(ProxOperator):
    name = "sparse_group_lasso"

    def __init__(self, l1: float, group: float):
        self.l1 = float(l1)
        self.group = float(group)

    @torch.no_grad()
    def prox(self, tensors: List[torch.Tensor], step_size: float) -> List[torch.Tensor]:
        tau_l1 = step_size * self.l1
        tau_group = step_size * self.group
        return [_group_shrink(soft_threshold(tensor, tau_l1), tau_group) for tensor in tensors]

    def penalty(self, tensors: Iterable[torch.Tensor]) -> float:
        total = 0.0
        for tensor in tensors:
            total += self.l1 * torch.sum(torch.abs(tensor)).item()
            total += self.group * _group_penalty(tensor).item()
        return total


class NonnegativeProx(ProxOperator):
    name = "nonnegative"

    @torch.no_grad()
    def prox(self, tensors: List[torch.Tensor], step_size: float) -> List[torch.Tensor]:
        del step_size
        return [torch.clamp(tensor, min=0.0) for tensor in tensors]

    def penalty(self, tensors: Iterable[torch.Tensor]) -> float:
        for tensor in tensors:
            if torch.any(tensor < 0):
                return float("inf")
        return 0.0


class BoxProx(ProxOperator):
    name = "box"

    def __init__(self, lower: float, upper: float):
        if lower > upper:
            raise ValueError("BoxProx requires lower <= upper.")
        self.lower = float(lower)
        self.upper = float(upper)

    @torch.no_grad()
    def prox(self, tensors: List[torch.Tensor], step_size: float) -> List[torch.Tensor]:
        del step_size
        return [torch.clamp(tensor, min=self.lower, max=self.upper) for tensor in tensors]

    def penalty(self, tensors: Iterable[torch.Tensor]) -> float:
        for tensor in tensors:
            if torch.any(tensor < self.lower) or torch.any(tensor > self.upper):
                return float("inf")
        return 0.0


def make_prox_operator(
    name: str = "none",
    lam: float = 0.0,
    l1: float = 0.0,
    l2: float = 0.0,
    group: float = 0.0,
    lower: float = -1.0,
    upper: float = 1.0,
) -> ProxOperator:
    name = name.lower()
    if name == "none":
        return NoProx()
    if name == "l1":
        return L1Prox(lam=lam)
    if name == "l2":
        return L2Prox(lam=lam)
    if name == "elastic_net":
        return ElasticNetProx(l1=l1, l2=l2)
    if name == "group_lasso":
        return GroupLassoProx(group=group)
    if name == "sparse_group_lasso":
        return SparseGroupLassoProx(l1=l1, group=group)
    if name == "nonnegative":
        return NonnegativeProx()
    if name == "box":
        return BoxProx(lower=lower, upper=upper)
    raise ValueError(f"Unknown prox operator: {name}")
