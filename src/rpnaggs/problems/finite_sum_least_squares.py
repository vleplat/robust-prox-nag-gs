from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FiniteSumLeastSquaresProblem:
    A: torch.Tensor
    b: torch.Tensor
    mu_reg: float
    x_true: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.A = self.A.detach()
        self.b = self.b.detach()
        self.mu_reg = float(self.mu_reg)
        self.n, self.d = self.A.shape
        self._hessian = (self.A.T @ self.A) / self.n + self.mu_reg * torch.eye(
            self.d, device=self.A.device, dtype=self.A.dtype
        )
        eigvals = torch.linalg.eigvalsh(self._hessian)
        self.mu_f = float(eigvals.min().item())
        self.L = float(eigvals.max().item())
        self.mu_F = self.mu_f
        rhs = (self.A.T @ self.b) / self.n
        self.x_star = torch.linalg.solve(self._hessian, rhs)
        self.F_star = float(self.objective(self.x_star).item())

    def sample_batch_indices(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        batch_size = min(batch_size, self.n)
        return torch.randperm(self.n, generator=generator, device=self.A.device)[:batch_size]

    def smooth_objective(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.A @ x - self.b
        return 0.5 * torch.sum(residual * residual) / self.n + 0.5 * self.mu_reg * torch.sum(x * x)

    def nonsmooth_penalty(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros([], device=x.device, dtype=x.dtype)

    def objective(self, x: torch.Tensor) -> torch.Tensor:
        return self.smooth_objective(x) + self.nonsmooth_penalty(x)

    def smooth_full_grad(self, x: torch.Tensor) -> torch.Tensor:
        return (self.A.T @ (self.A @ x - self.b)) / self.n + self.mu_reg * x

    def smooth_batch_grad(self, x: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        A_b = self.A[batch_indices]
        b_b = self.b[batch_indices]
        batch_size = max(int(batch_indices.numel()), 1)
        return (A_b.T @ (A_b @ x - b_b)) / batch_size + self.mu_reg * x
