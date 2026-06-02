from __future__ import annotations

import torch

from rpnaggs.problems.finite_sum_least_squares import FiniteSumLeastSquaresProblem
from rpnaggs.problems.reference_solvers import fista_lasso_reference


class CompositeLassoProblem(FiniteSumLeastSquaresProblem):
    def __init__(self, A: torch.Tensor, b: torch.Tensor, mu_reg: float, lam: float, x_true=None):
        self.lam = float(lam)
        super().__init__(A=A, b=b, mu_reg=mu_reg, x_true=x_true)
        self._refresh_reference()

    def nonsmooth_penalty(self, x: torch.Tensor) -> torch.Tensor:
        return self.lam * torch.sum(torch.abs(x))

    def _refresh_reference(self) -> None:
        reference = fista_lasso_reference(self.A, self.b, self.mu_reg, self.lam)
        self.x_star = reference["x_star"]
        self.F_star = float(reference["F_star"].item())
        self.mu_F = self.mu_f
