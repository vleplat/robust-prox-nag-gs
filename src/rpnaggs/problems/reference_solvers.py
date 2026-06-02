from __future__ import annotations

from typing import Dict

import torch

from rpnaggs.optim.prox import make_prox_operator


@torch.no_grad()
def fista_lasso_reference(
    A: torch.Tensor,
    b: torch.Tensor,
    mu_reg: float,
    lam: float,
    max_iters: int = 5000,
    tol: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    n, d = A.shape
    hessian = (A.T @ A) / n + mu_reg * torch.eye(d, device=A.device, dtype=A.dtype)
    lipschitz = float(torch.linalg.eigvalsh(hessian).max().item())
    step_size = 1.0 / max(lipschitz, 1e-12)
    prox = make_prox_operator("l1", lam=lam)

    x = torch.zeros(d, device=A.device, dtype=A.dtype)
    y = x.clone()
    t = 1.0

    def smooth_grad(vec: torch.Tensor) -> torch.Tensor:
        return (A.T @ (A @ vec - b)) / n + mu_reg * vec

    for _ in range(max_iters):
        x_prev = x.clone()
        grad = smooth_grad(y)
        x = prox.prox([y - step_size * grad], step_size)[0]
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = x + ((t - 1.0) / t_next) * (x - x_prev)
        if torch.linalg.norm(x - x_prev).item() <= tol * max(1.0, torch.linalg.norm(x_prev).item()):
            t = t_next
            break
        t = t_next

    objective = 0.5 * torch.sum((A @ x - b) ** 2).item() / n
    objective += 0.5 * mu_reg * torch.sum(x * x).item()
    objective += lam * torch.sum(torch.abs(x)).item()
    return {
        "x_star": x,
        "F_star": torch.tensor(objective, device=A.device, dtype=A.dtype),
        "L": torch.tensor(lipschitz, device=A.device, dtype=A.dtype),
    }
