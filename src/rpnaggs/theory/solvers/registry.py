from __future__ import annotations

from typing import Optional

from rpnaggs.optim.prox import make_prox_operator
from rpnaggs.theory.solvers.prox_nag_gs import VectorProxNAGGS
from rpnaggs.theory.solvers.sgd import VectorSGD


def build_theory_solver(
    method: str,
    dim: int,
    lr: float = 0.05,
    a: float = 0.4,
    mu_hat: float = 1.0,
    robust_map: str = "identity",
    threshold: Optional[float] = None,
    prox_name: str = "none",
    prox_lam: float = 0.0,
):
    prox_operator = make_prox_operator(prox_name, lam=prox_lam)
    if method == "sgd":
        return VectorSGD(dim=dim, lr=lr, robust_map="identity", threshold=None, name="sgd")
    if method == "clipped_sgd":
        return VectorSGD(dim=dim, lr=lr, robust_map=robust_map, threshold=threshold, name=f"clipped_sgd_{robust_map}")
    if method == "prox_sgd":
        return VectorSGD(
            dim=dim,
            lr=lr,
            robust_map="identity",
            threshold=None,
            prox_operator=prox_operator,
            name="prox_sgd",
        )
    if method == "clipped_prox_sgd":
        return VectorSGD(
            dim=dim,
            lr=lr,
            robust_map=robust_map,
            threshold=threshold,
            prox_operator=prox_operator,
            name=f"clipped_prox_sgd_{robust_map}",
        )
    if method == "proxnaggs":
        return VectorProxNAGGS(
            dim=dim,
            a=a,
            mu_hat=mu_hat,
            robust_map=robust_map,
            threshold=threshold,
            prox_operator=prox_operator,
            name=f"proxnaggs_{robust_map}",
        )
    raise ValueError(f"Unsupported theory method: {method}")
