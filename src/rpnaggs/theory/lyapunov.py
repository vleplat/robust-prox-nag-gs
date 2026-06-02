from __future__ import annotations

from typing import Dict


def compute_lyapunov_coefficients(a: float, mu_hat: float, mu_f: float, mu_F: float) -> Dict[str, float]:
    b = mu_hat / (2.0 * a)
    beta = (mu_hat - mu_f) / 2.0
    s = (mu_F + mu_f) / 4.0
    c_lower = beta * (1.0 - a) / a
    c_upper = (mu_hat + mu_F - 2.0 * s) / (2.0 * a) - beta
    valid = c_lower < c_upper
    c = 0.5 * (c_lower + c_upper) if valid else float("nan")
    return {
        "b": b,
        "beta": beta,
        "s": s,
        "c_lower": c_lower,
        "c_upper": c_upper,
        "c": c,
        "valid": valid,
    }


def compute_lyapunov_value(G_k: float, V_k: float, X_k: float, coeffs: Dict[str, float]) -> float:
    if not coeffs["valid"]:
        return float("nan")
    return G_k + (coeffs["b"] - coeffs["s"]) * V_k + coeffs["c"] * X_k
