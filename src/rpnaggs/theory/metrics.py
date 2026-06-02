from __future__ import annotations

from typing import Dict, Optional

import torch

from rpnaggs.optim.transforms import robust_transform_vector
from rpnaggs.theory.lyapunov import compute_lyapunov_value


def compute_gradient_metrics(
    batch_grad: torch.Tensor,
    full_grad: torch.Tensor,
    transformed_grad: torch.Tensor,
    robust_map: str,
    threshold: Optional[float],
) -> Dict[str, float]:
    raw_error = batch_grad - full_grad
    transformed_error = transformed_grad - full_grad
    full_grad_transformed = robust_transform_vector(full_grad, mode=robust_map, threshold=threshold)
    batch_grad_norm = torch.linalg.norm(batch_grad).item()
    full_grad_norm = torch.linalg.norm(full_grad).item()
    transformed_norm = torch.linalg.norm(transformed_grad).item()
    raw_error_norm = torch.linalg.norm(raw_error).item()
    transformed_error_norm = torch.linalg.norm(transformed_error).item()
    transform_displacement_norm = torch.linalg.norm(batch_grad - transformed_grad).item()
    deterministic_bias_norm = torch.linalg.norm(full_grad_transformed - full_grad).item()
    clipping_ratio = 0.0
    if batch_grad_norm > 0:
        clipping_ratio = transform_displacement_norm / (batch_grad_norm + 1e-12)
    coordinate_activation_ratio_batch = float("nan")
    coordinate_activation_ratio_full = float("nan")
    true_coord_bias_norm = float("nan")
    true_norm_bias = float("nan")
    if threshold is not None and robust_map in {"coord_clip", "tanh"}:
        coordinate_activation_ratio_batch = float((torch.abs(batch_grad) > threshold).float().mean().item())
        coordinate_activation_ratio_full = float((torch.abs(full_grad) > threshold).float().mean().item())
        true_coord_bias_norm = float(torch.linalg.norm(torch.clamp(torch.abs(full_grad) - threshold, min=0.0)).item())
    if threshold is not None and robust_map == "norm_clip":
        true_norm_bias = max(full_grad_norm - threshold, 0.0)
    return {
        "raw_error_norm": raw_error_norm,
        "raw_error_sq": raw_error_norm ** 2,
        "transformed_error_norm": transformed_error_norm,
        "transformed_error_sq": transformed_error_norm ** 2,
        "batch_grad_norm": batch_grad_norm,
        "full_grad_norm": full_grad_norm,
        "transformed_grad_norm": transformed_norm,
        "transform_displacement_norm": transform_displacement_norm,
        "deterministic_bias_norm": deterministic_bias_norm,
        "clipping_ratio": clipping_ratio,
        "coordinate_activation_ratio_batch": coordinate_activation_ratio_batch,
        "coordinate_activation_ratio_full": coordinate_activation_ratio_full,
        "true_coord_bias_norm": true_coord_bias_norm,
        "true_norm_bias": true_norm_bias,
    }


def compute_state_metrics(
    problem,
    x_state: torch.Tensor,
    v_state: torch.Tensor,
    lyapunov_coeffs: Dict[str, float],
) -> Dict[str, float]:
    G_k = float((problem.objective(v_state) - problem.F_star).item())
    V_k = float(torch.sum((v_state - problem.x_star) ** 2).item())
    X_k = float(torch.sum((x_state - problem.x_star) ** 2).item())
    return {
        "objective_gap": G_k,
        "distance_v": float(torch.linalg.norm(v_state - problem.x_star).item()),
        "distance_x": float(torch.linalg.norm(x_state - problem.x_star).item()),
        "G_k": G_k,
        "V_k": V_k,
        "X_k": X_k,
        "lyapunov": compute_lyapunov_value(G_k, V_k, X_k, lyapunov_coeffs),
        "lyapunov_valid": bool(lyapunov_coeffs["valid"]),
    }


def compute_lasso_metrics(v_state: torch.Tensor, x_true: Optional[torch.Tensor]) -> Dict[str, float]:
    sparsity = float((torch.abs(v_state) <= 1e-8).float().mean().item())
    metrics = {"sparsity": sparsity, "support_recovery": float("nan")}
    if x_true is not None:
        recovered = torch.abs(v_state) > 1e-8
        target = torch.abs(x_true) > 1e-8
        metrics["support_recovery"] = float((recovered == target).float().mean().item())
    return metrics
