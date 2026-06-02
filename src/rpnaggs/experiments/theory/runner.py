from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import pandas as pd
import torch

from rpnaggs.theory.metrics import compute_gradient_metrics, compute_lasso_metrics, compute_state_metrics


def run_theory_experiment(
    *,
    test_id: str,
    run_id: str,
    problem,
    solver,
    batch_size: int,
    iterations: int,
    seed: int,
    log_every: int,
    lyapunov_coeffs: Dict[str, float],
    threshold_quantile: Optional[float] = None,
    mu_hat_factor: Optional[float] = None,
    phase: str = "main",
    outside_theory: bool = False,
    explosion_objective_gap: float = 1e12,
    explosion_distance_v: float = 1e6,
    explosion_grad_norm: float = 1e12,
    batch_size_schedule: Optional[Callable[[int], int]] = None,
    threshold_schedule: Optional[Callable[[int], tuple[Optional[float], Optional[float]]]] = None,
    batch_size_label=None,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    generator = torch.Generator(device=problem.A.device)
    generator.manual_seed(seed)
    rows = []
    status = "success"
    first_bad_iteration = None
    max_objective_gap = -float("inf")
    max_distance_v = -float("inf")
    max_batch_grad_norm = -float("inf")
    max_transformed_grad_norm = -float("inf")

    for iteration in range(1, iterations + 1):
        current_batch_size = int(batch_size_schedule(iteration)) if batch_size_schedule is not None else int(batch_size)
        current_threshold_quantile = threshold_quantile
        if threshold_schedule is not None:
            current_threshold, current_threshold_quantile = threshold_schedule(iteration)
            solver.threshold = current_threshold
        if hasattr(solver, "prediction_point") and callable(solver.prediction_point):
            x_eval = solver.prediction_point()
        else:
            x_eval = solver.x
        batch_indices = problem.sample_batch_indices(current_batch_size, generator)
        full_grad = problem.smooth_full_grad(x_eval)
        batch_grad = problem.smooth_batch_grad(x_eval, batch_indices)
        step_info = solver.step(batch_grad)
        grad_metrics = compute_gradient_metrics(
            batch_grad=step_info["batch_grad"],
            full_grad=full_grad,
            transformed_grad=step_info["transformed_grad"],
            robust_map=getattr(solver, "robust_map", "identity"),
            threshold=getattr(solver, "threshold", None),
        )
        state_metrics = compute_state_metrics(
            problem=problem,
            x_state=step_info["x_state"],
            v_state=step_info["v_state"],
            lyapunov_coeffs=lyapunov_coeffs,
        )
        lasso_metrics = compute_lasso_metrics(step_info["v_state"], getattr(problem, "x_true", None))
        objective = float(problem.objective(step_info["v_state"]).item())
        max_objective_gap = max(max_objective_gap, state_metrics["objective_gap"])
        max_distance_v = max(max_distance_v, state_metrics["distance_v"])
        max_batch_grad_norm = max(max_batch_grad_norm, grad_metrics["batch_grad_norm"])
        max_transformed_grad_norm = max(max_transformed_grad_norm, grad_metrics["transformed_grad_norm"])

        has_theory_mu_hat = step_info["mu_hat"] == step_info["mu_hat"]
        mu_hat_ge_L = bool(step_info["mu_hat"] >= problem.L) if has_theory_mu_hat else None
        row = {
            "test_id": test_id,
            "run_id": run_id,
            "phase": phase,
            "seed": seed,
            "iteration": iteration,
            "method": solver.name,
            "batch_size": batch_size_label if batch_size_label is not None else current_batch_size,
            "effective_batch_size": current_batch_size,
            "a": step_info["a"],
            "mu_hat": step_info["mu_hat"],
            "mu_hat_factor": mu_hat_factor,
            "h": step_info["step_size"],
            "step_size": step_info["step_size"],
            "robust_map": getattr(solver, "robust_map", "identity"),
            "threshold": step_info["threshold"],
            "prox_name": step_info["prox_name"],
            "data_loss": float(problem.smooth_objective(step_info["v_state"]).item()),
            "regularization_penalty": float(problem.nonsmooth_penalty(step_info["v_state"]).item()),
            "total_objective": objective,
            "objective": objective,
            "mu_f": problem.mu_f,
            "mu_F": problem.mu_F,
            "L": problem.L,
            "mu_hat_ge_L": mu_hat_ge_L,
            "outside_theory": (outside_theory or not bool(mu_hat_ge_L)) if has_theory_mu_hat else False,
            "threshold_quantile": current_threshold_quantile,
            **grad_metrics,
            **state_metrics,
            **lasso_metrics,
            "x_v_distance": step_info["x_v_distance"],
            "L_k": state_metrics["lyapunov"],
            "lyapunov_s": lyapunov_coeffs.get("s", float("nan")),
        }
        if extra_metadata:
            row.update(extra_metadata)
        bad_reason = None
        if any(
            not math.isfinite(value)
            for value in [
                row["objective_gap"],
                row["lyapunov"],
                row["distance_v"],
                row["batch_grad_norm"],
                row["transformed_grad_norm"],
            ]
        ):
            status = "nonfinite"
            bad_reason = "nonfinite"
        elif (
            row["objective_gap"] > explosion_objective_gap
            or row["distance_v"] > explosion_distance_v
            or row["batch_grad_norm"] > explosion_grad_norm
        ):
            status = "exploded"
            bad_reason = "exploded"
        row["status"] = status
        row["first_bad_iteration"] = iteration if bad_reason is not None else first_bad_iteration
        row["max_objective_gap"] = max_objective_gap
        row["max_distance_v"] = max_distance_v
        row["max_batch_grad_norm"] = max_batch_grad_norm
        row["max_transformed_grad_norm"] = max_transformed_grad_norm
        if iteration % log_every == 0 or bad_reason is not None:
            rows.append(row)
        if bad_reason is not None:
            first_bad_iteration = iteration
            break
    if rows:
        final_first_bad_iteration = first_bad_iteration
        for row in rows:
            row["status"] = status
            row["first_bad_iteration"] = final_first_bad_iteration
            row["max_objective_gap"] = max_objective_gap
            row["max_distance_v"] = max_distance_v
            row["max_batch_grad_norm"] = max_batch_grad_norm
            row["max_transformed_grad_norm"] = max_transformed_grad_norm
    return pd.DataFrame(rows)
