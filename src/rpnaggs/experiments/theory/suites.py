from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import torch

from rpnaggs.experiments.theory.runner import run_theory_experiment
from rpnaggs.problems.data_generators import (
    generate_gaussian_linear_data,
    generate_leverage_mixture_linear_data,
    generate_sparse_ground_truth,
    generate_student_t_linear_data,
)
from rpnaggs.problems.finite_sum_least_squares import FiniteSumLeastSquaresProblem
from rpnaggs.problems.lasso import CompositeLassoProblem
from rpnaggs.theory.lyapunov import compute_lyapunov_coefficients
from rpnaggs.theory.solvers.registry import build_theory_solver


def _concat_nonempty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _run_deterministic_sanity_check(problem, config, test_id: str, seed: int) -> Tuple[pd.DataFrame, bool]:
    mu_hat = config.deterministic_sanity_mu_hat_factor * problem.L
    lyapunov_coeffs = compute_lyapunov_coefficients(config.deterministic_sanity_a, mu_hat, problem.mu_f, problem.mu_F)
    solver = build_theory_solver(
        method="proxnaggs",
        dim=problem.d,
        a=config.deterministic_sanity_a,
        mu_hat=mu_hat,
        robust_map="identity",
        threshold=None,
    ).to(problem.A.device, problem.A.dtype)
    solver.name = "debug_proxnaggs_identity_full_gradient"
    run_df = run_theory_experiment(
        test_id=test_id,
        run_id=f"{test_id}_deterministic_sanity_seed{seed}",
        problem=problem,
        solver=solver,
        batch_size=problem.n,
        iterations=min(config.iterations, config.calibration_iterations),
        seed=seed,
        log_every=1,
        lyapunov_coeffs=lyapunov_coeffs,
        mu_hat_factor=config.deterministic_sanity_mu_hat_factor,
        phase="debug_sanity",
        outside_theory=False,
        explosion_objective_gap=config.explosion_objective_gap,
        explosion_distance_v=config.explosion_distance_v,
        explosion_grad_norm=config.explosion_grad_norm,
    )
    objective_decreases = False
    if not run_df.empty:
        objective_decreases = bool(run_df["total_objective"].iloc[-1] <= run_df["total_objective"].iloc[0] + 1e-10)
        run_df["deterministic_sanity_passed"] = bool(run_df["status"].iloc[-1] == "success" and objective_decreases)
    return run_df, bool(not run_df.empty and run_df["deterministic_sanity_passed"].iloc[-1])


def _collect_safe_gradient_samples(problem, batch_size: int, seed: int, config) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=problem.A.device)
    generator.manual_seed(seed)
    solver = build_theory_solver(
        method="proxnaggs",
        dim=problem.d,
        a=config.deterministic_sanity_a,
        mu_hat=config.deterministic_sanity_mu_hat_factor * problem.L,
        robust_map="identity",
        threshold=None,
    ).to(problem.A.device, problem.A.dtype)
    coord_values = []
    norm_values = []
    for _ in range(config.calibration_iterations):
        x_eval = solver.prediction_point()
        batch_indices = problem.sample_batch_indices(batch_size, generator)
        grad = problem.smooth_batch_grad(x_eval, batch_indices)
        coord_values.append(torch.abs(grad).reshape(-1))
        norm_values.append(torch.linalg.norm(grad))
        solver.step(problem.smooth_full_grad(x_eval))
    return {
        "coord_abs": torch.cat(coord_values) if coord_values else torch.zeros(1, device=problem.A.device, dtype=problem.A.dtype),
        "norms": torch.stack(norm_values) if norm_values else torch.zeros(1, device=problem.A.device, dtype=problem.A.dtype),
    }


def _coord_quantile_threshold(samples: Dict[str, torch.Tensor], q: float) -> float:
    return float(torch.quantile(samples["coord_abs"], q).item())


def _norm_quantile_threshold(samples: Dict[str, torch.Tensor], q: float) -> float:
    return float(torch.quantile(samples["norms"], q).item())


def _main_method_thresholds(problem, batch_size: int, seed: int, config) -> Tuple[float, float]:
    samples = _collect_safe_gradient_samples(problem, batch_size, seed, config)
    q = config.main_threshold_quantile
    return _coord_quantile_threshold(samples, q), _norm_quantile_threshold(samples, q)


def _make_test2_problem(config, d: int, mu_reg: float, variant: str):
    if variant == "mild":
        effective_mu_reg = max(mu_reg, 0.1)
        if config.heavy_tail_kind == "student_t":
            return (
                FiniteSumLeastSquaresProblem(
                    *generate_student_t_linear_data(
                        config.n,
                        d,
                        df=max(config.heavy_tail_df, 5.0),
                        noise_std=config.noise_std,
                        seed=2000 + d,
                    )[:2],
                    mu_reg=effective_mu_reg,
                ),
                effective_mu_reg,
            )
        return (
            FiniteSumLeastSquaresProblem(
                *generate_leverage_mixture_linear_data(
                    config.n,
                    d,
                    leverage_fraction=config.heavy_tail_leverage_fraction,
                    leverage_scale=min(config.heavy_tail_leverage_scale, 10.0),
                    noise_std=config.noise_std,
                    seed=2000 + d,
                )[:2],
                mu_reg=effective_mu_reg,
            ),
            effective_mu_reg,
        )
    if config.heavy_tail_kind == "student_t":
        return (
            FiniteSumLeastSquaresProblem(
                *generate_student_t_linear_data(
                    config.n,
                    d,
                    df=min(config.heavy_tail_df, 3.0),
                    noise_std=config.noise_std,
                    seed=4000 + d,
                )[:2],
                mu_reg=mu_reg,
            ),
            mu_reg,
        )
    return (
        FiniteSumLeastSquaresProblem(
            *generate_leverage_mixture_linear_data(
                config.n,
                d,
                leverage_fraction=config.heavy_tail_leverage_fraction,
                leverage_scale=max(config.heavy_tail_leverage_scale, 20.0),
                noise_std=config.noise_std,
                seed=4000 + d,
            )[:2],
            mu_reg=mu_reg,
        ),
        mu_reg,
    )


def run_test1_suite(config) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for d in config.d_values:
        for mu_reg in config.mu_reg_values:
            A, b, _ = generate_gaussian_linear_data(config.n, d, noise_std=config.noise_std, seed=1000 + d)
            problem = FiniteSumLeastSquaresProblem(A=A, b=b, mu_reg=mu_reg)
            for seed in config.seeds:
                sanity_df, sanity_passed = _run_deterministic_sanity_check(problem, config, "test1", seed)
                frames.append(sanity_df)
                if not sanity_passed:
                    continue
                for batch_size in config.batch_sizes_test1:
                    coord_threshold, _ = _main_method_thresholds(problem, batch_size, seed, config)
                    methods = [("sgd", {"lr": config.sgd_lr, "mu_hat_factor": None})]
                    methods.append(("clipped_sgd", {"lr": config.sgd_lr, "robust_map": "coord_clip", "threshold": coord_threshold, "mu_hat_factor": None}))
                    for a in config.a_values:
                        for mu_hat_factor in config.mu_hat_factors:
                            mu_hat = mu_hat_factor * problem.L
                            methods.extend(
                                [
                                    ("proxnaggs", {"a": a, "mu_hat": mu_hat, "robust_map": "identity", "threshold": None, "mu_hat_factor": mu_hat_factor}),
                                    ("proxnaggs", {"a": a, "mu_hat": mu_hat, "robust_map": "coord_clip", "threshold": coord_threshold, "mu_hat_factor": mu_hat_factor}),
                                    ("proxnaggs", {"a": a, "mu_hat": mu_hat, "robust_map": "tanh", "threshold": coord_threshold, "mu_hat_factor": mu_hat_factor}),
                                ]
                            )
                    for method, kwargs in methods:
                        mu_hat_factor = kwargs.pop("mu_hat_factor")
                        solver = build_theory_solver(method=method, dim=d, **kwargs).to(problem.A.device, problem.A.dtype)
                        if method == "proxnaggs" and kwargs.get("robust_map") == "identity":
                            solver.name = "proxnaggs_identity"
                        elif method == "proxnaggs":
                            solver.name = f"robust_proxnaggs_{kwargs['robust_map']}"
                        run_id = f"test1_d{d}_mu{mu_reg}_b{batch_size}_{solver.name}_a{kwargs.get('a', 'na')}_mf{mu_hat_factor}_seed{seed}"
                        lyapunov_coeffs = compute_lyapunov_coefficients(
                            kwargs.get("a", config.deterministic_sanity_a),
                            kwargs.get("mu_hat", max(problem.L, 1.0)),
                            problem.mu_f,
                            problem.mu_F,
                        )
                        frames.append(
                            run_theory_experiment(
                                test_id="test1",
                                run_id=run_id,
                                problem=problem,
                                solver=solver,
                                batch_size=batch_size,
                                iterations=config.iterations,
                                seed=seed,
                                log_every=config.log_every,
                                lyapunov_coeffs=lyapunov_coeffs,
                                mu_hat_factor=mu_hat_factor,
                                outside_theory=bool(mu_hat_factor is not None and kwargs.get("mu_hat", 0.0) < problem.L),
                                explosion_objective_gap=config.explosion_objective_gap,
                                explosion_distance_v=config.explosion_distance_v,
                                explosion_grad_norm=config.explosion_grad_norm,
                            )
                        )
    return _concat_nonempty(frames)


def run_test2_suite(config, variant: str = "mild") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for d in config.d_values:
        for mu_reg in config.mu_reg_values:
            problem, effective_mu_reg = _make_test2_problem(config, d, mu_reg, variant)
            for batch_size in config.batch_sizes_test2:
                for seed in config.seeds:
                    sanity_df, sanity_passed = _run_deterministic_sanity_check(problem, config, f"test2_{variant}", seed)
                    frames.append(sanity_df)
                    if not sanity_passed:
                        continue
                    samples = _collect_safe_gradient_samples(problem, batch_size, seed, config)
                    for a in config.a_values:
                        for mu_hat_factor in config.mu_hat_factors:
                            mu_hat = mu_hat_factor * problem.L
                            lyapunov_coeffs = compute_lyapunov_coefficients(a, mu_hat, problem.mu_f, problem.mu_F)
                            for robust_map in config.test2_robust_maps:
                                quantiles = [None] if robust_map == "identity" else config.threshold_quantiles
                                for q in quantiles:
                                    threshold = None
                                    if robust_map in {"coord_clip", "tanh"} and q is not None:
                                        threshold = _coord_quantile_threshold(samples, q)
                                    elif robust_map == "norm_clip" and q is not None:
                                        threshold = _norm_quantile_threshold(samples, q)
                                    solver = build_theory_solver(
                                        method="proxnaggs",
                                        dim=d,
                                        a=a,
                                        mu_hat=mu_hat,
                                        robust_map=robust_map,
                                        threshold=threshold,
                                    ).to(problem.A.device, problem.A.dtype)
                                    solver.name = "proxnaggs_identity" if robust_map == "identity" else f"robust_proxnaggs_{robust_map}"
                                    run_id = f"test2_{variant}_d{d}_mu{effective_mu_reg}_b{batch_size}_{robust_map}_a{a}_mf{mu_hat_factor}_q{q}_seed{seed}"
                                    frames.append(
                                        run_theory_experiment(
                                            test_id=f"test2_{variant}",
                                            run_id=run_id,
                                            problem=problem,
                                            solver=solver,
                                            batch_size=batch_size,
                                            iterations=config.iterations,
                                            seed=seed,
                                            log_every=config.log_every,
                                            lyapunov_coeffs=lyapunov_coeffs,
                                            threshold_quantile=q,
                                            mu_hat_factor=mu_hat_factor,
                                            outside_theory=bool(mu_hat < problem.L),
                                            explosion_objective_gap=config.explosion_objective_gap,
                                            explosion_distance_v=config.explosion_distance_v,
                                            explosion_grad_norm=config.explosion_grad_norm,
                                        )
                                    )
    return _concat_nonempty(frames)


def run_test3_suite(config) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for d in config.d_values:
        for mu_reg in config.mu_reg_values:
            x_true = generate_sparse_ground_truth(d, sparsity=config.lasso_sparsity, seed=3000 + d)
            A, b, x_true = generate_gaussian_linear_data(config.n, d, noise_std=config.noise_std, seed=3000 + d, x_true=x_true)
            problem = CompositeLassoProblem(A=A, b=b, mu_reg=mu_reg, lam=config.lasso_lambda, x_true=x_true)
            for batch_size in config.batch_sizes_test3:
                for seed in config.seeds:
                    sanity_df, sanity_passed = _run_deterministic_sanity_check(problem, config, "test3", seed)
                    frames.append(sanity_df)
                    if not sanity_passed:
                        continue
                    coord_threshold, _ = _main_method_thresholds(problem, batch_size, seed, config)
                    methods = [
                        ("prox_sgd", {"lr": config.sgd_lr, "prox_name": "l1", "prox_lam": config.lasso_lambda, "mu_hat_factor": None}),
                        ("clipped_prox_sgd", {"lr": config.sgd_lr, "robust_map": "coord_clip", "threshold": coord_threshold, "prox_name": "l1", "prox_lam": config.lasso_lambda, "mu_hat_factor": None}),
                    ]
                    for a in config.a_values:
                        for mu_hat_factor in config.mu_hat_factors:
                            mu_hat = mu_hat_factor * problem.L
                            methods.extend(
                                [
                                    ("proxnaggs", {"a": a, "mu_hat": mu_hat, "robust_map": "identity", "threshold": None, "prox_name": "l1", "prox_lam": config.lasso_lambda, "mu_hat_factor": mu_hat_factor}),
                                    ("proxnaggs", {"a": a, "mu_hat": mu_hat, "robust_map": "coord_clip", "threshold": coord_threshold, "prox_name": "l1", "prox_lam": config.lasso_lambda, "mu_hat_factor": mu_hat_factor}),
                                    ("proxnaggs", {"a": a, "mu_hat": mu_hat, "robust_map": "tanh", "threshold": coord_threshold, "prox_name": "l1", "prox_lam": config.lasso_lambda, "mu_hat_factor": mu_hat_factor}),
                                ]
                            )
                    for method, kwargs in methods:
                        mu_hat_factor = kwargs.pop("mu_hat_factor")
                        solver = build_theory_solver(method=method, dim=d, **kwargs).to(problem.A.device, problem.A.dtype)
                        if method == "proxnaggs" and kwargs.get("robust_map") == "identity":
                            solver.name = "proxnaggs_identity"
                        elif method == "proxnaggs":
                            solver.name = f"robust_proxnaggs_{kwargs['robust_map']}"
                        run_id = f"test3_d{d}_mu{mu_reg}_b{batch_size}_{solver.name}_a{kwargs.get('a', 'na')}_mf{mu_hat_factor}_seed{seed}"
                        lyapunov_coeffs = compute_lyapunov_coefficients(
                            kwargs.get("a", config.deterministic_sanity_a),
                            kwargs.get("mu_hat", max(problem.L, 1.0)),
                            problem.mu_f,
                            problem.mu_F,
                        )
                        frames.append(
                            run_theory_experiment(
                                test_id="test3",
                                run_id=run_id,
                                problem=problem,
                                solver=solver,
                                batch_size=batch_size,
                                iterations=config.iterations,
                                seed=seed,
                                log_every=config.log_every,
                                lyapunov_coeffs=lyapunov_coeffs,
                                mu_hat_factor=mu_hat_factor,
                                outside_theory=bool(mu_hat_factor is not None and kwargs.get("mu_hat", 0.0) < problem.L),
                                explosion_objective_gap=config.explosion_objective_gap,
                                explosion_distance_v=config.explosion_distance_v,
                                explosion_grad_norm=config.explosion_grad_norm,
                            )
                        )
    return _concat_nonempty(frames)


def run_test3_local_suite(config) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for d in config.d_values:
        for mu_reg in config.mu_reg_values:
            x_true = generate_sparse_ground_truth(d, sparsity=config.lasso_sparsity, seed=5000 + d)
            A, b, x_true = generate_gaussian_linear_data(config.n, d, noise_std=config.noise_std, seed=5000 + d, x_true=x_true)
            problem = CompositeLassoProblem(A=A, b=b, mu_reg=mu_reg, lam=config.lasso_lambda, x_true=x_true)
            for batch_size in config.batch_sizes_test3:
                for seed in config.seeds:
                    sanity_df, sanity_passed = _run_deterministic_sanity_check(problem, config, "test3_local", seed)
                    frames.append(sanity_df)
                    if not sanity_passed:
                        continue
                    coord_threshold, _ = _main_method_thresholds(problem, batch_size, seed, config)
                    for a in config.test3_local_a_values:
                        for h in config.test3_local_h_values:
                            mu_hat = a / h
                            if mu_hat < problem.L:
                                continue
                            lyapunov_coeffs = compute_lyapunov_coefficients(a, mu_hat, problem.mu_f, problem.mu_F)
                            for robust_map in ["identity", "coord_clip", "tanh"]:
                                solver = build_theory_solver(
                                    method="proxnaggs",
                                    dim=d,
                                    a=a,
                                    mu_hat=mu_hat,
                                    robust_map=robust_map,
                                    threshold=None if robust_map == "identity" else coord_threshold,
                                    prox_name="l1",
                                    prox_lam=config.lasso_lambda,
                                ).to(problem.A.device, problem.A.dtype)
                                solver.name = "proxnaggs_identity" if robust_map == "identity" else f"robust_proxnaggs_{robust_map}"
                                run_id = f"test3_local_d{d}_mu{mu_reg}_b{batch_size}_{solver.name}_a{a}_h{h}_seed{seed}"
                                frames.append(
                                    run_theory_experiment(
                                        test_id="test3_local",
                                        run_id=run_id,
                                        problem=problem,
                                        solver=solver,
                                        batch_size=batch_size,
                                        iterations=config.iterations,
                                        seed=seed,
                                        log_every=config.log_every,
                                        lyapunov_coeffs=lyapunov_coeffs,
                                        mu_hat_factor=mu_hat / problem.L,
                                        outside_theory=False,
                                        explosion_objective_gap=config.explosion_objective_gap,
                                        explosion_distance_v=config.explosion_distance_v,
                                        explosion_grad_norm=config.explosion_grad_norm,
                                    )
                                )
    return _concat_nonempty(frames)


def run_test4_suite(config) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    a_value = 0.4
    constant_quantile = 0.95
    schedule_batch_points = [(1, 500, 64), (501, 1000, 128), (1001, 1500, 256), (1501, 10**9, None)]
    include_identity = "identity" in set(config.test2_robust_maps)
    for d in config.d_values:
        for mu_reg in config.mu_reg_values:
            problem, effective_mu_reg = _make_test2_problem(config, d, mu_reg, "strong")
            for seed in config.seeds:
                sanity_df, sanity_passed = _run_deterministic_sanity_check(problem, config, "test4_floor_vs_theory", seed)
                frames.append(sanity_df)
                if not sanity_passed:
                    continue
                quantiles_needed = sorted({0.90, 0.95, 0.99})
                threshold_cache: Dict[int, Dict[float, float]] = {}
                for batch_size in sorted({64, 128, 256, problem.n}):
                    samples = _collect_safe_gradient_samples(problem, batch_size, seed, config)
                    threshold_cache[batch_size] = {q: _coord_quantile_threshold(samples, q) for q in quantiles_needed}

                for batch_size in config.batch_sizes_test2:
                    if batch_size not in {64, 256}:
                        continue
                    coord_threshold = threshold_cache[int(batch_size)][constant_quantile]
                    for mu_hat_factor in config.mu_hat_factors:
                        mu_hat = mu_hat_factor * problem.L
                        if mu_hat < problem.L:
                            continue
                        lyapunov_coeffs = compute_lyapunov_coefficients(a_value, mu_hat, problem.mu_f, problem.mu_F)
                        if include_identity:
                            solver = build_theory_solver(
                                method="proxnaggs",
                                dim=d,
                                a=a_value,
                                mu_hat=mu_hat,
                                robust_map="identity",
                                threshold=None,
                            ).to(problem.A.device, problem.A.dtype)
                            solver.name = "proxnaggs_identity"
                            frames.append(
                                run_theory_experiment(
                                    test_id="test4_floor_vs_theory",
                                    run_id=f"test4_floor_d{d}_mu{effective_mu_reg}_constant_b{batch_size}_identity_a{a_value}_mf{mu_hat_factor}_seed{seed}",
                                    problem=problem,
                                    solver=solver,
                                    batch_size=batch_size,
                                    iterations=config.iterations,
                                    seed=seed,
                                    log_every=config.log_every,
                                    lyapunov_coeffs=lyapunov_coeffs,
                                    threshold_quantile=None,
                                    mu_hat_factor=mu_hat_factor,
                                    outside_theory=False,
                                    explosion_objective_gap=config.explosion_objective_gap,
                                    explosion_distance_v=config.explosion_distance_v,
                                    explosion_grad_norm=config.explosion_grad_norm,
                                    extra_metadata={
                                        "schedule_name": f"constant_b{batch_size}_identity",
                                        "schedule_family": "constant_batch",
                                        "threshold_schedule_name": "identity",
                                        "test4_part": "constant",
                                        "run_group": f"identity_b{batch_size}",
                                    },
                                )
                            )

                        solver = build_theory_solver(
                            method="proxnaggs",
                            dim=d,
                            a=a_value,
                            mu_hat=mu_hat,
                            robust_map="coord_clip",
                            threshold=coord_threshold,
                        ).to(problem.A.device, problem.A.dtype)
                        solver.name = "robust_proxnaggs_coord_clip"
                        frames.append(
                            run_theory_experiment(
                                test_id="test4_floor_vs_theory",
                                run_id=f"test4_floor_d{d}_mu{effective_mu_reg}_constant_b{batch_size}_coord_clip_a{a_value}_mf{mu_hat_factor}_q0.95_seed{seed}",
                                problem=problem,
                                solver=solver,
                                batch_size=batch_size,
                                iterations=config.iterations,
                                seed=seed,
                                log_every=config.log_every,
                                lyapunov_coeffs=lyapunov_coeffs,
                                threshold_quantile=constant_quantile,
                                mu_hat_factor=mu_hat_factor,
                                outside_theory=False,
                                explosion_objective_gap=config.explosion_objective_gap,
                                explosion_distance_v=config.explosion_distance_v,
                                explosion_grad_norm=config.explosion_grad_norm,
                                extra_metadata={
                                    "schedule_name": f"constant_b{batch_size}_coord_clip",
                                    "schedule_family": "constant_batch",
                                    "threshold_schedule_name": "fixed_q0.95",
                                    "test4_part": "constant",
                                    "run_group": f"coord_clip_b{batch_size}",
                                },
                            )
                        )

                def batch_schedule(iteration: int) -> int:
                    for start, end, batch_value in schedule_batch_points:
                        if start <= iteration <= end:
                            return problem.n if batch_value is None else int(batch_value)
                    return problem.n

                for mu_hat_factor in config.mu_hat_factors:
                    mu_hat = mu_hat_factor * problem.L
                    if mu_hat < problem.L:
                        continue
                    lyapunov_coeffs = compute_lyapunov_coefficients(a_value, mu_hat, problem.mu_f, problem.mu_F)
                    fixed_threshold = threshold_cache[64][0.95]

                    solver = build_theory_solver(
                        method="proxnaggs",
                        dim=d,
                        a=a_value,
                        mu_hat=mu_hat,
                        robust_map="coord_clip",
                        threshold=fixed_threshold,
                    ).to(problem.A.device, problem.A.dtype)
                    solver.name = "robust_proxnaggs_coord_clip"
                    frames.append(
                        run_theory_experiment(
                            test_id="test4_floor_vs_theory",
                            run_id=f"test4_floor_d{d}_mu{effective_mu_reg}_schedule_fixedclip_a{a_value}_mf{mu_hat_factor}_seed{seed}",
                            problem=problem,
                            solver=solver,
                            batch_size=64,
                            iterations=config.iterations,
                            seed=seed,
                            log_every=config.log_every,
                            lyapunov_coeffs=lyapunov_coeffs,
                            threshold_quantile=0.95,
                            mu_hat_factor=mu_hat_factor,
                            outside_theory=False,
                            explosion_objective_gap=config.explosion_objective_gap,
                            explosion_distance_v=config.explosion_distance_v,
                            explosion_grad_norm=config.explosion_grad_norm,
                            batch_size_schedule=batch_schedule,
                            batch_size_label=0,
                            threshold_schedule=lambda iteration, fixed_threshold=fixed_threshold: (fixed_threshold, 0.95),
                            extra_metadata={
                                "schedule_name": "increasing_batch_fixed_clip",
                                "schedule_family": "increasing_batch",
                                "threshold_schedule_name": "fixed_q0.95",
                                "test4_part": "schedule",
                                "run_group": "coord_clip_increasing_batch_fixed_clip",
                            },
                        )
                    )

                    def relaxed_threshold_schedule(iteration: int):
                        current_batch = batch_schedule(iteration)
                        if iteration <= 500:
                            return threshold_cache[current_batch][0.90], 0.90
                        if iteration <= 1000:
                            return threshold_cache[current_batch][0.95], 0.95
                        if iteration <= 1500:
                            return threshold_cache[current_batch][0.99], 0.99
                        return None, None

                    solver = build_theory_solver(
                        method="proxnaggs",
                        dim=d,
                        a=a_value,
                        mu_hat=mu_hat,
                        robust_map="coord_clip",
                        threshold=threshold_cache[64][0.90],
                    ).to(problem.A.device, problem.A.dtype)
                    solver.name = "robust_proxnaggs_coord_clip"
                    frames.append(
                        run_theory_experiment(
                            test_id="test4_floor_vs_theory",
                            run_id=f"test4_floor_d{d}_mu{effective_mu_reg}_schedule_relaxedclip_a{a_value}_mf{mu_hat_factor}_seed{seed}",
                            problem=problem,
                            solver=solver,
                            batch_size=64,
                            iterations=config.iterations,
                            seed=seed,
                            log_every=config.log_every,
                            lyapunov_coeffs=lyapunov_coeffs,
                            threshold_quantile=0.90,
                            mu_hat_factor=mu_hat_factor,
                            outside_theory=False,
                            explosion_objective_gap=config.explosion_objective_gap,
                            explosion_distance_v=config.explosion_distance_v,
                            explosion_grad_norm=config.explosion_grad_norm,
                            batch_size_schedule=batch_schedule,
                            batch_size_label=0,
                            threshold_schedule=relaxed_threshold_schedule,
                            extra_metadata={
                                "schedule_name": "increasing_batch_relaxed_clip",
                                "schedule_family": "increasing_batch",
                                "threshold_schedule_name": "q0.90_q0.95_q0.99_none",
                                "test4_part": "schedule",
                                "run_group": "coord_clip_increasing_batch_relaxed_clip",
                            },
                        )
                    )
    return _concat_nonempty(frames)
