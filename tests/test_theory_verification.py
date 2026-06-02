import math
from pathlib import Path

import torch

from rpnaggs.problems.data_generators import generate_gaussian_linear_data, generate_sparse_ground_truth
from rpnaggs.problems.finite_sum_least_squares import FiniteSumLeastSquaresProblem
from rpnaggs.problems.lasso import CompositeLassoProblem
from rpnaggs.theory.lyapunov import compute_lyapunov_coefficients, compute_lyapunov_value
from rpnaggs.theory.solvers.registry import build_theory_solver
from rpnaggs.theory_cli import main as theory_main


def test_least_squares_exact_solution_has_zero_full_gradient():
    A, b, _ = generate_gaussian_linear_data(n=64, d=10, seed=0)
    problem = FiniteSumLeastSquaresProblem(A=A, b=b, mu_reg=0.1)
    grad = problem.smooth_full_grad(problem.x_star)

    assert torch.linalg.norm(grad).item() < 1e-5
    assert problem.F_star >= 0.0


def test_lasso_reference_solution_improves_over_zero():
    x_true = generate_sparse_ground_truth(d=12, sparsity=0.25, seed=0)
    A, b, x_true = generate_gaussian_linear_data(n=80, d=12, seed=1, x_true=x_true)
    problem = CompositeLassoProblem(A=A, b=b, mu_reg=0.1, lam=0.05, x_true=x_true)

    zero_objective = problem.objective(torch.zeros(problem.d)).item()
    assert problem.F_star <= zero_objective


def test_lyapunov_coefficients_compute_valid_value():
    coeffs = compute_lyapunov_coefficients(a=0.4, mu_hat=1.0, mu_f=0.1, mu_F=0.1)
    value = compute_lyapunov_value(1.0, 2.0, 3.0, coeffs)

    assert coeffs["valid"]
    assert math.isfinite(value)


def test_vector_proxnaggs_uses_expected_step_size():
    solver = build_theory_solver(method="proxnaggs", dim=5, a=0.4, mu_hat=2.0, robust_map="identity")
    batch_grad = torch.ones(5)
    info = solver.step(batch_grad)

    assert math.isclose(info["step_size"], 0.2, rel_tol=1e-9)
    assert torch.allclose(info["x_state"], torch.zeros(5))
    assert torch.allclose(info["v_state"], -0.2 * torch.ones(5))


def test_theory_cli_smoke_test1(tmp_path: Path):
    output_dir = tmp_path / "theory_verification"
    exit_code = theory_main(
        [
            "--test",
            "1",
            "--output-dir",
            str(output_dir),
            "--iterations",
            "5",
            "--seeds",
            "0",
            "--n",
            "64",
            "--d-values",
            "10",
            "--mu-reg-values",
            "0.1",
            "--batch-sizes-test1",
            "1",
            "4",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "csv" / "all_iterations.csv").exists()
    assert (output_dir / "csv" / "summary_by_run.csv").exists()
    assert (output_dir / "figures" / "fig_1_mean_lyapunov_vs_iterations.pdf").exists()
