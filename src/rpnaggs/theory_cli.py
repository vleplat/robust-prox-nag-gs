from __future__ import annotations

import argparse

import pandas as pd

from rpnaggs.experiments.theory.config import TheoryConfig
from rpnaggs.experiments.theory.io import prepare_output_dirs, save_config, save_iteration_outputs, save_summaries
from rpnaggs.experiments.theory.plots import save_theory_figures
from rpnaggs.experiments.theory.suites import run_test1_suite, run_test2_suite, run_test3_suite, run_test4_suite
from rpnaggs.utils.repro import set_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled theory-verification tests for robust Prox-NAG-GS.")
    parser.add_argument("--test", choices=["1", "2_mild", "2_strong", "3", "3_local", "4_floor_vs_theory", "all"], default="all")
    parser.add_argument("--output-dir", default="./robust_proxnaggs_outputs/theory_verification")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--d-values", nargs="+", type=int, default=[50])
    parser.add_argument("--mu-reg-values", nargs="+", type=float, default=[0.1])
    parser.add_argument("--batch-sizes-test1", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--batch-sizes-test2", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--batch-sizes-test3", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--test2-robust-maps", nargs="+", choices=["identity", "coord_clip", "norm_clip", "tanh"], default=["identity", "coord_clip", "norm_clip", "tanh"])
    parser.add_argument("--heavy-tail-kind", choices=["student_t", "leverage_mixture"], default="student_t")
    parser.add_argument("--heavy-tail-df", type=float, default=5.0)
    parser.add_argument("--heavy-tail-leverage-fraction", type=float, default=0.05)
    parser.add_argument("--heavy-tail-leverage-scale", type=float, default=10.0)
    parser.add_argument("--threshold-quantiles", nargs="+", type=float, default=[0.80, 0.90, 0.95, 0.99])
    parser.add_argument("--main-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--a-values", nargs="+", type=float, default=[0.1, 0.2, 0.4])
    parser.add_argument("--mu-hat-factors", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    parser.add_argument("--test3-local-a-values", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    parser.add_argument("--test3-local-h-values", nargs="+", type=float, default=[0.04, 0.05, 0.06, 0.08])
    parser.add_argument("--sgd-lr", type=float, default=0.05)
    parser.add_argument("--lasso-lambda", type=float, default=0.05)
    parser.add_argument("--lasso-sparsity", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.1)
    return parser


def _config_from_args(args) -> TheoryConfig:
    tests = ["1", "3", "2_mild"] if args.test == "all" else [args.test]
    return TheoryConfig(
        output_dir=args.output_dir,
        tests=tests,
        seeds=args.seeds,
        iterations=args.iterations,
        log_every=args.log_every,
        n=args.n,
        d_values=args.d_values,
        mu_reg_values=args.mu_reg_values,
        batch_sizes_test1=args.batch_sizes_test1,
        batch_sizes_test2=args.batch_sizes_test2,
        batch_sizes_test3=args.batch_sizes_test3,
        test2_robust_maps=args.test2_robust_maps,
        heavy_tail_kind=args.heavy_tail_kind,
        heavy_tail_df=args.heavy_tail_df,
        heavy_tail_leverage_fraction=args.heavy_tail_leverage_fraction,
        heavy_tail_leverage_scale=args.heavy_tail_leverage_scale,
        threshold_quantiles=args.threshold_quantiles,
        main_threshold_quantile=args.main_threshold_quantile,
        a_values=args.a_values,
        mu_hat_factors=args.mu_hat_factors,
        test3_local_a_values=args.test3_local_a_values,
        test3_local_h_values=args.test3_local_h_values,
        sgd_lr=args.sgd_lr,
        lasso_lambda=args.lasso_lambda,
        lasso_sparsity=args.lasso_sparsity,
        noise_std=args.noise_std,
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    config = _config_from_args(args)
    set_seed(config.seeds[0] if config.seeds else 123)
    output_dirs = prepare_output_dirs(config.output_path)
    save_config(config, output_dirs["configs"])

    frames = []
    if "1" in config.tests:
        print("[theory] running test 1")
        frames.append(run_test1_suite(config))
    if "2_mild" in config.tests:
        print("[theory] running test 2 mild")
        frames.append(run_test2_suite(config, variant="mild"))
    if "2_strong" in config.tests:
        print("[theory] running test 2 strong")
        frames.append(run_test2_suite(config, variant="strong"))
    if "3" in config.tests:
        print("[theory] running test 3")
        frames.append(run_test3_suite(config))
    if "3_local" in config.tests:
        print("[theory] running test 3 local")
        from rpnaggs.experiments.theory.suites import run_test3_local_suite

        frames.append(run_test3_local_suite(config))
    if "4_floor_vs_theory" in config.tests:
        print("[theory] running test 4 floor vs theory")
        frames.append(run_test4_suite(config))

    iteration_df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if iteration_df.empty:
        print("[theory] no runs were produced")
        return 1

    save_iteration_outputs(iteration_df, output_dirs["csv"])
    save_summaries(iteration_df, output_dirs["csv"])
    save_theory_figures(iteration_df, output_dirs["figures"])
    print(f"[theory] saved outputs under {output_dirs['root'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
