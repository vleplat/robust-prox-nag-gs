#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OPTUNA_MNIST_FIRST_PASS_DIR = REPO_ROOT / "robust_proxnaggs_outputs" / "optuna_mnist_first_pass"
OPTUNA_MNIST_ROBUST_DIR = REPO_ROOT / "robust_proxnaggs_outputs" / "optuna_mnist_rproxnaggs_deep"

# Hardcoded fallbacks (match best_trial.json in the Optuna output dirs above).
OPTUNA_MNIST_FIRST_PASS_ADAMW = {
    "lr": 0.0014648955132800737,
    "weight_decay": 1.461896279370496e-05,
}
OPTUNA_MNIST_FIRST_PASS_SGD = {
    "lr": 0.10796284678192915,
    "momentum": 0.643041680330606,
    "weight_decay": 6.122298041332345e-06,
}
OPTUNA_MNIST_FIRST_PASS_CLIPPED_SGD = {
    "lr": 0.07434972361893011,
    "momentum": 0.5503021300975375,
    "weight_decay": 0.0001713647315832817,
    "robust_map": "norm_clip",
    "clip_threshold": 1.0,
}
OPTUNA_MNIST_DEEP_ROBUST_PROXNAGGS = {
    "pnaggs_a": 0.6,
    "pnaggs_mu_hat": 1.25,
    "robust_map": "coord_clip",
    "clip_threshold": 0.03,
    "warmup_fraction": 0.1,
}

# Seeds from each method's Optuna best trial (critical for VGG-7-Mini on MNIST).
OPTUNA_MNIST_METHOD_SEEDS = {
    "adamw": 51,
    "sgd": 51,
    "clipped_sgd": 42,
    "robust_proxnaggs": 54,
}

SUITE_PATH = Path(__file__).resolve().parent / "run_method_suite.py"
spec = importlib.util.spec_from_file_location("run_method_suite", SUITE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load suite module from {SUITE_PATH}")
run_method_suite = importlib.util.module_from_spec(spec)
sys.modules["run_method_suite"] = run_method_suite
spec.loader.exec_module(run_method_suite)


def _load_best_trial_config(method: str, optuna_dir: Path) -> dict[str, Any] | None:
    path = optuna_dir / method / "best_trial.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    config_str = (payload.get("best_user_attrs") or {}).get("config")
    if not config_str:
        return None
    return json.loads(config_str)


def _load_best_params(method: str, optuna_dir: Path) -> dict[str, Any] | None:
    path = optuna_dir / method / "best_trial.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("best_params") or {})


def _clip_threshold_from_optuna_params(params: dict[str, Any]) -> float | None:
    for key in ("clip_threshold", "coord_clip_threshold", "norm_clip_threshold", "coord_like_threshold"):
        value = params.get(key)
        if value is not None:
            return float(value)
    return None


def _seed_from_optuna_trial(method: str, optuna_dir: Path, fallback: int) -> int:
    config = _load_best_trial_config(method, optuna_dir)
    if config and "seed" in config:
        return int(config["seed"])
    return fallback


def mnist_tuned_defaults() -> dict[str, Any]:
    """Build MNIST suite defaults from Optuna best_trial.json when present, else hardcoded fallbacks."""
    adamw = dict(OPTUNA_MNIST_FIRST_PASS_ADAMW)
    sgd = dict(OPTUNA_MNIST_FIRST_PASS_SGD)
    clipped = dict(OPTUNA_MNIST_FIRST_PASS_CLIPPED_SGD)
    robust = dict(OPTUNA_MNIST_DEEP_ROBUST_PROXNAGGS)

    loaded_adamw = _load_best_params("adamw", OPTUNA_MNIST_FIRST_PASS_DIR)
    if loaded_adamw:
        adamw["lr"] = float(loaded_adamw["lr"])
        adamw["weight_decay"] = float(loaded_adamw["weight_decay"])

    loaded_sgd = _load_best_params("sgd", OPTUNA_MNIST_FIRST_PASS_DIR)
    if loaded_sgd:
        sgd["lr"] = float(loaded_sgd["lr"])
        sgd["momentum"] = float(loaded_sgd["momentum"])
        sgd["weight_decay"] = float(loaded_sgd["weight_decay"])

    loaded_clipped = _load_best_params("clipped_sgd", OPTUNA_MNIST_FIRST_PASS_DIR)
    if loaded_clipped:
        clipped["lr"] = float(loaded_clipped["lr"])
        clipped["momentum"] = float(loaded_clipped["momentum"])
        clipped["weight_decay"] = float(loaded_clipped["weight_decay"])
        if "robust_map" in loaded_clipped:
            clipped["robust_map"] = str(loaded_clipped["robust_map"])
        threshold = _clip_threshold_from_optuna_params(loaded_clipped)
        if threshold is not None:
            clipped["clip_threshold"] = threshold

    loaded_robust = _load_best_params("robust_proxnaggs", OPTUNA_MNIST_ROBUST_DIR)
    if loaded_robust:
        if "a" in loaded_robust:
            robust["pnaggs_a"] = float(loaded_robust["a"])
        if "mu_hat" in loaded_robust:
            robust["pnaggs_mu_hat"] = float(loaded_robust["mu_hat"])
        if "robust_map" in loaded_robust:
            robust["robust_map"] = str(loaded_robust["robust_map"])
        threshold = _clip_threshold_from_optuna_params(loaded_robust)
        if threshold is not None:
            robust["clip_threshold"] = threshold
        if "warmup_fraction" in loaded_robust:
            robust["warmup_fraction"] = float(loaded_robust["warmup_fraction"])

    return {
        "dataset": "mnist",
        "model": "vgg7_mini_mnist",
        "output_dir": "./robust_proxnaggs_outputs/mnist",
        "compare_tag": "mnist_suite_comparison",
        "no_optuna": True,
        "no_optuna_robust": True,
        "adamw_lr": adamw["lr"],
        "adamw_weight_decay": adamw["weight_decay"],
        "adamw_seed": _seed_from_optuna_trial("adamw", OPTUNA_MNIST_FIRST_PASS_DIR, OPTUNA_MNIST_METHOD_SEEDS["adamw"]),
        "sgd_lr": sgd["lr"],
        "sgd_momentum": sgd["momentum"],
        "sgd_weight_decay": sgd["weight_decay"],
        "sgd_seed": _seed_from_optuna_trial("sgd", OPTUNA_MNIST_FIRST_PASS_DIR, OPTUNA_MNIST_METHOD_SEEDS["sgd"]),
        "clipped_lr": clipped["lr"],
        "clipped_momentum": clipped["momentum"],
        "clipped_weight_decay": clipped["weight_decay"],
        "clipped_map": clipped["robust_map"],
        "clipped_threshold": clipped["clip_threshold"],
        "clipped_sgd_seed": _seed_from_optuna_trial(
            "clipped_sgd", OPTUNA_MNIST_FIRST_PASS_DIR, OPTUNA_MNIST_METHOD_SEEDS["clipped_sgd"]
        ),
        "pnaggs_a": robust["pnaggs_a"],
        "pnaggs_mu_hat": robust["pnaggs_mu_hat"],
        "robust_map": robust["robust_map"],
        "robust_threshold": robust["clip_threshold"],
        "robust_warmup_fraction": robust["warmup_fraction"],
        "robust_proxnaggs_seed": _seed_from_optuna_trial(
            "robust_proxnaggs", OPTUNA_MNIST_ROBUST_DIR, OPTUNA_MNIST_METHOD_SEEDS["robust_proxnaggs"]
        ),
    }


def build_mnist_arg_parser() -> argparse.ArgumentParser:
    parser = run_method_suite.build_arg_parser()
    parser.set_defaults(**mnist_tuned_defaults())
    return parser


def _print_mnist_hyperparameters(args: argparse.Namespace) -> None:
    print("MNIST tuned hyperparameters (Optuna JSON when present, else hardcoded fallbacks):")
    print(
        "  AdamW:",
        f"lr={args.adamw_lr:.6g}, weight_decay={args.adamw_weight_decay:.6g}, seed={args.adamw_seed}",
        f"(from {OPTUNA_MNIST_FIRST_PASS_DIR.name})",
    )
    print(
        "  SGD:",
        f"lr={args.sgd_lr:.6g}, momentum={args.sgd_momentum:.6g}, weight_decay={args.sgd_weight_decay:.6g}, seed={args.sgd_seed}",
        f"(from {OPTUNA_MNIST_FIRST_PASS_DIR.name})",
    )
    print(
        "  Clipped SGD:",
        f"lr={args.clipped_lr:.6g}, momentum={args.clipped_momentum:.6g}, "
        f"weight_decay={args.clipped_weight_decay:.6g}, map={args.clipped_map}, "
        f"threshold={args.clipped_threshold}, seed={args.clipped_sgd_seed}",
        f"(from {OPTUNA_MNIST_FIRST_PASS_DIR.name})",
    )
    print(
        "  Robust Prox-NAG-GS:",
        f"a={args.pnaggs_a:.6g}, mu_hat={args.pnaggs_mu_hat:.6g}, map={args.robust_map}, "
        f"threshold={args.robust_threshold:.6g}, warmup_fraction={args.robust_warmup_fraction:.6g}, "
        f"seed={args.robust_proxnaggs_seed}",
        f"(from {OPTUNA_MNIST_ROBUST_DIR.name})",
    )


def _print_planned_commands(args: argparse.Namespace) -> None:
    shared = run_method_suite._shared_run_args(args)
    print("\nPlanned rpnaggs-run commands:")
    for method in args.methods:
        method_args, _, label = run_method_suite._method_run_args(method, args)
        command = run_method_suite._python_entrypoint("rpnaggs.cli", [*method_args, *shared])
        print(f"  {label}: {' '.join(command[3:])}")


if __name__ == "__main__":
    parser = build_mnist_arg_parser()
    args = parser.parse_args()
    _print_mnist_hyperparameters(args)
    _print_planned_commands(args)
    raise SystemExit(run_method_suite.run_suite(args))
