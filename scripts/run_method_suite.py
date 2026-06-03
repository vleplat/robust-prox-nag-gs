from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTUNA_DIR = REPO_ROOT / "robust_proxnaggs_outputs" / "optuna_first_pass"
DEFAULT_OPTUNA_ROBUST_DIR = REPO_ROOT / "robust_proxnaggs_outputs" / "optuna_rproxnaggs_polish"
# Previous robust Optuna reload directory (deep study):
# DEFAULT_OPTUNA_ROBUST_DIR = REPO_ROOT / "robust_proxnaggs_outputs" / "optuna_rproxnaggs_deep"

# Fallbacks from optuna_first_pass (10 trials, 3 epochs, 10k subset, maximize val_acc).
OPTUNA_FIRST_PASS_ADAMW = {
    "lr": 0.0010253509690168491,
    "weight_decay": 0.006351221010640699,
}
OPTUNA_FIRST_PASS_SGD = {
    "lr": 0.10796284678192915,
    "momentum": 0.643041680330606,
    "weight_decay": 6.122298041332345e-06,
}
OPTUNA_FIRST_PASS_CLIPPED_SGD = {
    "lr": 0.0832343173336595,
    "momentum": 0.8242909633616264,
    "weight_decay": 1.2973013707985646e-06,
    "robust_map": "coord_clip",
    "clip_threshold": 0.02,
}

# Fallbacks from optuna_rproxnaggs_polish (robust_proxnaggs only; 20 trials, 10 epochs, 10k subset).
OPTUNA_POLISH_ROBUST_PROXNAGGS = {
    "pnaggs_a": 0.4,
    "pnaggs_mu_hat": 0.625,
    "robust_map": "tanh",
    "clip_threshold": 0.06,
    "warmup_fraction": 0.0,
}

# Default suite seed: optuna_rproxnaggs_polish best robust trial (trial_0013, 56.08% val acc on 10k subset).
DEFAULT_SUITE_SEED = 55

# Previous fallbacks from optuna_rproxnaggs_deep (53.62% val acc on 10k subset):
# OPTUNA_DEEP_ROBUST_PROXNAGGS = {
#     "pnaggs_a": 0.4,
#     "pnaggs_mu_hat": 0.75,
#     "robust_map": "tanh",
#     "clip_threshold": 0.05,
#     "warmup_fraction": 0.0,
# }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the current benchmark methods sequentially and optionally build comparison plots."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["adamw", "sgd", "clipped_sgd", "robust_proxnaggs"],
        choices=["adamw", "sgd", "clipped_sgd", "robust_proxnaggs"],
        help="Methods to run, in order.",
    )
    parser.add_argument("--model", default="small_cifar_cnn", help="Model name passed to each run.")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name passed to each run.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for every run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size passed to each run.")
    parser.add_argument("--label-noise-rate", type=float, default=0.2, help="Symmetric label-noise rate.")
    parser.add_argument("--output-dir", default="./robust_proxnaggs_outputs", help="Directory where all artifacts are saved.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count passed to each run.")
    parser.add_argument("--full-train", action="store_true", help="Use the full training set.")
    parser.add_argument("--train-subset-size", type=int, default=2048, help="Subset size when --full-train is not used.")
    parser.add_argument(
        "--optuna-dir",
        default=None,
        help=(
            "Load AdamW / SGD / clipped SGD hyperparameters from <dir>/<method>/best_trial.json. "
            f"When omitted, uses {DEFAULT_OPTUNA_DIR} if it exists."
        ),
    )
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Do not load competitor hyperparameters from Optuna outputs.",
    )
    parser.add_argument(
        "--optuna-robust-dir",
        default=None,
        help=(
            "Load robust_proxnaggs hyperparameters from <dir>/robust_proxnaggs/best_trial.json. "
            f"When omitted, uses {DEFAULT_OPTUNA_ROBUST_DIR} if it exists."
        ),
    )
    parser.add_argument(
        "--no-optuna-robust",
        action="store_true",
        help="Do not load robust_proxnaggs hyperparameters from Optuna outputs.",
    )
    parser.add_argument("--adamw-lr", type=float, default=OPTUNA_FIRST_PASS_ADAMW["lr"])
    parser.add_argument("--adamw-weight-decay", type=float, default=OPTUNA_FIRST_PASS_ADAMW["weight_decay"])
    parser.add_argument("--sgd-lr", type=float, default=OPTUNA_FIRST_PASS_SGD["lr"])
    parser.add_argument("--sgd-momentum", type=float, default=OPTUNA_FIRST_PASS_SGD["momentum"])
    parser.add_argument("--sgd-weight-decay", type=float, default=OPTUNA_FIRST_PASS_SGD["weight_decay"])
    parser.add_argument("--clipped-lr", type=float, default=OPTUNA_FIRST_PASS_CLIPPED_SGD["lr"])
    parser.add_argument("--clipped-momentum", type=float, default=OPTUNA_FIRST_PASS_CLIPPED_SGD["momentum"])
    parser.add_argument("--clipped-weight-decay", type=float, default=OPTUNA_FIRST_PASS_CLIPPED_SGD["weight_decay"])
    parser.add_argument(
        "--clipped-map",
        default=OPTUNA_FIRST_PASS_CLIPPED_SGD["robust_map"],
        choices=["identity", "norm_clip", "coord_clip", "tanh"],
    )
    parser.add_argument(
        "--clipped-threshold",
        type=float,
        default=OPTUNA_FIRST_PASS_CLIPPED_SGD["clip_threshold"],
        help="Clip threshold for clipped SGD when --clipped-auto-threshold is not set.",
    )
    parser.add_argument(
        "--clipped-auto-threshold",
        action="store_true",
        help="Use automatic threshold estimation for clipped SGD (Optuna first pass used fixed thresholds).",
    )
    parser.add_argument(
        "--robust-map",
        default=OPTUNA_POLISH_ROBUST_PROXNAGGS["robust_map"],
        choices=["identity", "norm_clip", "coord_clip", "tanh"],
    )
    parser.add_argument(
        "--robust-threshold",
        type=float,
        default=OPTUNA_POLISH_ROBUST_PROXNAGGS["clip_threshold"],
        help="Threshold used by the robust Prox-NAG-GS run in the suite.",
    )
    parser.add_argument("--robust-auto-threshold", action="store_true", help="Use automatic threshold estimation for the robust Prox-NAG-GS run.")
    parser.add_argument(
        "--pnaggs-a",
        type=float,
        default=OPTUNA_POLISH_ROBUST_PROXNAGGS["pnaggs_a"],
        help="Coupling parameter for robust Prox-NAG-GS.",
    )
    parser.add_argument(
        "--pnaggs-mu-hat",
        type=float,
        default=OPTUNA_POLISH_ROBUST_PROXNAGGS["pnaggs_mu_hat"],
        help="Algorithmic curvature parameter for robust Prox-NAG-GS.",
    )
    parser.add_argument(
        "--robust-warmup-fraction",
        type=float,
        default=OPTUNA_POLISH_ROBUST_PROXNAGGS["warmup_fraction"],
        help="Warmup fraction for robust Prox-NAG-GS only (>0 enables --use-warmup on that run).",
    )
    parser.add_argument("--use-warmup", action="store_true", help="Enable linear warmup of the effective step size.")
    parser.add_argument("--warmup-fraction", type=float, default=0.05, help="Warmup length as a fraction of total training steps.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Explicit warmup steps. Overrides --warmup-fraction when set.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SUITE_SEED)
    parser.add_argument("--adamw-seed", type=int, default=None, help="Seed for AdamW runs (defaults to --seed).")
    parser.add_argument("--sgd-seed", type=int, default=None, help="Seed for SGD runs (defaults to --seed).")
    parser.add_argument(
        "--clipped-sgd-seed",
        type=int,
        default=None,
        help="Seed for clipped SGD runs (defaults to --seed).",
    )
    parser.add_argument(
        "--robust-proxnaggs-seed",
        type=int,
        default=None,
        help="Seed for robust Prox-NAG-GS runs (defaults to --seed).",
    )
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip heavy-tail diagnostics in all runs.")
    parser.add_argument("--skip-compare", action="store_true", help="Do not generate comparison plots after the runs.")
    parser.add_argument("--compare-tag", default="suite_comparison", help="Suffix used for comparison output filenames.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running the remaining methods if one command fails.",
    )
    return parser


def _clip_threshold_from_optuna_params(params: dict[str, Any]) -> float | None:
    for key in ("clip_threshold", "coord_clip_threshold", "norm_clip_threshold", "coord_like_threshold"):
        value = params.get(key)
        if value is not None:
            return float(value)
    return None


def _load_optuna_best_params(method: str, optuna_dir: Path) -> dict[str, Any] | None:
    path = optuna_dir / method / "best_trial.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("best_params") or {})


def apply_optuna_competitor_tuning(args: argparse.Namespace) -> Path | None:
    if args.no_optuna:
        return None

    optuna_dir = Path(args.optuna_dir) if args.optuna_dir else DEFAULT_OPTUNA_DIR
    if not optuna_dir.is_dir():
        return None

    adamw = _load_optuna_best_params("adamw", optuna_dir)
    if adamw:
        args.adamw_lr = float(adamw["lr"])
        args.adamw_weight_decay = float(adamw["weight_decay"])

    sgd = _load_optuna_best_params("sgd", optuna_dir)
    if sgd:
        args.sgd_lr = float(sgd["lr"])
        args.sgd_momentum = float(sgd["momentum"])
        args.sgd_weight_decay = float(sgd["weight_decay"])

    clipped = _load_optuna_best_params("clipped_sgd", optuna_dir)
    if clipped:
        args.clipped_lr = float(clipped["lr"])
        args.clipped_momentum = float(clipped["momentum"])
        args.clipped_weight_decay = float(clipped["weight_decay"])
        if "robust_map" in clipped:
            args.clipped_map = str(clipped["robust_map"])
        threshold = _clip_threshold_from_optuna_params(clipped)
        if threshold is not None:
            args.clipped_threshold = threshold

    return optuna_dir


def apply_optuna_robust_proxnaggs_tuning(args: argparse.Namespace) -> Path | None:
    if args.no_optuna_robust:
        return None

    optuna_dir = Path(args.optuna_robust_dir) if args.optuna_robust_dir else DEFAULT_OPTUNA_ROBUST_DIR
    if not optuna_dir.is_dir():
        return None

    params = _load_optuna_best_params("robust_proxnaggs", optuna_dir)
    if not params:
        return None

    if "a" in params:
        args.pnaggs_a = float(params["a"])
    if "mu_hat" in params:
        args.pnaggs_mu_hat = float(params["mu_hat"])
    if "robust_map" in params:
        args.robust_map = str(params["robust_map"])
    threshold = _clip_threshold_from_optuna_params(params)
    if threshold is not None:
        args.robust_threshold = threshold
    if "warmup_fraction" in params:
        args.robust_warmup_fraction = float(params["warmup_fraction"])

    return optuna_dir


def _python_entrypoint(module: str, args: list[str]) -> list[str]:
    code = f"from {module} import main; raise SystemExit(main())"
    return [sys.executable, "-c", code, *args]


def _resolve_method_seed(method: str, args: argparse.Namespace) -> int:
    method_seed = {
        "adamw": args.adamw_seed,
        "sgd": args.sgd_seed,
        "clipped_sgd": args.clipped_sgd_seed,
        "robust_proxnaggs": args.robust_proxnaggs_seed,
    }.get(method)
    return args.seed if method_seed is None else method_seed


def _shared_run_args(args: argparse.Namespace) -> list[str]:
    shared = [
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--label-noise-rate",
        str(args.label_noise_rate),
        "--output-dir",
        args.output_dir,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.full_train:
        shared.append("--full-train")
    else:
        shared.extend(["--train-subset-size", str(args.train_subset_size)])
    if args.skip_diagnostics:
        shared.append("--no-diagnostics")
    if args.use_warmup:
        shared.append("--use-warmup")
        shared.extend(["--warmup-fraction", str(args.warmup_fraction)])
        if args.warmup_steps is not None:
            shared.extend(["--warmup-steps", str(args.warmup_steps)])
    return shared


def _method_run_args(method: str, args: argparse.Namespace) -> tuple[list[str], str, str]:
    seed_args = ["--seed", str(_resolve_method_seed(method, args))]
    if method == "adamw":
        return (
            [
                "--algorithm",
                "adamw",
                "--robust-map",
                "identity",
                "--lr",
                str(args.adamw_lr),
                "--weight-decay",
                str(args.adamw_weight_decay),
                *seed_args,
            ],
            "adamw_identity",
            "AdamW",
        )
    if method == "sgd":
        return (
            [
                "--algorithm",
                "sgd",
                "--robust-map",
                "identity",
                "--lr",
                str(args.sgd_lr),
                "--momentum",
                str(args.sgd_momentum),
                "--weight-decay",
                str(args.sgd_weight_decay),
                *seed_args,
            ],
            "sgd_identity",
            "SGD",
        )
    if method == "clipped_sgd":
        suffix = f"clipped_sgd_{args.clipped_map}"
        label = "Clipped SGD" if args.clipped_map == "norm_clip" else f"Clipped SGD ({args.clipped_map})"
        method_args = [
            "--algorithm",
            "clipped_sgd",
            "--robust-map",
            args.clipped_map,
            "--lr",
            str(args.clipped_lr),
            "--momentum",
            str(args.clipped_momentum),
            "--weight-decay",
            str(args.clipped_weight_decay),
            *seed_args,
        ]
        if args.clipped_auto_threshold:
            method_args.extend(["--threshold-multiplier", "1.5"])
        else:
            method_args.extend(
                ["--clip-threshold", str(args.clipped_threshold), "--no-auto-threshold"]
            )
        return method_args, suffix, label
    if method == "robust_proxnaggs":
        suffix = f"robust_proxnaggs_{args.robust_map}"
        label = "Robust Prox-NAG-GS" if args.robust_map == "tanh" else f"Robust Prox-NAG-GS ({args.robust_map})"
        method_args = [
            "--algorithm",
            "robust_proxnaggs",
            "--robust-map",
            args.robust_map,
            "--pnaggs-a",
            str(args.pnaggs_a),
            "--pnaggs-mu-hat",
            str(args.pnaggs_mu_hat),
            *seed_args,
        ]
        if args.robust_auto_threshold:
            method_args.extend(["--threshold-multiplier", "1.5"])
        else:
            method_args.extend(["--clip-threshold", str(args.robust_threshold), "--no-auto-threshold"])
        if args.robust_warmup_fraction > 0.0:
            method_args.extend(
                ["--use-warmup", "--warmup-fraction", str(args.robust_warmup_fraction)]
            )
        return method_args, suffix, label
    raise ValueError(f"Unsupported method: {method}")


def _run_command(command: list[str], title: str) -> int:
    print(f"\n=== {title} ===")
    print("Command:", " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT)
    return completed.returncode


def run_suite(args: argparse.Namespace) -> int:
    optuna_dir = apply_optuna_competitor_tuning(args)
    if optuna_dir is not None:
        print(f"Using competitor hyperparameters from Optuna directory: {optuna_dir.resolve()}")
        print(
            "  AdamW:",
            f"lr={args.adamw_lr:.6g}, weight_decay={args.adamw_weight_decay:.6g}",
        )
        print(
            "  SGD:",
            f"lr={args.sgd_lr:.6g}, momentum={args.sgd_momentum:.6g}, weight_decay={args.sgd_weight_decay:.6g}",
        )
        print(
            "  Clipped SGD:",
            f"lr={args.clipped_lr:.6g}, momentum={args.clipped_momentum:.6g}, "
            f"weight_decay={args.clipped_weight_decay:.6g}, map={args.clipped_map}, "
            f"threshold={args.clipped_threshold}",
        )

    optuna_robust_dir = apply_optuna_robust_proxnaggs_tuning(args)
    if optuna_robust_dir is not None:
        print(f"Using robust_proxnaggs hyperparameters from Optuna directory: {optuna_robust_dir.resolve()}")
        print(
            "  Robust Prox-NAG-GS:",
            f"a={args.pnaggs_a:.6g}, mu_hat={args.pnaggs_mu_hat:.6g}, map={args.robust_map}, "
            f"threshold={args.robust_threshold:.6g}, warmup_fraction={args.robust_warmup_fraction:.6g}",
        )

    shared_args = _shared_run_args(args)
    history_paths: list[str] = []
    labels: list[str] = []
    failures: list[tuple[str, int]] = []

    for method in args.methods:
        method_args, history_suffix, label = _method_run_args(method, args)
        command = _python_entrypoint("rpnaggs.cli", [*method_args, *shared_args])
        returncode = _run_command(command, f"Running {label}")
        if returncode != 0:
            failures.append((method, returncode))
            print(f"Method {method} failed with exit code {returncode}.")
            if not args.continue_on_error:
                return returncode
            continue

        history_paths.append(str(Path(args.output_dir) / f"history_{history_suffix}.csv"))
        labels.append(label)

    if history_paths and not args.skip_compare:
        compare_args = [*history_paths, "--output-dir", args.output_dir, "--tag", args.compare_tag]
        if labels:
            compare_args.extend(["--labels", *labels])
        compare_command = _python_entrypoint("rpnaggs.compare_cli", compare_args)
        compare_returncode = _run_command(compare_command, "Building comparison figures")
        if compare_returncode != 0:
            return compare_returncode

    if failures:
        print("\nCompleted with failures:")
        for method, code in failures:
            print(f"- {method}: exit code {code}")
        return 1

    print("\nAll requested methods completed successfully.")
    print(f"Results saved in: {Path(args.output_dir).resolve()}")
    return 0


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
