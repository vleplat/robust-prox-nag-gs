from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for every run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size passed to each run.")
    parser.add_argument("--label-noise-rate", type=float, default=0.2, help="Symmetric label-noise rate.")
    parser.add_argument("--output-dir", default="./robust_proxnaggs_outputs", help="Directory where all artifacts are saved.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count passed to each run.")
    parser.add_argument("--full-train", action="store_true", help="Use the full training set.")
    parser.add_argument("--train-subset-size", type=int, default=2048, help="Subset size when --full-train is not used.")
    parser.add_argument("--clipped-map", default="norm_clip", choices=["identity", "norm_clip", "coord_clip", "tanh"])
    parser.add_argument("--robust-map", default="tanh", choices=["identity", "norm_clip", "coord_clip", "tanh"])
    parser.add_argument("--robust-threshold", type=float, default=0.02, help="Threshold used by the robust Prox-NAG-GS run in the suite.")
    parser.add_argument("--robust-auto-threshold", action="store_true", help="Use automatic threshold estimation for the robust Prox-NAG-GS run.")
    parser.add_argument("--pnaggs-a", type=float, default=0.4, help="Coupling parameter for robust Prox-NAG-GS.")
    parser.add_argument("--pnaggs-mu-hat", type=float, default=1.0, help="Algorithmic curvature parameter for robust Prox-NAG-GS.")
    parser.add_argument("--use-warmup", action="store_true", help="Enable linear warmup of the effective step size.")
    parser.add_argument("--warmup-fraction", type=float, default=0.05, help="Warmup length as a fraction of total training steps.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Explicit warmup steps. Overrides --warmup-fraction when set.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip heavy-tail diagnostics in all runs.")
    parser.add_argument("--skip-compare", action="store_true", help="Do not generate comparison plots after the runs.")
    parser.add_argument("--compare-tag", default="suite_comparison", help="Suffix used for comparison output filenames.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running the remaining methods if one command fails.",
    )
    return parser


def _python_entrypoint(module: str, args: list[str]) -> list[str]:
    code = f"from {module} import main; raise SystemExit(main())"
    return [sys.executable, "-c", code, *args]


def _shared_run_args(args: argparse.Namespace) -> list[str]:
    shared = [
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
        "--seed",
        str(args.seed),
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
    if method == "adamw":
        return ["--algorithm", "adamw", "--robust-map", "identity"], "adamw_identity", "AdamW"
    if method == "sgd":
        return ["--algorithm", "sgd", "--robust-map", "identity"], "sgd_identity", "SGD"
    if method == "clipped_sgd":
        suffix = f"clipped_sgd_{args.clipped_map}"
        label = "Clipped SGD" if args.clipped_map == "norm_clip" else f"Clipped SGD ({args.clipped_map})"
        return ["--algorithm", "clipped_sgd", "--robust-map", args.clipped_map], suffix, label
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
        ]
        if args.robust_auto_threshold:
            method_args.extend(["--threshold-multiplier", "1.5"])
        else:
            method_args.extend(["--clip-threshold", str(args.robust_threshold), "--no-auto-threshold"])
        return method_args, suffix, label
    raise ValueError(f"Unsupported method: {method}")


def _run_command(command: list[str], title: str) -> int:
    print(f"\n=== {title} ===")
    print("Command:", " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT)
    return completed.returncode


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
