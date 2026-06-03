from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import optuna
import pandas as pd

from rpnaggs.config import ExperimentConfig
from rpnaggs.experiments.runner import run_experiment


DEFAULT_METHODS = ["adamw", "sgd", "clipped_sgd", "robust_proxnaggs"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune each solver with Optuna under a shared experimental budget."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Methods to tune. Each method gets the same number of trials and epochs.",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials per method.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per Optuna trial.")
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=10000,
        help="Subset of the training set used in each trial for faster fair comparisons.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for all trials.")
    parser.add_argument("--dataset", default="cifar10", help="Dataset used in every Optuna trial.")
    parser.add_argument("--model", default="small_cifar_cnn", help="Model used in every Optuna trial.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--metric",
        choices=["val_acc", "neg_val_loss"],
        default="val_acc",
        help="Objective metric to maximize. val_acc uses final test_acc as validation proxy.",
    )
    parser.add_argument(
        "--output-dir",
        default="./robust_proxnaggs_outputs/optuna",
        help="Directory where studies, summaries, and best configurations are saved.",
    )
    parser.add_argument(
        "--rpnaggs-a-values",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.4, 0.6, 0.8],
        help="Discrete values sampled for robust Prox-NAG-GS parameter a.",
    )
    parser.add_argument(
        "--rpnaggs-mu-hat-values",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 20.0, 40.0, 100.0],
        help="Discrete values sampled for robust Prox-NAG-GS parameter mu_hat.",
    )
    parser.add_argument(
        "--rpnaggs-robust-maps",
        nargs="+",
        default=["coord_clip", "norm_clip", "tanh"],
        choices=["coord_clip", "norm_clip", "tanh"],
        help="Robust maps considered for robust Prox-NAG-GS trials.",
    )
    parser.add_argument(
        "--rpnaggs-coord-threshold-values",
        nargs="+",
        type=float,
        default=[0.002, 0.005, 0.01, 0.02, 0.05],
        help="Threshold values used when robust Prox-NAG-GS samples coord_clip or tanh.",
    )
    parser.add_argument(
        "--rpnaggs-norm-threshold-values",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.5, 1.0],
        help="Threshold values used when robust Prox-NAG-GS samples norm_clip.",
    )
    parser.add_argument(
        "--rpnaggs-warmup-fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.05, 0.1],
        help="Warmup fractions sampled for robust Prox-NAG-GS. Use 0 to disable warmup.",
    )
    return parser


def build_base_config(args: argparse.Namespace, method: str, output_dir: Path, trial_number: int) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=args.dataset,
        model_name=args.model,
        algorithm_name=method,
        output_dir=str(output_dir / method / f"trial_{trial_number:04d}"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_subset_size=args.train_subset_size,
        seed=args.seed + trial_number,
        num_workers=args.num_workers,
        robust_map="identity",
        auto_set_threshold=False,
        run_diagnostics=False,
    )


def suggest_method_config(trial: optuna.Trial, base_config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    method = base_config.algorithm_name
    config = replace(base_config)

    if method == "adamw":
        config.lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        return config

    if method == "sgd":
        config.lr = trial.suggest_float("lr", 5e-4, 2e-1, log=True)
        config.momentum = trial.suggest_float("momentum", 0.5, 0.98)
        config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        return config

    if method == "clipped_sgd":
        config.lr = trial.suggest_float("lr", 5e-4, 2e-1, log=True)
        config.momentum = trial.suggest_float("momentum", 0.5, 0.98)
        config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        config.robust_map = trial.suggest_categorical("robust_map", ["norm_clip", "coord_clip"])
        config.auto_set_threshold = False
        if config.robust_map == "coord_clip":
            config.clip_threshold = trial.suggest_categorical(
                "coord_clip_threshold", [0.002, 0.005, 0.01, 0.02, 0.05]
            )
        else:
            config.clip_threshold = trial.suggest_categorical(
                "norm_clip_threshold", [0.05, 0.1, 0.2, 0.5, 1.0]
            )
        return config

    if method == "robust_proxnaggs":
        config.pnaggs_a = trial.suggest_categorical("a", args.rpnaggs_a_values)
        config.pnaggs_mu_hat = trial.suggest_categorical("mu_hat", args.rpnaggs_mu_hat_values)
        config.robust_map = trial.suggest_categorical("robust_map", args.rpnaggs_robust_maps)
        config.auto_set_threshold = False
        if config.robust_map in {"coord_clip", "tanh"}:
            config.clip_threshold = trial.suggest_categorical(
                "coord_like_threshold", args.rpnaggs_coord_threshold_values
            )
        else:
            config.clip_threshold = trial.suggest_categorical(
                "norm_clip_threshold", args.rpnaggs_norm_threshold_values
            )
        warmup_fraction = trial.suggest_categorical("warmup_fraction", args.rpnaggs_warmup_fractions)
        config.use_warmup = warmup_fraction > 0.0
        config.warmup_steps = None
        config.warmup_fraction = warmup_fraction
        return config

    raise ValueError(f"Unsupported method: {method}")


def objective_factory(args: argparse.Namespace, method: str, root_output_dir: Path):
    metric_name = args.metric

    def objective(trial: optuna.Trial) -> float:
        base_config = build_base_config(args, method, root_output_dir, trial.number)
        trial_config = suggest_method_config(trial, base_config, args)
        result = run_experiment(trial_config, save_artifacts=False)
        history = result["history"]
        final_row = history.iloc[-1]
        best_val_acc = float(history["test_acc"].max())
        best_val_loss = float(history["test_loss"].min())
        final_val_acc = float(final_row["test_acc"])
        final_val_loss = float(final_row["test_loss"])
        trial.set_user_attr("diverged", bool(final_row.get("diverged", False)))
        trial.set_user_attr("final_train_loss", float(final_row["train_loss"]))
        trial.set_user_attr("final_val_acc", final_val_acc)
        trial.set_user_attr("final_val_loss", final_val_loss)
        trial.set_user_attr("best_val_acc", best_val_acc)
        trial.set_user_attr("best_val_loss", best_val_loss)
        trial.set_user_attr("gradient_norm", _finite_or_none(final_row.get("gradient_norm")))
        trial.set_user_attr("transformed_gradient_norm", _finite_or_none(final_row.get("transformed_gradient_norm")))
        trial.set_user_attr("clipping_ratio", _finite_or_none(final_row.get("clipping_ratio")))
        trial.set_user_attr("coordinate_clipping_ratio", _finite_or_none(final_row.get("coordinate_clipping_ratio")))
        trial.set_user_attr("x_v_distance", _finite_or_none(final_row.get("x_v_distance")))
        trial.set_user_attr("config", json.dumps(asdict(trial_config), sort_keys=True))

        score = best_val_acc if metric_name == "val_acc" else -best_val_loss
        print(
            f"[optuna] method={method} trial={trial.number} "
            f"score={score:.6f} best_val_acc={best_val_acc:.4f} "
            f"best_val_loss={best_val_loss:.4f} diverged={bool(final_row.get('diverged', False))}"
        )
        return score

    return objective


def _finite_or_none(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display_df = df.copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].map(lambda value: "" if pd.isna(value) else str(value))
    widths = {
        column: max(len(str(column)), display_df[column].map(len).max())
        for column in display_df.columns
    }
    header = "| " + " | ".join(str(column).ljust(widths[column]) for column in display_df.columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in display_df.columns) + " |"
    rows = [
        "| " + " | ".join(display_df.iloc[idx][column].ljust(widths[column]) for column in display_df.columns) + " |"
        for idx in range(len(display_df))
    ]
    return "\n".join([header, separator, *rows])


def save_method_artifacts(study: optuna.Study, method: str, method_output_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    method_output_dir.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_path = method_output_dir / "trials.csv"
    trials_df.to_csv(trials_path, index=False)

    best_trial = study.best_trial
    best_payload = {
        "method": method,
        "metric": args.metric,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
    }
    with open(method_output_dir / "best_trial.json", "w", encoding="utf-8") as fh:
        json.dump(best_payload, fh, indent=2)

    command = build_replay_command(method, best_trial.params, args)
    with open(method_output_dir / "best_command.txt", "w", encoding="utf-8") as fh:
        fh.write(command + "\n")

    return {
        "method": method,
        "best_value": best_trial.value,
        "best_val_acc": best_trial.user_attrs.get("best_val_acc"),
        "best_val_loss": best_trial.user_attrs.get("best_val_loss"),
        "diverged": best_trial.user_attrs.get("diverged"),
        "output_dir": str(method_output_dir.resolve()),
        "best_command": command,
    }


def build_replay_command(method: str, params: Dict[str, object], args: argparse.Namespace) -> str:
    clip_threshold = params.get("clip_threshold")
    if clip_threshold is None:
        clip_threshold = params.get("coord_clip_threshold")
    if clip_threshold is None:
        clip_threshold = params.get("norm_clip_threshold")
    if clip_threshold is None:
        clip_threshold = params.get("coord_like_threshold")

    parts: List[str] = [
        "rpnaggs-run",
        f"--dataset {args.dataset}",
        f"--model {args.model}",
        f"--algorithm {method}",
        f"--epochs {args.epochs}",
        f"--train-subset-size {args.train_subset_size}",
        f"--batch-size {args.batch_size}",
        f"--num-workers {args.num_workers}",
    ]

    if method == "adamw":
        parts.extend(
            [
                f"--lr {params['lr']}",
                f"--weight-decay {params['weight_decay']}",
            ]
        )
    elif method == "sgd":
        parts.extend(
            [
                f"--lr {params['lr']}",
                f"--momentum {params['momentum']}",
                f"--weight-decay {params['weight_decay']}",
            ]
        )
    elif method == "clipped_sgd":
        parts.extend(
            [
                f"--lr {params['lr']}",
                f"--momentum {params['momentum']}",
                f"--weight-decay {params['weight_decay']}",
                f"--robust-map {params['robust_map']}",
                f"--clip-threshold {clip_threshold}",
                "--no-auto-threshold",
            ]
        )
    elif method == "robust_proxnaggs":
        parts.extend(
            [
                f"--pnaggs-a {params['a']}",
                f"--pnaggs-mu-hat {params['mu_hat']}",
                f"--robust-map {params['robust_map']}",
                f"--clip-threshold {clip_threshold}",
                "--no-auto-threshold",
            ]
        )
        warmup_fraction = params.get("warmup_fraction", 0.0)
        if warmup_fraction and float(warmup_fraction) > 0.0:
            parts.extend(["--use-warmup", f"--warmup-fraction {warmup_fraction}"])
    return " \\\n  ".join(parts)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for method_index, method in enumerate(args.methods):
        sampler = optuna.samplers.TPESampler(seed=args.seed + method_index)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        print(
            f"[optuna] starting method={method} dataset={args.dataset} model={args.model} "
            f"trials={args.trials} epochs={args.epochs} train_subset_size={args.train_subset_size}"
        )
        study.optimize(objective_factory(args, method, root_output_dir), n_trials=args.trials)
        summary = save_method_artifacts(study, method, root_output_dir / method, args)
        summaries.append(summary)
        print(
            f"[optuna] finished method={method} best_value={summary['best_value']:.6f} "
            f"best_val_acc={summary['best_val_acc']}"
        )

    summary_df = pd.DataFrame(summaries).sort_values(by="best_val_acc", ascending=False)
    summary_df.to_csv(root_output_dir / "summary.csv", index=False)
    with open(root_output_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(_dataframe_to_markdown(summary_df))
        fh.write("\n")

    print(f"[optuna] saved study artifacts under {root_output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
