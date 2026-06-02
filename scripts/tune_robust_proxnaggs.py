from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn

from rpnaggs.algorithms.robust_proxnaggs import RobustProxNAGGSAlgorithm
from rpnaggs.config import ExperimentConfig
from rpnaggs.data.cifar10 import get_cifar10_loaders
from rpnaggs.models.registry import build_model
from rpnaggs.tuning_distill import distill_tuning_results
from rpnaggs.utils.repro import get_device, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged tuning for robust_proxnaggs.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="small_cifar_cnn")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./robust_proxnaggs_outputs/tuning_robust_proxnaggs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--label-noise-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--train-subset-size", type=int, default=4096)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-warmup", action="store_true")
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--top-k-stage2", type=int, default=3)
    parser.add_argument("--top-k-stage3", type=int, default=3)
    parser.add_argument("--divergence-loss-threshold", type=float, default=20.0)
    parser.add_argument("--divergence-gradient-threshold", type=float, default=1e4)
    parser.add_argument("--divergence-xv-threshold", type=float, default=1e4)
    parser.add_argument("--stage1-a-values", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--stage1-mu-hat-values", nargs="+", type=float, default=[1.0, 10.0, 20.0])
    parser.add_argument("--stage1-step-size-values", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage1-eta-values", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage2-maps", nargs="+", default=["coord_clip", "norm_clip", "tanh"])
    parser.add_argument("--stage2-coord-threshold-values", nargs="+", type=float, default=[0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--stage2-norm-threshold-values", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--stage2-tanh-threshold-values", nargs="+", type=float, default=[0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--stage2-threshold-values", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage2-threshold-multipliers", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage3-mu-hat-values", nargs="+", type=float, default=[1.0, 10.0, 20.0])
    parser.add_argument("--stage3-muhat-values", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage3-a-values", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8])
    return parser


def build_base_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=args.dataset,
        model_name=args.model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_noise_rate=args.label_noise_rate,
        use_train_subset=not args.full_train,
        train_subset_size=args.train_subset_size,
        epochs=args.epochs,
        algorithm_name="robust_proxnaggs",
        weight_decay=args.weight_decay,
        robust_map="identity",
        clip_threshold=1.0,
        auto_set_threshold=False,
        threshold_multiplier=1.0,
        pnaggs_a=0.2,
        pnaggs_mu_hat=200.0,
        use_warmup=args.use_warmup,
        warmup_fraction=args.warmup_fraction,
        warmup_steps=args.warmup_steps,
        l1_reg=0.0,
        run_diagnostics=False,
        seed=args.seed,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size
    return total_loss / total, total_correct / total


def run_one_epoch(
    model: nn.Module,
    train_loader,
    algorithm: RobustProxNAGGSAlgorithm,
    criterion,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], bool]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    grad_norms: List[float] = []
    transformed_grad_norms: List[float] = []
    clipping_ratios: List[float] = []
    coordinate_clipping_ratios: List[float] = []
    x_v_distances: List[float] = []
    diverged = False

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        loss_value, acc = algorithm.train_batch(model, x, y, criterion)
        step_stats = algorithm.last_step_stats

        total_loss += loss_value * batch_size
        total_acc += acc * batch_size
        total += batch_size
        grad_norms.append(step_stats["gradient_norm"])
        transformed_grad_norms.append(step_stats["transformed_gradient_norm"])
        clipping_ratios.append(step_stats["clipping_ratio"])
        if not torch.isnan(torch.tensor(step_stats["coordinate_clipping_ratio"])).item():
            coordinate_clipping_ratios.append(step_stats["coordinate_clipping_ratio"])
        x_v_distances.append(step_stats["x_v_distance"])

        if (
            step_stats["nonfinite"]
            or not torch.isfinite(torch.tensor(loss_value))
            or loss_value > args.divergence_loss_threshold
            or step_stats["gradient_norm"] > args.divergence_gradient_threshold
            or step_stats["x_v_distance"] > args.divergence_xv_threshold
        ):
            diverged = True
            break

    epoch_metrics = {
        "train_loss": total_loss / max(total, 1),
        "train_acc": total_acc / max(total, 1),
        "gradient_norm": float(sum(grad_norms) / max(len(grad_norms), 1)),
        "transformed_gradient_norm": float(sum(transformed_grad_norms) / max(len(transformed_grad_norms), 1)),
        "clipping_ratio": float(sum(clipping_ratios) / max(len(clipping_ratios), 1)),
        "coordinate_clipping_ratio": float(sum(coordinate_clipping_ratios) / max(len(coordinate_clipping_ratios), 1))
        if coordinate_clipping_ratios
        else float("nan"),
        "x_v_distance": float(sum(x_v_distances) / max(len(x_v_distances), 1)),
    }
    return epoch_metrics, diverged


def run_tuning_experiment(
    *,
    config: ExperimentConfig,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    device: torch.device,
    stage_name: str,
    run_name: str,
    a: float,
    step_size: float,
    robust_map: str,
    clip_threshold: Optional[float],
    threshold_multiplier: Optional[float],
    calibration_gradient_norm: Optional[float],
    threshold_source: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    set_seed(config.seed)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    mu_hat = a / step_size
    algorithm = RobustProxNAGGSAlgorithm(
        model=model,
        a=a,
        mu_hat=mu_hat,
        robust_map=robust_map,
        threshold=clip_threshold,
        l1_reg=0.0,
        use_warmup=config.use_warmup,
        warmup_fraction=config.warmup_fraction,
        warmup_steps=config.warmup_steps,
        total_training_steps=len(train_loader) * config.epochs,
    )

    history_rows = []
    diverged = False
    best_val_acc = float("-inf")

    print(
        f"[{stage_name}] start {run_name} | a={a:g} | mu_hat={mu_hat:g} | step_size={step_size:g} | "
        f"map={robust_map} | threshold={clip_threshold if clip_threshold is not None else 'none'} | "
        f"threshold_source={threshold_source}"
    )

    for epoch in range(1, config.epochs + 1):
        train_metrics, epoch_diverged = run_one_epoch(model, train_loader, algorithm, criterion, device, args)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        diverged = diverged or epoch_diverged or not torch.isfinite(torch.tensor(val_loss)).item()
        best_val_acc = max(best_val_acc, val_acc)
        print(
            f"[{stage_name}] {run_name} | epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['train_loss']:.4f} | val_acc={val_acc:.4f} | "
            f"grad_norm={train_metrics['gradient_norm']:.4f} | xv={train_metrics['x_v_distance']:.4f} | "
            f"clip_ratio={train_metrics['clipping_ratio']:.4f} | "
            f"coord_clip_ratio={train_metrics['coordinate_clipping_ratio'] if pd.notna(train_metrics['coordinate_clipping_ratio']) else float('nan'):.4f}"
        )
        history_rows.append(
            {
                "stage": stage_name,
                "run_name": run_name,
                "epoch": epoch,
                "a": a,
                "mu_hat": mu_hat,
                "step_size": step_size,
                "base_step_size": algorithm.base_step_size,
                "effective_step_size": algorithm.last_step_stats["step_size"],
                "warmup_factor": algorithm.last_step_stats["warmup_factor"],
                "robust_map": robust_map,
                "clip_threshold": clip_threshold,
                "threshold_multiplier": threshold_multiplier,
                "calibration_gradient_norm": calibration_gradient_norm,
                "threshold_source": threshold_source,
                "train_loss": train_metrics["train_loss"],
                "train_acc": train_metrics["train_acc"],
                "val_loss": val_loss,
                "val_acc": val_acc,
                "gradient_norm": train_metrics["gradient_norm"],
                "transformed_gradient_norm": train_metrics["transformed_gradient_norm"],
                "clipping_ratio": train_metrics["clipping_ratio"],
                "coordinate_clipping_ratio": train_metrics["coordinate_clipping_ratio"],
                "x_v_distance": train_metrics["x_v_distance"],
                "diverged": diverged,
            }
        )
        if diverged:
            print(f"[{stage_name}] {run_name} diverged; stopping early.")
            break

    history_df = pd.DataFrame(history_rows)
    final_row = history_df.iloc[-1].to_dict()
    summary = {
        "stage": stage_name,
        "run_name": run_name,
        "a": a,
        "mu_hat": mu_hat,
        "step_size": step_size,
        "base_step_size": algorithm.base_step_size,
        "final_effective_step_size": final_row["effective_step_size"],
        "robust_map": robust_map,
        "clip_threshold": clip_threshold,
        "threshold_multiplier": threshold_multiplier,
        "calibration_gradient_norm": calibration_gradient_norm,
        "threshold_source": threshold_source,
        "final_train_loss": final_row["train_loss"],
        "final_train_acc": final_row["train_acc"],
        "final_val_loss": final_row["val_loss"],
        "final_val_acc": final_row["val_acc"],
        "best_val_acc": best_val_acc,
        "mean_gradient_norm": float(history_df["gradient_norm"].mean()),
        "median_gradient_norm": float(history_df["gradient_norm"].median()),
        "mean_transformed_gradient_norm": float(history_df["transformed_gradient_norm"].mean()),
        "mean_clipping_ratio": float(history_df["clipping_ratio"].mean()),
        "mean_coordinate_clipping_ratio": float(history_df["coordinate_clipping_ratio"].dropna().mean())
        if history_df["coordinate_clipping_ratio"].notna().any()
        else float("nan"),
        "mean_x_v_distance": float(history_df["x_v_distance"].mean()),
        "mean_warmup_factor": float(history_df["warmup_factor"].mean()),
        "diverged": bool(history_df["diverged"].iloc[-1]),
        "epochs_completed": int(len(history_df)),
    }
    print(
        f"[{stage_name}] done {run_name} | best_val_acc={summary['best_val_acc']:.4f} | "
        f"final_val_acc={summary['final_val_acc']:.4f} | diverged={summary['diverged']}"
    )
    return history_df, summary


def save_stage_outputs(
    stage_dir: Path,
    history_frames: List[pd.DataFrame],
    summary_rows: List[Dict[str, float]],
    stage_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    history_df.to_csv(stage_dir / f"{stage_name}_history.csv", index=False)
    summary_df.to_csv(stage_dir / f"{stage_name}_summary.csv", index=False)
    return history_df, summary_df


def select_top_runs(summary_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    filtered = summary_df[summary_df["diverged"] == False].copy()
    if filtered.empty:
        filtered = summary_df.copy()
    filtered = filtered.sort_values(["best_val_acc", "final_val_acc"], ascending=False)
    return filtered.head(top_k).reset_index(drop=True)


def run_stage1(
    base_config: ExperimentConfig,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    device: torch.device,
    root_output_dir: Path,
) -> pd.DataFrame:
    stage_dir = root_output_dir / "stage1"
    history_frames = []
    summary_rows = []
    if args.stage1_eta_values is not None:
        stage1_step_sizes = list(args.stage1_eta_values)
    elif args.stage1_step_size_values is not None:
        stage1_step_sizes = list(args.stage1_step_size_values)
    else:
        stage1_step_sizes = None

    stage1_mu_hat_values = args.stage1_mu_hat_values if stage1_step_sizes is None else None
    total_runs = len(args.stage1_a_values) * len(stage1_mu_hat_values if stage1_mu_hat_values is not None else stage1_step_sizes)
    run_idx = 0
    print("=== Stage 1: identity dynamics scan ===")
    for a in args.stage1_a_values:
        iter_values = stage1_mu_hat_values if stage1_mu_hat_values is not None else stage1_step_sizes
        assert iter_values is not None
        for value in iter_values:
            run_idx += 1
            print(f"[stage1] run {run_idx}/{total_runs}")
            if stage1_mu_hat_values is not None:
                mu_hat = value
                step_size = a / mu_hat
            else:
                step_size = value
                mu_hat = a / step_size
            config = ExperimentConfig(**asdict(base_config))
            config.robust_map = "identity"
            config.clip_threshold = 1.0
            config.pnaggs_a = a
            config.pnaggs_mu_hat = mu_hat
            run_name = f"stage1_a{a:g}_muhat{mu_hat:g}_h{step_size:g}"
            history_df, summary = run_tuning_experiment(
                config=config,
                args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                stage_name="stage1",
                run_name=run_name,
                a=a,
                step_size=step_size,
                robust_map="identity",
                clip_threshold=None,
                threshold_multiplier=None,
                calibration_gradient_norm=None,
                threshold_source="none",
            )
            history_frames.append(history_df)
            summary_rows.append(summary)
    _, summary_df = save_stage_outputs(stage_dir, history_frames, summary_rows, "stage1")
    selected_df = select_top_runs(summary_df, args.top_k_stage2)
    selected_df.to_csv(stage_dir / "stage1_selected.csv", index=False)
    return selected_df


def run_stage2(
    base_config: ExperimentConfig,
    args: argparse.Namespace,
    selected_stage1: pd.DataFrame,
    train_loader,
    val_loader,
    device: torch.device,
    root_output_dir: Path,
) -> pd.DataFrame:
    stage_dir = root_output_dir / "stage2"
    history_frames = []
    summary_rows = []
    stage2_threshold_mode = "manual"
    if args.stage2_threshold_multipliers is not None:
        stage2_threshold_mode = "calibrated_multiplier"
        total_runs = len(selected_stage1) * len(args.stage2_maps) * len(args.stage2_threshold_multipliers)
    else:
        if args.stage2_threshold_values is not None:
            coord_thresholds = list(args.stage2_threshold_values)
            norm_thresholds = list(args.stage2_threshold_values)
            tanh_thresholds = list(args.stage2_threshold_values)
        else:
            coord_thresholds = list(args.stage2_coord_threshold_values)
            norm_thresholds = list(args.stage2_norm_threshold_values)
            tanh_thresholds = list(args.stage2_tanh_threshold_values)
        threshold_values_by_map = {
            "coord_clip": coord_thresholds,
            "norm_clip": norm_thresholds,
            "tanh": tanh_thresholds,
        }
        total_runs = len(selected_stage1) * sum(len(threshold_values_by_map.get(map_name, [])) for map_name in args.stage2_maps)
    run_idx = 0
    print(f"=== Stage 2: robust-map scan ({total_runs} runs) ===")

    for _, row in selected_stage1.iterrows():
        a = float(row["a"])
        step_size = float(row["step_size"])
        calibration_gradient_norm = float(row["median_gradient_norm"])
        for robust_map in args.stage2_maps:
            map_threshold_values = args.stage2_threshold_multipliers if stage2_threshold_mode == "calibrated_multiplier" else threshold_values_by_map.get(robust_map, [])
            for threshold_value in map_threshold_values:
                run_idx += 1
                print(f"[stage2] run {run_idx}/{total_runs}")
                if stage2_threshold_mode == "calibrated_multiplier":
                    multiplier = float(threshold_value)
                    threshold = max(calibration_gradient_norm * multiplier, 1e-8)
                    threshold_source = "calibrated_multiplier"
                else:
                    multiplier = None
                    threshold = max(float(threshold_value), 1e-8)
                    threshold_source = "manual"
                config = ExperimentConfig(**asdict(base_config))
                config.robust_map = robust_map
                config.clip_threshold = threshold
                config.pnaggs_a = a
                config.pnaggs_mu_hat = a / step_size
                if threshold_source == "manual":
                    run_name = f"stage2_a{a:g}_muhat{(a / step_size):g}_h{step_size:g}_{robust_map}_thr{threshold:g}"
                else:
                    run_name = f"stage2_a{a:g}_muhat{(a / step_size):g}_h{step_size:g}_{robust_map}_m{multiplier:g}"
                history_df, summary = run_tuning_experiment(
                    config=config,
                    args=args,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    stage_name="stage2",
                    run_name=run_name,
                    a=a,
                    step_size=step_size,
                    robust_map=robust_map,
                    clip_threshold=threshold,
                    threshold_multiplier=multiplier,
                    calibration_gradient_norm=calibration_gradient_norm,
                    threshold_source=threshold_source,
                )
                history_frames.append(history_df)
                summary_rows.append(summary)

    _, summary_df = save_stage_outputs(stage_dir, history_frames, summary_rows, "stage2")
    selected_df = select_top_runs(summary_df, args.top_k_stage3)
    selected_df.to_csv(stage_dir / "stage2_selected.csv", index=False)
    return selected_df


def run_stage3(
    base_config: ExperimentConfig,
    args: argparse.Namespace,
    train_loader,
    val_loader,
    device: torch.device,
    root_output_dir: Path,
) -> pd.DataFrame:
    stage_dir = root_output_dir / "stage3"
    history_frames = []
    summary_rows = []
    stage3_mu_hat_values = args.stage3_muhat_values if args.stage3_muhat_values is not None else args.stage3_mu_hat_values
    total_runs = len(stage3_mu_hat_values) * len(args.stage3_a_values)
    run_idx = 0
    print(f"=== Stage 3: mu_hat reparameterization scan ({total_runs} runs) ===")
    for muhat in stage3_mu_hat_values:
        for a in args.stage3_a_values:
            run_idx += 1
            print(f"[stage3] run {run_idx}/{total_runs}")
            step_size = a / muhat
            config = ExperimentConfig(**asdict(base_config))
            config.robust_map = "identity"
            config.clip_threshold = 1.0
            config.pnaggs_a = a
            config.pnaggs_mu_hat = muhat
            run_name = f"stage3_muhat{muhat:g}_a{a:g}_h{step_size:g}"
            history_df, summary = run_tuning_experiment(
                config=config,
                args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                stage_name="stage3",
                run_name=run_name,
                a=a,
                step_size=step_size,
                robust_map="identity",
                clip_threshold=None,
                threshold_multiplier=None,
                calibration_gradient_norm=None,
                threshold_source="none",
            )
            history_frames.append(history_df)
            summary_rows.append(summary)

    _, summary_df = save_stage_outputs(stage_dir, history_frames, summary_rows, "stage3")
    summary_df.to_csv(stage_dir / "stage3_selected.csv", index=False)
    return summary_df


def write_manifest(root_output_dir: Path, args: argparse.Namespace, base_config: ExperimentConfig) -> None:
    manifest = {
        "script": "scripts/tune_robust_proxnaggs.py",
        "description": "Staged tuning for robust_proxnaggs with identity dynamics scan, robust-map scan, and mu_hat reparameterization scan.",
        "base_config": asdict(base_config),
        "args": vars(args),
        "stage_descriptions": {
            "stage1": "Identity map only. Tune dynamics over (a, mu_hat), with step_size = a / mu_hat.",
            "stage2": "Best Stage 1 pairs. Test robust maps with map-specific threshold grids and log coordinate clipping activity.",
            "stage3": "Identity map only. Reparameterize directly via mu_hat with step_size = a / mu_hat.",
        },
    }
    with open(root_output_dir / "tuning_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    base_config = build_base_config(args)
    write_manifest(root_output_dir, args, base_config)

    train_loader, _, val_loader = get_cifar10_loaders(base_config)

    selected_stage1 = run_stage1(base_config, args, train_loader, val_loader, device, root_output_dir)
    selected_stage2 = run_stage2(base_config, args, selected_stage1, train_loader, val_loader, device, root_output_dir)
    run_stage3(base_config, args, train_loader, val_loader, device, root_output_dir)

    if not selected_stage1.empty:
        selected_stage1.to_csv(root_output_dir / "best_stage1_pairs.csv", index=False)
    if not selected_stage2.empty:
        selected_stage2.to_csv(root_output_dir / "best_stage2_configs.csv", index=False)

    print("=== Distilling tuning results into recommendations ===")
    distilled = distill_tuning_results(root_output_dir)
    recommendations = distilled["recommendations"]
    if not recommendations.empty:
        top = recommendations.iloc[0]
        print(
            f"Primary recommendation: a={top['a']}, mu_hat={top['mu_hat']}, "
            f"step_size={top['step_size']}, robust_map={top['robust_map']}"
        )

    print(f"Completed staged tuning. Results saved in {root_output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
