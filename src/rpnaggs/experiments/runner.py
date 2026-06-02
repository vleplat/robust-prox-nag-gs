from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from rpnaggs.algorithms.registry import build_algorithm
from rpnaggs.config import ExperimentConfig
from rpnaggs.data.cifar10 import get_cifar10_loaders
from rpnaggs.diagnostics.gradients import (
    collect_gradient_error_statistics,
    estimate_threshold_from_reference_gradient,
    make_reference_subset_loader,
)
from rpnaggs.diagnostics.tails import save_tail_artifacts, save_training_history_artifacts, summarize_tail_stats
from rpnaggs.models.registry import build_model
from rpnaggs.utils.repro import get_device, set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs
    return total_loss / total, total_correct / total


def train_one_epoch(model: nn.Module, train_loader, algorithm, criterion, device: torch.device) -> dict:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    metric_sums = {
        "gradient_norm": 0.0,
        "transformed_gradient_norm": 0.0,
        "clipping_ratio": 0.0,
        "coordinate_clipping_ratio": 0.0,
        "x_v_distance": 0.0,
        "step_size": 0.0,
        "base_step_size": 0.0,
        "warmup_factor": 0.0,
        "data_loss": 0.0,
        "regularization_penalty": 0.0,
        "total_objective": 0.0,
    }
    metric_counts = {key: 0 for key in metric_sums}
    diverged = False

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        bs = x.shape[0]
        loss_value, acc = algorithm.train_batch(model, x, y, criterion)
        total_loss += loss_value * bs
        total_acc += acc * bs
        total += bs
        step_stats = getattr(algorithm, "last_step_stats", {})
        diverged = diverged or bool(step_stats.get("nonfinite", False))
        for key in metric_sums:
            value = step_stats.get(key)
            if value is None or not torch.isfinite(torch.tensor(value, dtype=torch.float32)).item():
                continue
            metric_sums[key] += float(value)
            metric_counts[key] += 1

    metrics = {
        "train_loss": total_loss / total,
        "train_acc": total_acc / total,
        "diverged": diverged,
    }
    for key in metric_sums:
        metrics[key] = metric_sums[key] / metric_counts[key] if metric_counts[key] > 0 else float("nan")
    return metrics


def run_experiment(config: ExperimentConfig, save_artifacts: bool = True) -> dict:
    set_seed(config.seed)
    device = get_device()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, train_eval_loader, test_loader = get_cifar10_loaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()

    effective_config = ExperimentConfig(**asdict(config))

    if effective_config.auto_set_threshold and effective_config.robust_map != "identity":
        ref_loader = make_reference_subset_loader(
            train_eval_loader,
            ref_size=effective_config.ref_size,
            batch_size=effective_config.batch_size,
            seed=effective_config.seed + 7,
        )
        threshold = estimate_threshold_from_reference_gradient(
            model,
            ref_loader,
            robust_map=effective_config.robust_map,
            multiplier=effective_config.threshold_multiplier,
            device=device,
        )
        effective_config.clip_threshold = max(float(threshold), 1e-8)
        print(f"Automatic clipping threshold: {effective_config.clip_threshold:.6f}")

    total_training_steps = len(train_loader) * effective_config.epochs
    algorithm = build_algorithm(model, effective_config, total_training_steps=total_training_steps)
    tag = f"{effective_config.algorithm_name.lower()}_{effective_config.robust_map}"

    history = []
    start_time = time.time()
    for epoch in range(1, effective_config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, algorithm, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "train_acc": train_metrics["train_acc"],
            "test_loss": test_loss,
            "test_acc": test_acc,
            "elapsed_sec": time.time() - start_time,
            "algorithm": algorithm.name,
            "model": effective_config.model_name,
            "robust_map": effective_config.robust_map,
            "clip_threshold": effective_config.clip_threshold,
            "a": effective_config.pnaggs_a if effective_config.algorithm_name.lower() == "robust_proxnaggs" else None,
            "mu_hat": effective_config.pnaggs_mu_hat if effective_config.algorithm_name.lower() == "robust_proxnaggs" else None,
            "prox_name": getattr(algorithm, "prox_name", None),
            "prox_target": getattr(algorithm, "prox_target", None),
            "base_step_size": getattr(algorithm, "base_step_size", None),
            "step_size": getattr(algorithm, "last_step_stats", {}).get("step_size") if hasattr(algorithm, "last_step_stats") else None,
            "use_warmup": effective_config.use_warmup if effective_config.algorithm_name.lower() == "robust_proxnaggs" else None,
            "warmup_steps": getattr(algorithm, "warmup_steps", None) if effective_config.algorithm_name.lower() == "robust_proxnaggs" else None,
            "data_loss": train_metrics["data_loss"],
            "regularization_penalty": train_metrics["regularization_penalty"],
            "total_objective": train_metrics["total_objective"],
            "gradient_norm": train_metrics["gradient_norm"],
            "transformed_gradient_norm": train_metrics["transformed_gradient_norm"],
            "clipping_ratio": train_metrics["clipping_ratio"],
            "coordinate_clipping_ratio": train_metrics["coordinate_clipping_ratio"],
            "x_v_distance": train_metrics["x_v_distance"],
            "warmup_factor": train_metrics["warmup_factor"],
            "diverged": train_metrics["diverged"],
            "device": str(device),
        }
        history.append(row)
        print(row)
        if train_metrics["diverged"]:
            break

    history_df = pd.DataFrame(history)
    if save_artifacts:
        history_df.to_csv(output_dir / f"history_{tag}.csv", index=False)
        save_training_history_artifacts(history_df, output_dir, tag)

    stats = None
    summary = None
    hill_raw = None
    hill_coord = None
    if effective_config.run_diagnostics:
        stats = collect_gradient_error_statistics(model, train_eval_loader, effective_config, device=device, tag=tag)
        summary, hill_raw, hill_coord = summarize_tail_stats(stats)
        if save_artifacts:
            save_tail_artifacts(stats, summary, hill_raw, hill_coord, output_dir)

    if save_artifacts:
        with open(output_dir / f"used_config_{tag}.json", "w", encoding="utf-8") as fh:
            json.dump(asdict(effective_config), fh, indent=2)

    return {
        "model": model,
        "history": history_df,
        "stats": stats,
        "summary": summary,
        "hill_raw": hill_raw,
        "hill_coord": hill_coord,
        "config": effective_config,
        "device": str(device),
        "output_dir": str(output_dir.resolve()),
    }
