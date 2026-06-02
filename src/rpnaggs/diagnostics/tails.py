from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rpnaggs-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FIG_DPI = 300
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#d62728"
TERTIARY_COLOR = "#2ca02c"
COLOR_CYCLE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
]


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _finalize_figure(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def _display_run_label(history_df: pd.DataFrame, fallback: str) -> str:
    if "algorithm" in history_df.columns and history_df["algorithm"].notna().any():
        algo = str(history_df["algorithm"].iloc[0]).replace("_", " ")
        algo = " ".join(token.upper() if token.lower() in {"sgd", "adamw"} else token.capitalize() for token in algo.split())
        return algo
    return fallback


def hill_tail_index(samples, k_values=None) -> pd.DataFrame:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    x = np.sort(x)[::-1]
    n = len(x)
    if n < 5:
        return pd.DataFrame({"k": [], "alpha_hat": []})
    if k_values is None:
        k_values = np.unique(np.linspace(5, max(6, n // 2), 20).astype(int))
    rows = []
    for k in k_values:
        if k + 1 >= n:
            continue
        top = x[:k]
        threshold = x[k]
        hill_value = np.mean(np.log(top / (threshold + 1e-12)))
        alpha_hat = math.inf if hill_value <= 0 else 1.0 / hill_value
        rows.append({"k": int(k), "alpha_hat": alpha_hat})
    return pd.DataFrame(rows)


def summarize_tail_stats(stats) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = stats["raw_norms"]
    transformed = stats["transformed_norms"]
    summary = pd.DataFrame(
        {
            "quantity": ["raw_error_norm", "transformed_error_norm"],
            "mean": [raw.mean(), transformed.mean()],
            "median": [np.median(raw), np.median(transformed)],
            "p90": [np.quantile(raw, 0.90), np.quantile(transformed, 0.90)],
            "p95": [np.quantile(raw, 0.95), np.quantile(transformed, 0.95)],
            "p99": [np.quantile(raw, 0.99), np.quantile(transformed, 0.99)],
            "max": [raw.max(), transformed.max()],
        }
    )
    hill_raw = hill_tail_index(raw)
    hill_coord = hill_tail_index(np.abs(stats["coord_errors"]))
    return summary, hill_raw, hill_coord


def _save_histogram(samples_a, samples_b, output_path: Path, label_a: str, label_b: str, title: str) -> None:
    _apply_plot_style()
    plt.figure(figsize=(6, 4))
    plt.hist(samples_a, bins=30, alpha=0.65, density=True, label=label_a, color=PRIMARY_COLOR, edgecolor="white")
    if samples_b is not None:
        plt.hist(samples_b, bins=30, alpha=0.55, density=True, label=label_b, color=SECONDARY_COLOR, edgecolor="white")
    plt.xlabel("error norm")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _finalize_figure(output_path)


def _save_survival_plot(samples_a, samples_b, output_path: Path, label_a: str, label_b: str, title: str) -> None:
    _apply_plot_style()
    plt.figure(figsize=(6, 4))
    color_map = {label_a: PRIMARY_COLOR, label_b: SECONDARY_COLOR}
    for samples, label in [(samples_a, label_a), (samples_b, label_b)]:
        if samples is None:
            continue
        x = np.asarray(samples, dtype=float)
        x = x[np.isfinite(x)]
        x = x[x > 0]
        x = np.sort(x)
        n = len(x)
        if n == 0:
            continue
        survival = 1.0 - np.arange(n) / n
        plt.loglog(x, survival, marker="o", markersize=3, linestyle="none", label=label, color=color_map.get(label, PRIMARY_COLOR), alpha=0.8)
    plt.xlabel("error norm")
    plt.ylabel("empirical survival P(X >= t)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    _finalize_figure(output_path)


def _save_hill_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    _apply_plot_style()
    plt.figure(figsize=(6, 4))
    if not df.empty:
        plt.plot(df["k"], df["alpha_hat"], marker="o", linewidth=2, color=TERTIARY_COLOR)
    plt.xlabel("number of upper-order statistics k")
    plt.ylabel("Hill alpha estimate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    _finalize_figure(output_path)


def save_training_history_artifacts(history_df: pd.DataFrame, output_dir: Path, tag: str) -> None:
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", linewidth=2, color=PRIMARY_COLOR, label="train")
    axes[0].plot(history_df["epoch"], history_df["test_loss"], marker="s", linewidth=2, color=SECONDARY_COLOR, label="test")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], marker="o", linewidth=2, color=PRIMARY_COLOR, label="train")
    axes[1].plot(history_df["epoch"], history_df["test_acc"], marker="s", linewidth=2, color=SECONDARY_COLOR, label="test")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.suptitle(f"Training Curves: {tag}", fontsize=14)
    _finalize_figure(output_dir / f"training_curves_{tag}.pdf")

    _apply_plot_style()
    plt.figure(figsize=(6, 4.5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", linewidth=2, color=PRIMARY_COLOR, label="train loss")
    plt.plot(history_df["epoch"], history_df["test_loss"], marker="s", linewidth=2, color=SECONDARY_COLOR, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(f"Loss Curves: {tag}")
    plt.legend()
    _finalize_figure(output_dir / f"loss_curves_{tag}.pdf")

    _apply_plot_style()
    plt.figure(figsize=(6, 4.5))
    plt.plot(history_df["epoch"], history_df["train_acc"], marker="o", linewidth=2, color=PRIMARY_COLOR, label="train accuracy")
    plt.plot(history_df["epoch"], history_df["test_acc"], marker="s", linewidth=2, color=SECONDARY_COLOR, label="test accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"Accuracy Curves: {tag}")
    plt.legend()
    _finalize_figure(output_dir / f"accuracy_curves_{tag}.pdf")


def save_history_comparison_artifacts(history_frames, output_dir: Path, tag: str, run_labels=None) -> None:
    if not history_frames:
        raise ValueError("history_frames must contain at least one history dataframe")

    _apply_plot_style()
    labels = []
    for idx, history_df in enumerate(history_frames):
        fallback = run_labels[idx] if run_labels is not None else f"run_{idx + 1}"
        labels.append(_display_run_label(history_df, fallback))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    optimizer_handles = []
    split_handles = [
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="-", label="train"),
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="--", label="test"),
    ]

    for idx, (history_df, label) in enumerate(zip(history_frames, labels)):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        axes[0].plot(history_df["epoch"], history_df["train_loss"], color=color, linewidth=2.4, linestyle="-")
        axes[0].plot(history_df["epoch"], history_df["test_loss"], color=color, linewidth=2.4, linestyle="--")
        axes[1].plot(history_df["epoch"], history_df["train_acc"], color=color, linewidth=2.4, linestyle="-")
        axes[1].plot(history_df["epoch"], history_df["test_acc"], color=color, linewidth=2.4, linestyle="--")
        optimizer_handles.append(Line2D([0], [0], color=color, linewidth=2.6, label=label))

    axes[0].set_title("Loss Comparison")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].set_title("Accuracy Comparison")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)

    optimizer_legend = axes[1].legend(handles=optimizer_handles, title="optimizer", loc="center left", bbox_to_anchor=(1.02, 0.65))
    axes[1].add_artist(optimizer_legend)
    axes[1].legend(handles=split_handles, title="split", loc="center left", bbox_to_anchor=(1.02, 0.25))

    fig.suptitle("Optimizer Comparison", fontsize=14)
    _finalize_figure(output_dir / f"comparison_curves_{tag}.pdf")

    _apply_plot_style()
    plt.figure(figsize=(6.6, 4.6))
    for idx, (history_df, label) in enumerate(zip(history_frames, labels)):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        plt.plot(history_df["epoch"], history_df["test_loss"], color=color, linewidth=2.4, linestyle="-", label=label)
    plt.xlabel("epoch")
    plt.ylabel("test loss")
    plt.title("Test Loss Comparison")
    plt.legend()
    _finalize_figure(output_dir / f"comparison_test_loss_{tag}.pdf")

    _apply_plot_style()
    plt.figure(figsize=(6.6, 4.6))
    for idx, (history_df, label) in enumerate(zip(history_frames, labels)):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        plt.plot(history_df["epoch"], history_df["test_acc"], color=color, linewidth=2.4, linestyle="-", label=label)
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Test Accuracy Comparison")
    plt.legend()
    _finalize_figure(output_dir / f"comparison_test_accuracy_{tag}.pdf")


def save_tail_artifacts(stats, summary: pd.DataFrame, hill_raw: pd.DataFrame, hill_coord: pd.DataFrame, output_dir: Path) -> None:
    tag = stats["tag"]
    summary.to_csv(output_dir / f"tail_summary_{tag}.csv", index=False)
    hill_raw.to_csv(output_dir / f"hill_raw_norms_{tag}.csv", index=False)
    hill_coord.to_csv(output_dir / f"hill_coord_errors_{tag}.csv", index=False)

    _save_histogram(
        stats["raw_norms"],
        stats["transformed_norms"],
        output_dir / f"hist_{tag}.pdf",
        label_a="raw",
        label_b="transformed",
        title=f"{tag}: gradient-error norm histogram",
    )
    _save_survival_plot(
        stats["raw_norms"],
        stats["transformed_norms"],
        output_dir / f"survival_norms_{tag}.pdf",
        label_a="raw",
        label_b="transformed",
        title=f"{tag}: log-log survival of gradient-error norms",
    )
    _save_survival_plot(
        np.abs(stats["coord_errors"]),
        None,
        output_dir / f"survival_coord_errors_{tag}.pdf",
        label_a="abs coordinate errors",
        label_b="unused",
        title=f"{tag}: coordinate-level absolute errors",
    )
    _save_hill_plot(hill_raw, output_dir / f"hill_raw_norms_{tag}.pdf", title=f"{tag}: Hill estimate for raw error norms")
    _save_hill_plot(hill_coord, output_dir / f"hill_coord_errors_{tag}.pdf", title=f"{tag}: Hill estimate for coordinate errors")
