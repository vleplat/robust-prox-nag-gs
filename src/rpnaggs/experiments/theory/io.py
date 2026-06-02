from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


def prepare_output_dirs(root: Path) -> dict:
    csv_dir = root / "csv"
    figures_dir = root / "figures"
    configs_dir = root / "configs"
    for directory in [root, csv_dir, figures_dir, configs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": root, "csv": csv_dir, "figures": figures_dir, "configs": configs_dir}


def save_config(config, config_dir: Path, filename: str = "theory_config.json") -> None:
    with open(config_dir / filename, "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)


def save_iteration_outputs(iteration_df: pd.DataFrame, csv_dir: Path) -> None:
    iteration_df.to_csv(csv_dir / "all_iterations.csv", index=False)


def _tail_summary(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values("iteration").reset_index(drop=True)
    tail_count = max(1, int(len(ordered) * 0.2))
    tail = ordered.tail(tail_count)
    s_values = tail["lyapunov_s"] if "lyapunov_s" in tail.columns else pd.Series(dtype=float)
    s_values = s_values.dropna()
    residual_proxy = float("nan")
    if not s_values.empty:
        s_value = float(s_values.iloc[-1])
        if s_value > 0:
            residual_proxy = float(tail["transformed_error_sq"].mean() / (4.0 * s_value))
    return pd.Series(
        {
            "empirical_floor_L": float(tail["lyapunov"].mean()) if "lyapunov" in tail.columns else float("nan"),
            "empirical_floor_G": float(tail["G_k"].mean()) if "G_k" in tail.columns else float("nan"),
            "empirical_floor_V": float(tail["V_k"].mean()) if "V_k" in tail.columns else float("nan"),
            "residual_proxy": residual_proxy,
        }
    )


def save_summaries(iteration_df: pd.DataFrame, csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows = (
        iteration_df.sort_values(["test_id", "run_id", "iteration"])
        .groupby(["test_id", "run_id"], as_index=False, dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )
    aggregate_by_run = (
        iteration_df.groupby(["test_id", "run_id"], as_index=False, dropna=False)
        .agg(
            mean_clipping_ratio=("clipping_ratio", "mean"),
            max_clipping_ratio=("clipping_ratio", "max"),
            fraction_iterations_clipping_active=("clipping_ratio", lambda s: float((s.fillna(0.0) > 1e-12).mean())),
            mean_coordinate_activation_ratio_batch=("coordinate_activation_ratio_batch", "mean"),
            max_coordinate_activation_ratio_batch=("coordinate_activation_ratio_batch", "max"),
            mean_deterministic_bias_norm=("deterministic_bias_norm", "mean"),
            max_deterministic_bias_norm=("deterministic_bias_norm", "max"),
            mean_transformed_error_norm=("transformed_error_norm", "mean"),
            mean_raw_error_norm=("raw_error_norm", "mean"),
        )
    )
    tail_by_run = (
        iteration_df.groupby(["test_id", "run_id"], as_index=False, dropna=False)
        .apply(_tail_summary, include_groups=False)
        .reset_index()
        .drop(columns=["level_2"], errors="ignore")
    )
    summary_by_run = final_rows.merge(
        aggregate_by_run,
        on=["test_id", "run_id"],
        how="left",
    ).merge(
        tail_by_run,
        on=["test_id", "run_id"],
        how="left",
    )
    main_summary = summary_by_run[
        (summary_by_run["phase"] == "main")
        & (~summary_by_run["outside_theory"])
        & (summary_by_run["status"] == "success")
    ]
    group_cols = ["test_id", "method"]
    if "schedule_name" in main_summary.columns and main_summary["schedule_name"].notna().any():
        group_cols.append("schedule_name")
    elif "batch_size" in main_summary.columns:
        group_cols.append("batch_size")
    summary_by_method = (
        main_summary.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            final_objective_gap=("objective_gap", "mean"),
            final_lyapunov=("lyapunov", "mean"),
            final_transformed_error_norm=("transformed_error_norm", "mean"),
            final_distance_v=("distance_v", "mean"),
            final_sparsity=("sparsity", "mean"),
            empirical_floor_L=("empirical_floor_L", "mean"),
            empirical_floor_G=("empirical_floor_G", "mean"),
            empirical_floor_V=("empirical_floor_V", "mean"),
            residual_proxy=("residual_proxy", "mean"),
        )
    )
    summary_by_run.to_csv(csv_dir / "summary_by_run.csv", index=False)
    summary_by_method.to_csv(csv_dir / "summary_by_method.csv", index=False)
    return summary_by_run, summary_by_method
