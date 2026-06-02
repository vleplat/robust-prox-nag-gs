from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FIG_DPI = 300
COLOR_CYCLE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
]
METHOD_COLORS = {
    "sgd": "#1f77b4",
    "clipped_sgd": "#ff7f0e",
    "prox_sgd": "#2ca02c",
    "clipped_prox_sgd": "#8c564b",
    "proxnaggs_identity": "#9467bd",
    "robust_proxnaggs_coord_clip": "#d62728",
    "robust_proxnaggs_tanh": "#17becf",
    "robust_proxnaggs_norm_clip": "#7f7f7f",
}
METHOD_LABELS = {
    "sgd": "SGD",
    "clipped_sgd": "Clipped SGD",
    "prox_sgd": "Prox-SGD",
    "clipped_prox_sgd": "Clipped Prox-SGD",
    "proxnaggs_identity": "Prox-NAG-GS",
    "robust_proxnaggs_coord_clip": "Robust Prox-NAG-GS (coord clip)",
    "robust_proxnaggs_tanh": "Robust Prox-NAG-GS (tanh)",
    "robust_proxnaggs_norm_clip": "Robust Prox-NAG-GS (norm clip)",
}
LINESTYLES = {2.0: "-", 5.0: "--", 10.0: ":"}
MARKERS = {0.1: "o", 0.2: "s", 0.4: "^"}


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _save_pdf(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()


def _method_base(method: str) -> str:
    if method.startswith("clipped_prox_sgd"):
        return "clipped_prox_sgd"
    if method.startswith("clipped_sgd"):
        return "clipped_sgd"
    if method.startswith("prox_sgd"):
        return "prox_sgd"
    if method.startswith("robust_proxnaggs_coord_clip"):
        return "robust_proxnaggs_coord_clip"
    if method.startswith("robust_proxnaggs_tanh"):
        return "robust_proxnaggs_tanh"
    if method.startswith("robust_proxnaggs_norm_clip"):
        return "robust_proxnaggs_norm_clip"
    if method.startswith("proxnaggs_identity"):
        return "proxnaggs_identity"
    return method


def _series_style(method: str, a_value, mu_hat_factor):
    method_base = _method_base(method)
    color = METHOD_COLORS.get(method_base, COLOR_CYCLE[0])
    linestyle = LINESTYLES.get(float(mu_hat_factor), "-") if pd.notna(mu_hat_factor) else "-"
    marker = MARKERS.get(round(float(a_value), 1), None) if pd.notna(a_value) else None
    return color, linestyle, marker


def _legend_handles(df: pd.DataFrame):
    method_bases = sorted({_method_base(method) for method in df["method"].dropna().unique()})
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS.get(method, COLOR_CYCLE[0]), linewidth=2.4, label=METHOD_LABELS.get(method, method))
        for method in method_bases
    ]
    factor_values = sorted({float(value) for value in df["mu_hat_factor"].dropna().unique()}) if "mu_hat_factor" in df.columns else []
    factor_handles = [
        Line2D([0], [0], color="black", linewidth=2.2, linestyle=LINESTYLES.get(value, "-"), label=f"mu_hat_factor={value:g}")
        for value in factor_values
    ]
    a_values = sorted({round(float(value), 1) for value in df["a"].dropna().unique()}) if "a" in df.columns else []
    a_handles = [
        Line2D([0], [0], color="black", linewidth=0, marker=MARKERS.get(value, "o"), markersize=6, label=f"a={value:g}")
        for value in a_values
    ]
    return method_handles, factor_handles, a_handles


def _use_log_y(y_col: str) -> bool:
    return y_col in {"lyapunov", "objective_gap", "distance_v"}


def _apply_positive_log_scale(axis, frame: pd.DataFrame, y_col: str) -> None:
    if not _use_log_y(y_col):
        return
    values = frame[y_col].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) == 0:
        return
    axis.set_yscale("log")


def _filter_test1_method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    baseline_mask = df["method"].map(_method_base).isin({"sgd", "clipped_sgd"})
    prox_df = df[
        df["method"].map(_method_base).isin(
            {"proxnaggs_identity", "robust_proxnaggs_coord_clip", "robust_proxnaggs_tanh"}
        )
    ].copy()
    available_a = sorted(float(value) for value in prox_df["a"].dropna().unique())
    available_factors = sorted(float(value) for value in prox_df["mu_hat_factor"].dropna().unique())
    target_a = 0.4
    if available_a and not any(np.isclose(target_a, value) for value in available_a):
        target_a = available_a[0]
    target_factor = 2.0
    if len(available_factors) == 1:
        target_factor = available_factors[0]
    elif available_factors and not any(np.isclose(target_factor, value) for value in available_factors):
        target_factor = available_factors[0]
    prox_mask = (
        df["method"].map(_method_base).isin(
            {"proxnaggs_identity", "robust_proxnaggs_coord_clip", "robust_proxnaggs_tanh"}
        )
        & np.isclose(df["a"], target_a, rtol=0.0, atol=1e-12)
        & np.isclose(df["mu_hat_factor"], target_factor, rtol=0.0, atol=1e-12)
    )
    return df[baseline_mask | prox_mask].copy()


def _filter_test3_method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    baseline_mask = df["method"].map(_method_base).isin({"prox_sgd", "clipped_prox_sgd"})
    prox_df = df[
        df["method"].map(_method_base).isin(
            {"proxnaggs_identity", "robust_proxnaggs_coord_clip", "robust_proxnaggs_tanh"}
        )
    ].copy()
    available_a = sorted(float(value) for value in prox_df["a"].dropna().unique())
    available_factors = sorted(float(value) for value in prox_df["mu_hat_factor"].dropna().unique())
    target_a = 0.4
    if available_a and not any(np.isclose(target_a, value) for value in available_a):
        target_a = available_a[0]
    target_factor = 2.0
    if len(available_factors) == 1:
        target_factor = available_factors[0]
    elif available_factors and not any(np.isclose(target_factor, value) for value in available_factors):
        target_factor = available_factors[0]
    prox_mask = (
        df["method"].map(_method_base).isin(
            {"proxnaggs_identity", "robust_proxnaggs_coord_clip", "robust_proxnaggs_tanh"}
        )
        & np.isclose(df["a"], target_a, rtol=0.0, atol=1e-12)
        & np.isclose(df["mu_hat_factor"], target_factor, rtol=0.0, atol=1e-12)
    )
    return df[baseline_mask | prox_mask].copy()


def _filter_test2_method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    method_mask = df["method"].map(_method_base).isin(
        {
            "proxnaggs_identity",
            "robust_proxnaggs_coord_clip",
            "robust_proxnaggs_tanh",
        }
    )
    available_a = sorted(float(value) for value in df.loc[method_mask, "a"].dropna().unique())
    available_factors = sorted(float(value) for value in df.loc[method_mask, "mu_hat_factor"].dropna().unique())
    available_quantiles = sorted(float(value) for value in df.loc[method_mask & df["threshold_quantile"].notna(), "threshold_quantile"].dropna().unique())
    target_a = 0.4
    if available_a and not any(np.isclose(target_a, value) for value in available_a):
        target_a = available_a[0]
    target_factor = 2.0
    if available_factors and not any(np.isclose(target_factor, value) for value in available_factors):
        target_factor = available_factors[0]
    target_quantile = 0.95
    if available_quantiles and not any(np.isclose(target_quantile, value) for value in available_quantiles):
        target_quantile = available_quantiles[0]
    return df[
        method_mask
        & np.isclose(df["a"], target_a, rtol=0.0, atol=1e-12)
        & np.isclose(df["mu_hat_factor"], target_factor, rtol=0.0, atol=1e-12)
        & (
            df["threshold_quantile"].isna()
            | np.isclose(df["threshold_quantile"], target_quantile, rtol=0.0, atol=1e-12)
        )
    ].copy()


def _build_run_summary(df: pd.DataFrame) -> pd.DataFrame:
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

    final_rows = (
        df.sort_values(["test_id", "run_id", "iteration"])
        .groupby(["test_id", "run_id"], as_index=False, dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )
    aggregate_by_run = (
        df.groupby(["test_id", "run_id"], as_index=False, dropna=False)
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
        df.groupby(["test_id", "run_id"], as_index=False, dropna=False)
        .apply(_tail_summary, include_groups=False)
        .reset_index()
        .drop(columns=["level_2"], errors="ignore")
    )
    return final_rows.merge(
        aggregate_by_run,
        on=["test_id", "run_id"],
        how="left",
    ).merge(
        tail_by_run,
        on=["test_id", "run_id"],
        how="left",
    ).rename(
        columns={
            "objective_gap": "final_objective_gap",
            "distance_v": "final_distance_v",
        }
    )


def _plot_mean_by_iteration(df: pd.DataFrame, y_col: str, output_path: Path, title: str) -> None:
    _apply_plot_style()
    batch_sizes = sorted(df["batch_size"].dropna().unique())
    ncols = min(2, max(1, len(batch_sizes)))
    nrows = int(np.ceil(len(batch_sizes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.5 * nrows), squeeze=False)
    grouped = df.groupby(["method", "batch_size", "mu_hat_factor", "a", "iteration"], as_index=False, dropna=False)[y_col].mean()
    for axis, batch_size in zip(axes.flatten(), batch_sizes):
        batch_frame = grouped[grouped["batch_size"] == batch_size]
        for (method, mu_hat_factor, a_value), frame in batch_frame.groupby(["method", "mu_hat_factor", "a"], dropna=False):
            color, linestyle, marker = _series_style(method, a_value, mu_hat_factor)
            markevery = max(len(frame) // 12, 1)
            axis.plot(
                frame["iteration"],
                frame[y_col],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=markevery,
                markersize=4.5,
                linewidth=2.1,
                alpha=0.95,
            )
        axis.set_title(f"|B| = {batch_size}")
        axis.set_xlabel("iteration")
        axis.set_ylabel(y_col.replace("_", " "))
        _apply_positive_log_scale(axis, batch_frame, y_col)
    for axis in axes.flatten()[len(batch_sizes):]:
        axis.axis("off")
    method_handles, factor_handles, a_handles = _legend_handles(grouped)
    if method_handles:
        fig.legend(handles=method_handles, loc="upper center", ncol=min(len(method_handles), 3), title="method", bbox_to_anchor=(0.5, 1.03))
    if factor_handles:
        fig.legend(handles=factor_handles, loc="upper right", title="line style", bbox_to_anchor=(1.02, 1.0))
    if a_handles:
        fig.legend(handles=a_handles, loc="center right", title="marker", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, fontsize=14)
    _save_pdf(output_path)


def _plot_final_vs_batch(df: pd.DataFrame, y_col: str, output_path: Path, title: str) -> None:
    _apply_plot_style()
    final_rows = (
        df.sort_values("iteration")
        .groupby(["method", "batch_size", "seed", "run_id"], as_index=False, dropna=False)
        .tail(1)
    )
    grouped = final_rows.groupby(["method", "batch_size", "mu_hat_factor", "a"], as_index=False, dropna=False)[y_col].mean()
    method_bases = sorted({_method_base(method) for method in grouped["method"].dropna().unique()})
    ncols = min(2, max(1, len(method_bases)))
    nrows = int(np.ceil(len(method_bases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.5 * nrows), squeeze=False)
    for axis, method_base in zip(axes.flatten(), method_bases):
        method_frame = grouped[grouped["method"].map(_method_base) == method_base]
        if method_frame.empty:
            axis.axis("off")
            continue
        for (method, mu_hat_factor, a_value), frame in method_frame.groupby(["method", "mu_hat_factor", "a"], dropna=False):
            color, linestyle, marker = _series_style(method, a_value, mu_hat_factor)
            axis.plot(
                frame["batch_size"],
                frame[y_col],
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2.1,
                markersize=5.0,
            )
        axis.set_xscale("log", base=2)
        axis.set_title(METHOD_LABELS.get(method_base, method_base))
        axis.set_xlabel("batch size")
        axis.set_ylabel(y_col.replace("_", " "))
        _apply_positive_log_scale(axis, method_frame, y_col)
    for axis in axes.flatten()[len(method_bases):]:
        axis.axis("off")
    method_handles, factor_handles, a_handles = _legend_handles(grouped)
    if factor_handles:
        fig.legend(handles=factor_handles, loc="upper center", ncol=len(factor_handles), title="line style", bbox_to_anchor=(0.5, 1.03))
    if a_handles:
        fig.legend(handles=a_handles, loc="upper right", title="marker", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(title, fontsize=14)
    _save_pdf(output_path)


def _plot_survival(df: pd.DataFrame, value_col: str, group_col: str, output_path: Path, title: str) -> None:
    _apply_plot_style()
    plt.figure(figsize=(6.5, 4.5))
    for idx, (group_name, frame) in enumerate(df.groupby(group_col)):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        values = np.sort(frame[value_col].to_numpy(dtype=float))
        values = values[np.isfinite(values)]
        values = values[values > 0]
        if len(values) == 0:
            continue
        survival = 1.0 - np.arange(len(values)) / len(values)
        plt.loglog(values, survival, marker="o", linestyle="none", markersize=3, alpha=0.75, color=color, label=str(group_name))
    plt.xlabel(value_col.replace("_", " "))
    plt.ylabel("empirical survival")
    plt.title(title)
    plt.legend()
    _save_pdf(output_path)


def _plot_metric_by_iteration(df: pd.DataFrame, y_col: str, output_path: Path, title: str, log_y: bool = False) -> None:
    _apply_plot_style()
    batch_sizes = sorted(df["batch_size"].dropna().unique())
    ncols = min(2, max(1, len(batch_sizes)))
    nrows = int(np.ceil(len(batch_sizes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.5 * nrows), squeeze=False)
    grouped = df.groupby(["method", "batch_size", "iteration"], as_index=False, dropna=False)[y_col].mean()
    for axis, batch_size in zip(axes.flatten(), batch_sizes):
        batch_frame = grouped[grouped["batch_size"] == batch_size]
        for method, frame in batch_frame.groupby("method", dropna=False):
            method_base = _method_base(method)
            axis.plot(
                frame["iteration"],
                frame[y_col],
                color=METHOD_COLORS.get(method_base, COLOR_CYCLE[0]),
                linewidth=2.2,
                label=METHOD_LABELS.get(method_base, method),
            )
        axis.set_title(f"|B| = {batch_size}")
        axis.set_xlabel("iteration")
        axis.set_ylabel(y_col.replace("_", " "))
        if log_y:
            _apply_positive_log_scale(axis, batch_frame, y_col)
    for axis in axes.flatten()[len(batch_sizes):]:
        axis.axis("off")
    method_handles, _, _ = _legend_handles(grouped)
    if method_handles:
        fig.legend(handles=method_handles, loc="upper center", ncol=min(len(method_handles), 4), title="method", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, fontsize=14)
    _save_pdf(output_path)


def _plot_lyapunov_components(df: pd.DataFrame, output_path: Path, title: str) -> None:
    _apply_plot_style()
    batch_sizes = sorted(df["batch_size"].dropna().unique())
    fig, axes = plt.subplots(len(batch_sizes), 4, figsize=(20, 4.5 * len(batch_sizes)), squeeze=False)
    metrics = [("G_k", True), ("V_k", True), ("X_k", True), ("lyapunov", True)]
    grouped = df.groupby(["method", "batch_size", "iteration"], as_index=False, dropna=False)[["G_k", "V_k", "X_k", "lyapunov"]].mean()
    for row_idx, batch_size in enumerate(batch_sizes):
        batch_frame = grouped[grouped["batch_size"] == batch_size]
        for col_idx, (metric, log_y) in enumerate(metrics):
            axis = axes[row_idx][col_idx]
            for method, frame in batch_frame.groupby("method", dropna=False):
                method_base = _method_base(method)
                axis.plot(
                    frame["iteration"],
                    frame[metric],
                    color=METHOD_COLORS.get(method_base, COLOR_CYCLE[0]),
                    linewidth=2.1,
                )
            axis.set_title(f"|B| = {batch_size} | {metric}")
            axis.set_xlabel("iteration")
            axis.set_ylabel(metric)
            if log_y:
                _apply_positive_log_scale(axis, batch_frame, metric)
    method_handles, _, _ = _legend_handles(grouped)
    if method_handles:
        fig.legend(handles=method_handles, loc="upper center", ncol=min(len(method_handles), 4), title="method", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, fontsize=14)
    _save_pdf(output_path)


def _plot_threshold_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    _apply_plot_style()
    plt.figure(figsize=(6.5, 4.5))
    grouped = (
        df.groupby(["method", "threshold_quantile"], as_index=False)[["objective_gap", "clipping_ratio"]]
        .mean()
    )
    for idx, (method, frame) in enumerate(grouped.groupby("method")):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        plt.plot(frame["clipping_ratio"], frame["objective_gap"], marker="o", linewidth=2, color=color, label=method)
    plt.xlabel("mean clipping ratio")
    plt.ylabel("mean objective gap")
    plt.title("Threshold Tradeoff")
    plt.legend()
    _save_pdf(output_path)


def _plot_threshold_metric_sweep(
    summary_df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str,
    y_label: str,
    log_y: bool = False,
) -> None:
    threshold_levels = sorted(float(value) for value in summary_df["threshold_quantile"].dropna().unique())
    method_bases = [
        method
        for method in ["proxnaggs_identity", "robust_proxnaggs_coord_clip", "robust_proxnaggs_tanh"]
        if method in {_method_base(name) for name in summary_df["method"].dropna().unique()}
    ]
    batch_sizes = sorted(summary_df["batch_size"].dropna().unique())
    if not threshold_levels or not method_bases or not batch_sizes:
        return
    _apply_plot_style()
    fig, axes = plt.subplots(
        len(method_bases),
        len(batch_sizes),
        figsize=(6.2 * len(batch_sizes), 4.0 * len(method_bases)),
        squeeze=False,
    )
    grouped = (
        summary_df.groupby(["method", "batch_size", "a", "mu_hat_factor", "threshold_quantile"], as_index=False, dropna=False)[metric_col]
        .mean()
    )
    x_values = np.asarray(threshold_levels, dtype=float)
    for row_idx, method_base in enumerate(method_bases):
        for col_idx, batch_size in enumerate(batch_sizes):
            axis = axes[row_idx][col_idx]
            frame = grouped[(grouped["method"].map(_method_base) == method_base) & (grouped["batch_size"] == batch_size)]
            if frame.empty:
                axis.axis("off")
                continue
            if method_base == "proxnaggs_identity":
                identity_rows = frame.groupby(["method", "mu_hat_factor", "a"], as_index=False, dropna=False)[metric_col].mean()
                for _, row in identity_rows.iterrows():
                    color, linestyle, marker = _series_style(row["method"], row["a"], row["mu_hat_factor"])
                    axis.plot(
                        x_values,
                        np.full_like(x_values, float(row[metric_col]), dtype=float),
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markevery=[0],
                        linewidth=2.1,
                        markersize=4.5,
                        alpha=0.95,
                    )
            else:
                for (method, mu_hat_factor, a_value), series in frame.groupby(["method", "mu_hat_factor", "a"], dropna=False):
                    color, linestyle, marker = _series_style(method, a_value, mu_hat_factor)
                    series = series.sort_values("threshold_quantile")
                    axis.plot(
                        series["threshold_quantile"],
                        series[metric_col],
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=2.1,
                        markersize=4.5,
                        alpha=0.95,
                    )
            axis.set_title(f"{METHOD_LABELS.get(method_base, method_base)} | |B| = {batch_size}")
            axis.set_xlabel("threshold quantile")
            axis.set_ylabel(y_label)
            axis.set_xticks(threshold_levels)
            if log_y:
                _apply_positive_log_scale(axis, frame, metric_col)
    method_handles, factor_handles, a_handles = _legend_handles(grouped)
    if method_handles:
        fig.legend(handles=method_handles, loc="upper center", ncol=min(len(method_handles), 3), title="method", bbox_to_anchor=(0.5, 1.03))
    if factor_handles:
        fig.legend(handles=factor_handles, loc="upper right", title="line style", bbox_to_anchor=(1.02, 1.0))
    if a_handles:
        fig.legend(handles=a_handles, loc="center right", title="marker", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, fontsize=14)
    _save_pdf(output_path)


def _plot_best_runs_by_metric(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    _apply_plot_style()
    batch_sizes = sorted(df["batch_size"].dropna().unique())
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(7.0 * len(batch_sizes), 4.8), squeeze=False)
    final_rows = (
        df.sort_values("iteration")
        .groupby(["method", "batch_size", "seed", "run_id"], as_index=False, dropna=False)
        .tail(1)
    )
    best_rows = (
        final_rows.sort_values(metric)
        .groupby(["method", "batch_size"], as_index=False, dropna=False)
        .head(1)
    )
    for axis, batch_size in zip(axes.flatten(), batch_sizes):
        batch_best = best_rows[best_rows["batch_size"] == batch_size]
        for _, best in batch_best.iterrows():
            frame = df[df["run_id"] == best["run_id"]].sort_values("iteration")
            method_base = _method_base(best["method"])
            axis.plot(
                frame["iteration"],
                frame[metric],
                color=METHOD_COLORS.get(method_base, COLOR_CYCLE[0]),
                linewidth=2.4,
                label=METHOD_LABELS.get(method_base, best["method"]),
            )
        axis.set_title(f"|B| = {batch_size}")
        axis.set_xlabel("iteration")
        axis.set_ylabel(metric.replace("_", " "))
        _apply_positive_log_scale(axis, batch_best, metric)
    method_handles, _, _ = _legend_handles(best_rows)
    if method_handles:
        fig.legend(handles=method_handles, loc="upper center", ncol=min(len(method_handles), 4), title="best run per method", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, fontsize=14)
    _save_pdf(output_path)


def _plot_heatmap_grid(df: pd.DataFrame, value_col: str, output_path: Path, title: str) -> None:
    _apply_plot_style()
    methods = [method for method in ["proxnaggs_identity", "robust_proxnaggs_coord_clip", "robust_proxnaggs_tanh"] if method in {_method_base(m) for m in df["method"].unique()}]
    batch_sizes = sorted(df["batch_size"].dropna().unique())
    fig, axes = plt.subplots(len(methods), len(batch_sizes), figsize=(5.0 * len(batch_sizes), 4.2 * len(methods)), squeeze=False)
    final_rows = (
        df.sort_values("iteration")
        .groupby(["method", "batch_size", "seed", "run_id"], as_index=False, dropna=False)
        .tail(1)
    )
    agg = final_rows.groupby(["method", "batch_size", "a", "mu_hat_factor"], as_index=False, dropna=False)[value_col].mean()
    for row_idx, method_base in enumerate(methods):
        method_rows = agg[agg["method"].map(_method_base) == method_base]
        for col_idx, batch_size in enumerate(batch_sizes):
            axis = axes[row_idx][col_idx]
            frame = method_rows[method_rows["batch_size"] == batch_size]
            if frame.empty:
                axis.axis("off")
                continue
            a_values = sorted(frame["a"].dropna().unique())
            factors = sorted(frame["mu_hat_factor"].dropna().unique())
            grid = np.full((len(a_values), len(factors)), np.nan)
            for i, a_value in enumerate(a_values):
                for j, factor in enumerate(factors):
                    match = frame[(frame["a"] == a_value) & (frame["mu_hat_factor"] == factor)]
                    if not match.empty:
                        grid[i, j] = float(match[value_col].iloc[0])
            image = axis.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            axis.set_xticks(range(len(factors)))
            axis.set_xticklabels([f"{factor:g}" for factor in factors])
            axis.set_yticks(range(len(a_values)))
            axis.set_yticklabels([f"{a:g}" for a in a_values])
            axis.set_xlabel("mu_hat_factor")
            axis.set_ylabel("a")
            axis.set_title(f"{METHOD_LABELS.get(method_base, method_base)} | |B|={batch_size}")
            for i in range(len(a_values)):
                for j in range(len(factors)):
                    if np.isfinite(grid[i, j]):
                        axis.text(j, i, f"{grid[i, j]:.2e}" if value_col == "final_objective_gap" else f"{grid[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    _save_pdf(output_path)


TEST4_GROUP_COLORS = {
    "identity_b64": "#7f7f7f",
    "identity_b256": "#4c4c4c",
    "coord_clip_b64": "#d62728",
    "coord_clip_b256": "#ff7f0e",
    "coord_clip_increasing_batch_fixed_clip": "#1f77b4",
    "coord_clip_increasing_batch_relaxed_clip": "#2ca02c",
}

TEST4_GROUP_LABELS = {
    "identity_b64": "Identity, constant |B|=64",
    "identity_b256": "Identity, constant |B|=256",
    "coord_clip_b64": "Coord clip, constant |B|=64",
    "coord_clip_b256": "Coord clip, constant |B|=256",
    "coord_clip_increasing_batch_fixed_clip": "Coord clip, increasing batch, fixed threshold",
    "coord_clip_increasing_batch_relaxed_clip": "Coord clip, increasing batch, relaxed threshold",
}


def _plot_test4_series(df: pd.DataFrame, y_col: str, output_path: Path, title: str, log_y: bool = False) -> None:
    _apply_plot_style()
    plt.figure(figsize=(9.0, 5.2))
    grouped = (
        df.groupby(["run_group", "mu_hat_factor", "iteration"], as_index=False, dropna=False)[y_col]
        .mean()
        .sort_values(["run_group", "mu_hat_factor", "iteration"])
    )
    for (run_group, mu_hat_factor), frame in grouped.groupby(["run_group", "mu_hat_factor"], dropna=False):
        color = TEST4_GROUP_COLORS.get(run_group, COLOR_CYCLE[0])
        linestyle = LINESTYLES.get(float(mu_hat_factor), "-") if pd.notna(mu_hat_factor) else "-"
        label = TEST4_GROUP_LABELS.get(run_group, str(run_group))
        if pd.notna(mu_hat_factor):
            label = f"{label} | mu_hat_factor={float(mu_hat_factor):g}"
        plt.plot(
            frame["iteration"],
            frame[y_col],
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            alpha=0.95,
            label=label,
        )
    for boundary in [500, 1000, 1500]:
        plt.axvline(boundary, color="#bdbdbd", linestyle=":", linewidth=1.0)
    plt.xlabel("iteration")
    plt.ylabel(y_col.replace("_", " "))
    plt.title(title)
    if log_y:
        axis = plt.gca()
        _apply_positive_log_scale(axis, grouped, y_col)
    plt.legend(loc="best", fontsize=8)
    _save_pdf(output_path)


def _plot_test4_floor_vs_batch(summary_df: pd.DataFrame, output_path: Path) -> None:
    constant = summary_df[summary_df["schedule_family"] == "constant_batch"].copy()
    if constant.empty:
        return
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), squeeze=False)
    metrics = [
        ("empirical_floor_L", "Empirical Floor of L_k"),
        ("empirical_floor_G", "Empirical Floor of G_k"),
        ("empirical_floor_V", "Empirical Floor of V_k"),
    ]
    grouped = (
        constant.groupby(["method", "batch_size", "mu_hat_factor"], as_index=False, dropna=False)[
            [metric for metric, _ in metrics]
        ]
        .mean()
    )
    for axis, (metric, title) in zip(axes.flatten(), metrics):
        for (method, mu_hat_factor), frame in grouped.groupby(["method", "mu_hat_factor"], dropna=False):
            method_base = _method_base(method)
            color = METHOD_COLORS.get(method_base, COLOR_CYCLE[0])
            axis.plot(
                frame["batch_size"],
                frame[metric],
                color=color,
                linestyle=LINESTYLES.get(float(mu_hat_factor), "-"),
                marker="o",
                linewidth=2.1,
                label=f"{METHOD_LABELS.get(method_base, method)} | mu_hat_factor={float(mu_hat_factor):g}",
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("batch size")
        axis.set_ylabel(metric)
        axis.set_title(title)
        _apply_positive_log_scale(axis, grouped, metric)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4), bbox_to_anchor=(0.5, 1.03))
    _save_pdf(output_path)


def _plot_test4_floor_vs_residual(summary_df: pd.DataFrame, output_path: Path) -> None:
    valid = summary_df[np.isfinite(summary_df["residual_proxy"]) & np.isfinite(summary_df["empirical_floor_L"])].copy()
    if valid.empty:
        return
    _apply_plot_style()
    plt.figure(figsize=(6.8, 4.8))
    for run_group, frame in valid.groupby("run_group", dropna=False):
        color = TEST4_GROUP_COLORS.get(run_group, COLOR_CYCLE[0])
        plt.scatter(
            frame["residual_proxy"],
            frame["empirical_floor_L"],
            color=color,
            alpha=0.8,
            s=38,
            label=TEST4_GROUP_LABELS.get(run_group, str(run_group)),
        )
    plt.xlabel("residual proxy")
    plt.ylabel("empirical floor L")
    plt.title("Test 4 Empirical Floor vs Residual Proxy")
    plt.legend(loc="best", fontsize=8)
    _save_pdf(output_path)


def save_theory_figures(iteration_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = iteration_df[
        (iteration_df["phase"] == "main")
        & (~iteration_df["outside_theory"])
        & (iteration_df["status"] == "success")
    ]

    test1 = plot_df[plot_df["test_id"] == "test1"]
    if not test1.empty:
        test1_simple = _filter_test1_method_comparison(test1)
        _plot_mean_by_iteration(test1_simple, "lyapunov", figures_dir / "fig_1_mean_lyapunov_vs_iterations.pdf", "Test 1 Mean Lyapunov vs Iterations")
        _plot_mean_by_iteration(test1_simple, "objective_gap", figures_dir / "fig_1_objective_gap_vs_iterations.pdf", "Test 1 Objective Gap vs Iterations")
        _plot_final_vs_batch(test1_simple, "lyapunov", figures_dir / "fig_1_final_lyapunov_vs_batch_size.pdf", "Test 1 Final Lyapunov vs Batch Size")
        _plot_survival(
            test1_simple[test1_simple["method"] == "proxnaggs_identity"],
            "transformed_error_norm",
            "batch_size",
            figures_dir / "fig_1_gradient_error_survival_by_batch_size.pdf",
            "Test 1 Gradient Error Survival by Batch Size",
        )

    for variant in ["test2_mild", "test2_strong"]:
        test2 = plot_df[plot_df["test_id"] == variant]
        if not test2.empty:
            suffix = "mild" if variant.endswith("mild") else "strong"
            test2_simple = _filter_test2_method_comparison(test2)
            _plot_survival(
                test2_simple,
                "raw_error_norm",
                "method",
                figures_dir / f"fig_2_{suffix}_raw_error_survival.pdf",
                f"Test 2 {suffix.capitalize()} Raw Error Survival",
            )
            _plot_survival(
                test2_simple,
                "transformed_error_norm",
                "method",
                figures_dir / f"fig_2_{suffix}_transformed_error_survival.pdf",
                f"Test 2 {suffix.capitalize()} Transformed Error Survival",
            )
            _plot_survival(
                test2_simple,
                "batch_grad_norm",
                "method",
                figures_dir / f"fig_2_{suffix}_batch_grad_survival.pdf",
                f"Test 2 {suffix.capitalize()} Batch Gradient Survival",
            )
            _plot_survival(
                test2_simple,
                "transformed_grad_norm",
                "method",
                figures_dir / f"fig_2_{suffix}_transformed_grad_survival.pdf",
                f"Test 2 {suffix.capitalize()} Transformed Gradient Survival",
            )
            _plot_metric_by_iteration(
                test2_simple,
                "clipping_ratio",
                figures_dir / f"fig_2_{suffix}_clipping_ratio_over_iterations.pdf",
                f"Test 2 {suffix.capitalize()} Clipping Ratio Over Iterations",
                log_y=False,
            )
            _plot_mean_by_iteration(
                test2_simple,
                "objective_gap",
                figures_dir / f"fig_2_{suffix}_objective_gap_heavytail_comparison.pdf",
                f"Test 2 {suffix.capitalize()} Objective Gap Comparison",
            )
            _plot_mean_by_iteration(
                test2_simple,
                "distance_v",
                figures_dir / f"fig_2_{suffix}_distance_to_solution_over_iterations.pdf",
                f"Test 2 {suffix.capitalize()} Distance to Solution",
            )
            _plot_lyapunov_components(
                test2_simple,
                figures_dir / f"fig_2_{suffix}_lyapunov_components.pdf",
                f"Test 2 {suffix.capitalize()} Lyapunov Components",
            )
            if "threshold_quantile" in test2.columns and test2["threshold_quantile"].notna().any():
                _plot_threshold_tradeoff(
                    test2[test2["threshold_quantile"].notna()],
                    figures_dir / f"fig_2_{suffix}_threshold_tradeoff.pdf",
                )
                threshold_methods = test2[
                    test2["method"].map(_method_base).isin(
                        {"proxnaggs_identity", "robust_proxnaggs_coord_clip", "robust_proxnaggs_tanh"}
                    )
                ].copy()
                threshold_summary = _build_run_summary(threshold_methods)
                _plot_threshold_metric_sweep(
                    threshold_summary,
                    "final_objective_gap",
                    figures_dir / f"fig_2_{suffix}_threshold_final_objective_gap_vs_quantile.pdf",
                    f"Test 2 {suffix.capitalize()} Final Objective Gap vs Threshold Quantile",
                    "final objective gap",
                    log_y=True,
                )
                _plot_threshold_metric_sweep(
                    threshold_summary,
                    "final_distance_v",
                    figures_dir / f"fig_2_{suffix}_threshold_final_distance_vs_quantile.pdf",
                    f"Test 2 {suffix.capitalize()} Final Distance vs Threshold Quantile",
                    "final distance",
                    log_y=True,
                )
                _plot_threshold_metric_sweep(
                    threshold_summary,
                    "mean_transformed_error_norm",
                    figures_dir / f"fig_2_{suffix}_threshold_mean_transformed_error_vs_quantile.pdf",
                    f"Test 2 {suffix.capitalize()} Mean Transformed Error vs Threshold Quantile",
                    "mean transformed error norm",
                    log_y=True,
                )
                _plot_threshold_metric_sweep(
                    threshold_summary,
                    "max_transformed_grad_norm",
                    figures_dir / f"fig_2_{suffix}_threshold_max_transformed_grad_vs_quantile.pdf",
                    f"Test 2 {suffix.capitalize()} Max Transformed Gradient Norm vs Threshold Quantile",
                    "max transformed grad norm",
                    log_y=True,
                )
                _plot_threshold_metric_sweep(
                    threshold_summary,
                    "mean_deterministic_bias_norm",
                    figures_dir / f"fig_2_{suffix}_threshold_mean_bias_vs_quantile.pdf",
                    f"Test 2 {suffix.capitalize()} Deterministic Bias Norm vs Threshold Quantile",
                    "mean deterministic bias norm",
                    log_y=True,
                )
                _plot_threshold_metric_sweep(
                    threshold_summary,
                    "mean_clipping_ratio",
                    figures_dir / f"fig_2_{suffix}_threshold_mean_clipping_ratio_vs_quantile.pdf",
                    f"Test 2 {suffix.capitalize()} Clipping Ratio vs Threshold Quantile",
                    "mean clipping ratio",
                    log_y=False,
                )

    test3 = plot_df[plot_df["test_id"] == "test3"]
    if not test3.empty:
        test3_simple = _filter_test3_method_comparison(test3)
        _plot_mean_by_iteration(test3_simple, "objective_gap", figures_dir / "fig_3_lasso_objective_gap.pdf", "Test 3 Lasso Objective Gap")
        _plot_mean_by_iteration(test3_simple, "distance_v", figures_dir / "fig_3_lasso_distance_to_solution.pdf", "Test 3 Lasso Distance to Solution")
        _plot_mean_by_iteration(test3_simple, "support_recovery", figures_dir / "fig_3_lasso_support_recovery.pdf", "Test 3 Lasso Support Recovery")
        _plot_mean_by_iteration(test3_simple, "sparsity", figures_dir / "fig_3_lasso_sparsity_vs_iterations.pdf", "Test 3 Lasso Sparsity vs Iterations")
        _plot_best_runs_by_metric(
            test3,
            "objective_gap",
            figures_dir / "fig_3_best_by_objective_gap.pdf",
            "Test 3 Best Tuned Trajectories by Final Objective Gap",
        )
        _plot_best_runs_by_metric(
            test3,
            "distance_v",
            figures_dir / "fig_3_best_by_distance_v.pdf",
            "Test 3 Best Tuned Trajectories by Final Distance to Solution",
        )
        summary_like = (
            test3.sort_values("iteration")
            .groupby(["method", "batch_size", "seed", "run_id"], as_index=False, dropna=False)
            .tail(1)
            .rename(
                columns={
                    "objective_gap": "final_objective_gap",
                    "distance_v": "final_distance_v",
                    "sparsity": "final_sparsity",
                    "support_recovery": "final_support_recovery",
                }
            )
        )
        _plot_heatmap_grid(
            summary_like,
            "final_objective_gap",
            figures_dir / "fig_3_heatmap_final_objective_gap.pdf",
            "Test 3 Heatmap: Final Objective Gap",
        )
        _plot_heatmap_grid(
            summary_like,
            "final_distance_v",
            figures_dir / "fig_3_heatmap_final_distance_v.pdf",
            "Test 3 Heatmap: Final Distance to Solution",
        )
        _plot_heatmap_grid(
            summary_like,
            "final_sparsity",
            figures_dir / "fig_3_heatmap_final_sparsity.pdf",
            "Test 3 Heatmap: Final Sparsity",
        )
        _plot_heatmap_grid(
            summary_like,
            "final_support_recovery",
            figures_dir / "fig_3_heatmap_final_support_recovery.pdf",
            "Test 3 Heatmap: Final Support Recovery",
        )

    test4 = plot_df[plot_df["test_id"] == "test4_floor_vs_theory"]
    if not test4.empty:
        test4_summary = _build_run_summary(test4)
        _plot_test4_series(
            test4,
            "lyapunov",
            figures_dir / "fig_4_lyapunov_vs_iterations_constant_vs_increasing.pdf",
            "Test 4 Lyapunov vs Iterations",
            log_y=True,
        )
        _plot_test4_series(
            test4,
            "objective_gap",
            figures_dir / "fig_4_objective_gap_vs_iterations_constant_vs_increasing.pdf",
            "Test 4 Objective Gap vs Iterations",
            log_y=True,
        )
        _plot_test4_series(
            test4,
            "transformed_error_sq",
            figures_dir / "fig_4_transformed_error_sq_vs_iterations.pdf",
            "Test 4 Transformed Error Squared vs Iterations",
            log_y=True,
        )
        _plot_test4_floor_vs_batch(
            test4_summary,
            figures_dir / "fig_4_tail_empirical_floor_vs_batch_size.pdf",
        )
        _plot_test4_floor_vs_residual(
            test4_summary,
            figures_dir / "fig_4_empirical_floor_vs_residual_proxy.pdf",
        )
        _plot_test4_series(
            test4,
            "deterministic_bias_norm",
            figures_dir / "fig_4_deterministic_bias_norm_vs_iterations.pdf",
            "Test 4 Deterministic Bias Norm vs Iterations",
            log_y=False,
        )
        _plot_test4_series(
            test4,
            "clipping_ratio",
            figures_dir / "fig_4_clipping_ratio_vs_iterations.pdf",
            "Test 4 Clipping Ratio vs Iterations",
            log_y=False,
        )
