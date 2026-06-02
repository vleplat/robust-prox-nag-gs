from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


EPS = 1e-12


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    if series.empty:
        return series
    min_value = float(series.min())
    max_value = float(series.max())
    if abs(max_value - min_value) < EPS:
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    if higher_is_better:
        return (series - min_value) / (max_value - min_value)
    return (max_value - series) / (max_value - min_value)


def _band_quality(series: pd.Series, lower: float, upper: float) -> pd.Series:
    quality = []
    for value in series:
        value = float(value)
        if lower <= value <= upper:
            quality.append(1.0)
        elif value < lower:
            quality.append(max(0.0, value / max(lower, EPS)))
        else:
            quality.append(max(0.0, upper / max(value, EPS)))
    return pd.Series(quality, index=series.index, dtype=float)


def _stage_weights(stage_name: str) -> Dict[str, float]:
    if stage_name == "stage2":
        return {
            "val_acc_score": 0.40,
            "val_loss_score": 0.15,
            "xv_score": 0.10,
            "grad_score": 0.05,
            "epochs_score": 0.05,
            "clip_ratio_score": 0.10,
            "retention_score": 0.10,
            "coord_clip_activity_score": 0.05,
        }
    return {
        "val_acc_score": 0.55,
        "val_loss_score": 0.15,
        "xv_score": 0.15,
        "grad_score": 0.10,
        "epochs_score": 0.05,
    }


def _prepare_stage_summary(summary_df: pd.DataFrame, stage_name: str, max_epochs: int) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    df = summary_df.copy()
    df["stage"] = stage_name
    df["retention_ratio"] = df["mean_transformed_gradient_norm"] / (df["mean_gradient_norm"] + EPS)
    df["epochs_fraction"] = df["epochs_completed"] / max(max_epochs, 1)

    viable = df[df["diverged"] == False].copy()
    if viable.empty:
        viable = df.copy()
    viable = viable.reset_index(drop=True)

    viable["val_acc_score"] = _normalize(viable["best_val_acc"], higher_is_better=True)
    viable["val_loss_score"] = _normalize(viable["final_val_loss"], higher_is_better=False)
    viable["xv_score"] = _normalize(viable["mean_x_v_distance"], higher_is_better=False)
    viable["grad_score"] = _normalize(viable["mean_gradient_norm"], higher_is_better=False)
    viable["epochs_score"] = _normalize(viable["epochs_fraction"], higher_is_better=True)

    if stage_name == "stage2":
        viable["clip_ratio_score"] = _band_quality(viable["mean_clipping_ratio"], lower=0.05, upper=0.50)
        viable["retention_score"] = _band_quality(viable["retention_ratio"], lower=0.25, upper=0.95)
        viable["coord_clip_activity_score"] = 0.5
        coord_like = viable["robust_map"].isin(["coord_clip", "tanh"])
        if coord_like.any():
            viable.loc[coord_like, "coord_clip_activity_score"] = _band_quality(
                viable.loc[coord_like, "mean_coordinate_clipping_ratio"],
                lower=0.05,
                upper=0.20,
            )

    weights = _stage_weights(stage_name)
    viable["selection_score"] = 0.0
    for score_name, weight in weights.items():
        viable["selection_score"] += weight * viable[score_name]

    viable = viable.sort_values(
        ["selection_score", "best_val_acc", "final_val_acc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    viable["rank_within_stage"] = range(1, len(viable) + 1)
    return viable


def _make_command(row: pd.Series, manifest: Dict) -> str:
    base_config = manifest.get("base_config", {})
    parts = [
        "rpnaggs-run",
        "--algorithm robust_proxnaggs",
        f"--model {base_config.get('model_name', 'small_cifar_cnn')}",
        f"--robust-map {row['robust_map']}",
        f"--pnaggs-a {row['a']}",
        f"--pnaggs-mu-hat {row['mu_hat']}",
        f"--label-noise-rate {base_config.get('label_noise_rate', 0.2)}",
        f"--epochs {base_config.get('epochs', 5)}",
    ]
    if bool(base_config.get("use_train_subset", False)):
        parts.append(f"--train-subset-size {base_config.get('train_subset_size', 4096)}")
    else:
        parts.append("--full-train")
    if float(row.get("clip_threshold", 0) or 0) > 0 and str(row["robust_map"]) != "identity":
        parts.append(f"--clip-threshold {row['clip_threshold']}")
        parts.append("--no-auto-threshold")
    if bool(base_config.get("use_warmup", False)):
        parts.append("--use-warmup")
        parts.append(f"--warmup-fraction {base_config.get('warmup_fraction', 0.05)}")
        if base_config.get("warmup_steps") is not None:
            parts.append(f"--warmup-steps {base_config.get('warmup_steps')}")
    return " \\\n  ".join(parts)


def _explanation(row: pd.Series) -> str:
    reasons = [
        f"best_val_acc={row['best_val_acc']:.4f}",
        f"final_val_loss={row['final_val_loss']:.4f}",
        f"x_v_distance={row['mean_x_v_distance']:.4f}",
    ]
    if row["stage"] == "stage2":
        reasons.append(f"clip_ratio={row['mean_clipping_ratio']:.4f}")
        reasons.append(f"retention={row['retention_ratio']:.4f}")
        if pd.notna(row.get("mean_coordinate_clipping_ratio", float("nan"))):
            reasons.append(f"coord_clip_ratio={row['mean_coordinate_clipping_ratio']:.4f}")
    if bool(row["diverged"]):
        reasons.append("diverged")
    else:
        reasons.append("stable")
    return ", ".join(reasons)


def _build_recommendations(
    stage1_ranked: pd.DataFrame,
    stage2_ranked: pd.DataFrame,
    stage3_ranked: pd.DataFrame,
    manifest: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    recommendations: List[Dict] = []

    if not stage2_ranked.empty:
        top_stage2 = stage2_ranked.head(3).reset_index(drop=True)
        roles = ["primary_robust", "backup_robust_1", "backup_robust_2"]
        for idx, (_, row) in enumerate(top_stage2.iterrows()):
            role = roles[idx] if idx < len(roles) else f"backup_robust_{idx}"
            recommendations.append(
                {
                    "role": role,
                    "stage": row["stage"],
                    "run_name": row["run_name"],
                    "selection_score": row["selection_score"],
                    "a": row["a"],
                    "mu_hat": row["mu_hat"],
                    "step_size": row["step_size"],
                    "robust_map": row["robust_map"],
                    "clip_threshold": row.get("clip_threshold"),
                    "threshold_multiplier": row.get("threshold_multiplier"),
                    "best_val_acc": row["best_val_acc"],
                    "final_val_loss": row["final_val_loss"],
                    "mean_x_v_distance": row["mean_x_v_distance"],
                    "mean_clipping_ratio": row.get("mean_clipping_ratio"),
                    "retention_ratio": row.get("retention_ratio"),
                    "diverged": row["diverged"],
                    "why": _explanation(row),
                    "command": _make_command(row, manifest),
                }
            )

    if not stage1_ranked.empty:
        row = stage1_ranked.iloc[0]
        recommendations.append(
            {
                "role": "best_identity_dynamics",
                "stage": row["stage"],
                "run_name": row["run_name"],
                "selection_score": row["selection_score"],
                "a": row["a"],
                "mu_hat": row["mu_hat"],
                "step_size": row["step_size"],
                "robust_map": row["robust_map"],
                "clip_threshold": row.get("clip_threshold"),
                "threshold_multiplier": row.get("threshold_multiplier"),
                "best_val_acc": row["best_val_acc"],
                "final_val_loss": row["final_val_loss"],
                "mean_x_v_distance": row["mean_x_v_distance"],
                "mean_clipping_ratio": row.get("mean_clipping_ratio"),
                "retention_ratio": row.get("retention_ratio"),
                "diverged": row["diverged"],
                "why": _explanation(row),
                "command": _make_command(row, manifest),
            }
        )

    if not stage3_ranked.empty:
        row = stage3_ranked.iloc[0]
        recommendations.append(
            {
                "role": "best_mu_hat_reference",
                "stage": row["stage"],
                "run_name": row["run_name"],
                "selection_score": row["selection_score"],
                "a": row["a"],
                "mu_hat": row["mu_hat"],
                "step_size": row["step_size"],
                "robust_map": row["robust_map"],
                "clip_threshold": row.get("clip_threshold"),
                "threshold_multiplier": row.get("threshold_multiplier"),
                "best_val_acc": row["best_val_acc"],
                "final_val_loss": row["final_val_loss"],
                "mean_x_v_distance": row["mean_x_v_distance"],
                "mean_clipping_ratio": row.get("mean_clipping_ratio"),
                "retention_ratio": row.get("retention_ratio"),
                "diverged": row["diverged"],
                "why": _explanation(row),
                "command": _make_command(row, manifest),
            }
        )

    recommendations_df = pd.DataFrame(recommendations)
    payload = recommendations[0] if recommendations else {}
    return recommendations_df, payload


def _write_markdown_report(
    output_dir: Path,
    stage1_ranked: pd.DataFrame,
    stage2_ranked: pd.DataFrame,
    stage3_ranked: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> None:
    lines = [
        "# Tuning Recommendations",
        "",
        "This report distills the staged tuning outputs into a small set of recommended parameter choices.",
        "",
    ]

    if not recommendations_df.empty:
        lines.extend(
            [
                "## Recommended Configurations",
                "",
            ]
        )
        for _, row in recommendations_df.iterrows():
            lines.extend(
                [
                    f"### {row['role']}",
                    "",
                    f"- Stage: `{row['stage']}`",
                    f"- Run: `{row['run_name']}`",
                    f"- Parameters: `a={row['a']}`, `mu_hat={row['mu_hat']}`, `step_size={row['step_size']}`, `robust_map={row['robust_map']}`",
                    f"- Rationale: {row['why']}",
                    "",
                    "Command:",
                    "",
                    "```bash",
                    row["command"],
                    "```",
                    "",
                ]
            )

    def _dataframe_to_markdown(df: pd.DataFrame) -> List[str]:
        headers = [str(col) for col in df.columns]
        rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
        separator = ["---"] * len(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return lines

    def _top_table(df: pd.DataFrame, title: str) -> None:
        if df.empty:
            return
        lines.append(f"## {title}")
        lines.append("")
        top = df.head(5)[
            [
                "rank_within_stage",
                "run_name",
                "a",
                "mu_hat",
                "step_size",
                "robust_map",
                "best_val_acc",
                "final_val_loss",
                "selection_score",
            ]
        ]
        lines.extend(_dataframe_to_markdown(top))
        lines.append("")

    _top_table(stage1_ranked, "Top Stage 1 Runs")
    _top_table(stage2_ranked, "Top Stage 2 Runs")
    _top_table(stage3_ranked, "Top Stage 3 Runs")

    (output_dir / "tuning_recommendations.md").write_text("\n".join(lines), encoding="utf-8")


def distill_tuning_results(input_dir: str | Path) -> Dict:
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = input_dir / "tuning_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    max_epochs = int(manifest.get("base_config", {}).get("epochs", 1))

    stage1_summary = _read_csv_if_exists(input_dir / "stage1" / "stage1_summary.csv")
    stage2_summary = _read_csv_if_exists(input_dir / "stage2" / "stage2_summary.csv")
    stage3_summary = _read_csv_if_exists(input_dir / "stage3" / "stage3_summary.csv")

    stage1_ranked = _prepare_stage_summary(stage1_summary, "stage1", max_epochs=max_epochs)
    stage2_ranked = _prepare_stage_summary(stage2_summary, "stage2", max_epochs=max_epochs)
    stage3_ranked = _prepare_stage_summary(stage3_summary, "stage3", max_epochs=max_epochs)

    stage1_ranked.to_csv(input_dir / "stage1_ranked.csv", index=False)
    stage2_ranked.to_csv(input_dir / "stage2_ranked.csv", index=False)
    stage3_ranked.to_csv(input_dir / "stage3_ranked.csv", index=False)

    recommendations_df, primary_payload = _build_recommendations(stage1_ranked, stage2_ranked, stage3_ranked, manifest)
    recommendations_df.to_csv(input_dir / "final_recommendations.csv", index=False)
    with open(input_dir / "recommended_config.json", "w", encoding="utf-8") as fh:
        json.dump(primary_payload, fh, indent=2)

    _write_markdown_report(input_dir, stage1_ranked, stage2_ranked, stage3_ranked, recommendations_df)
    return {
        "stage1_ranked": stage1_ranked,
        "stage2_ranked": stage2_ranked,
        "stage3_ranked": stage3_ranked,
        "recommendations": recommendations_df,
        "primary": primary_payload,
    }
