from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


REQUIRED_PREFIXES = [
    "history_",
    "tail_summary_",
    "hill_raw_norms_",
    "hill_coord_errors_",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect the minimal CSV bundle needed for downstream analysis."
    )
    parser.add_argument(
        "history_csvs",
        nargs="*",
        help="Optional explicit list of history_*.csv files. If omitted, all history_*.csv files in --output-dir are used.",
    )
    parser.add_argument(
        "--output-dir",
        default="./robust_proxnaggs_outputs",
        help="Directory containing the run outputs.",
    )
    parser.add_argument(
        "--bundle-name",
        default="analysis_bundle",
        help="Name of the bundle directory created inside --output-dir.",
    )
    parser.add_argument(
        "--include-configs",
        action="store_true",
        help="Also include used_config_*.json files for each selected run.",
    )
    return parser


def _discover_history_files(output_dir: Path, explicit_history_csvs: List[str]) -> List[Path]:
    if explicit_history_csvs:
        history_files = [Path(path).resolve() for path in explicit_history_csvs]
        missing = [str(path) for path in history_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"History CSV files not found: {missing}")
        return history_files
    return sorted(output_dir.glob("history_*.csv"))


def _suffix_from_history(history_path: Path) -> str:
    name = history_path.name
    if not (name.startswith("history_") and name.endswith(".csv")):
        raise ValueError(f"Not a history CSV filename: {history_path}")
    return name[len("history_") : -len(".csv")]


def _collect_related_files(history_files: List[Path], include_configs: bool) -> List[Path]:
    selected: List[Path] = []
    for history_path in history_files:
        if history_path.exists():
            selected.append(history_path)

        suffix = _suffix_from_history(history_path)
        run_dir = history_path.parent

        for prefix in REQUIRED_PREFIXES[1:]:
            candidate = run_dir / f"{prefix}{suffix}.csv"
            if candidate.exists():
                selected.append(candidate)

        if include_configs:
            config_path = run_dir / f"used_config_{suffix}.json"
            if config_path.exists():
                selected.append(config_path)

    unique = []
    seen = set()
    for path in selected:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _write_manifest(bundle_dir: Path, copied_files: List[Path], include_configs: bool) -> None:
    lines = [
        "# Analysis Bundle",
        "",
        "This bundle contains the minimal files typically needed by an analysis agent.",
        "",
        "Included file types:",
        "- `history_*.csv`: train/test loss and accuracy over epochs",
        "- `tail_summary_*.csv`: aggregate tail statistics",
        "- `hill_raw_norms_*.csv`: Hill estimates for gradient-error norms",
        "- `hill_coord_errors_*.csv`: Hill estimates for coordinate-wise errors",
    ]
    if include_configs:
        lines.append("- `used_config_*.json`: exact run configuration")

    lines.extend(
        [
            "",
            "Included files:",
            *[f"- `{path.name}`" for path in copied_files],
            "",
        ]
    )
    (bundle_dir / "manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    bundle_dir = output_dir / args.bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    history_files = _discover_history_files(output_dir, args.history_csvs)
    if not history_files:
        raise FileNotFoundError(f"No history_*.csv files found in {output_dir}")

    selected_files = _collect_related_files(history_files, include_configs=args.include_configs)
    if not selected_files:
        raise FileNotFoundError("No matching analysis files were found for the selected history files.")

    copied_files: List[Path] = []
    for source in selected_files:
        destination = bundle_dir / source.name
        shutil.copy2(source, destination)
        copied_files.append(destination)

    _write_manifest(bundle_dir, copied_files, include_configs=args.include_configs)

    print(f"Created analysis bundle in {bundle_dir}")
    for path in copied_files:
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
