from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rpnaggs.diagnostics.tails import save_history_comparison_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple training-history CSV files and save PDF figures.")
    parser.add_argument("history_csvs", nargs="+", help="Paths to history_*.csv files.")
    parser.add_argument("--labels", nargs="*", help="Optional custom labels, one per history CSV.")
    parser.add_argument("--output-dir", default="./robust_proxnaggs_outputs", help="Directory where comparison PDFs are saved.")
    parser.add_argument("--tag", default="multi_run", help="Suffix used in output filenames.")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.labels is not None and len(args.labels) not in {0, len(args.history_csvs)}:
        raise ValueError("If provided, --labels must have the same number of entries as history_csvs.")

    history_frames = [pd.read_csv(path) for path in args.history_csvs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels if args.labels else None

    save_history_comparison_artifacts(history_frames, output_dir, args.tag, run_labels=labels)
    print(f"Saved comparison figures to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
