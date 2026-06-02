from __future__ import annotations

import argparse
from pathlib import Path

from rpnaggs.tuning_distill import distill_tuning_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill staged tuning outputs into a compact recommendation report.")
    parser.add_argument(
        "--input-dir",
        default="./robust_proxnaggs_outputs/tuning_robust_proxnaggs",
        help="Directory containing the staged tuning outputs.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = distill_tuning_results(Path(args.input_dir))
    recommendations = result["recommendations"]
    print(f"Distilled tuning results in {Path(args.input_dir).resolve()}")
    if not recommendations.empty:
        top = recommendations.iloc[0]
        print(
            f"Primary recommendation: a={top['a']}, mu_hat={top['mu_hat']}, "
            f"step_size={top['step_size']}, robust_map={top['robust_map']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
