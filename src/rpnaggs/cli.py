from __future__ import annotations

from typing import List, Optional

from rpnaggs.config import build_arg_parser, config_from_args


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    from rpnaggs.experiments.runner import run_experiment

    result = run_experiment(config)
    print(f"Finished. Results saved in {result['output_dir']}")
    return 0
