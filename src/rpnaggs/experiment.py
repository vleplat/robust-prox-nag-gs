"""Backward-compatible entry points for the refactored package."""

from rpnaggs.cli import main
from rpnaggs.config import ExperimentConfig, build_arg_parser, config_from_args
from rpnaggs.experiments.runner import run_experiment

__all__ = ["ExperimentConfig", "build_arg_parser", "config_from_args", "main", "run_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())
