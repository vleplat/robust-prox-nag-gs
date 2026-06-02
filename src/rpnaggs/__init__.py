"""Robust Prox-NAG-GS research library."""

from rpnaggs.algorithms.registry import available_algorithms, build_algorithm
from rpnaggs.config import ExperimentConfig


def run_experiment(*args, **kwargs):
    from rpnaggs.experiments.runner import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)


__all__ = ["ExperimentConfig", "available_algorithms", "build_algorithm", "run_experiment"]
