import math

import torch
import torch.nn as nn

from rpnaggs.algorithms.registry import available_algorithms, build_algorithm
from rpnaggs.config import build_arg_parser, config_from_args


def test_expected_algorithms_are_registered():
    names = set(available_algorithms())
    assert {"adamw", "sgd", "clipped_sgd", "robust_proxnaggs"}.issubset(names)


def test_config_prox_name_builds_expected_operator():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--algorithm",
            "robust_proxnaggs",
            "--prox-name",
            "l2",
            "--prox-lam",
            "3.0",
            "--prox-target",
            "weights_only",
        ]
    )
    config = config_from_args(args)
    algorithm = build_algorithm(nn.Linear(2, 2), config)

    assert algorithm.prox_name == "l2"
    assert algorithm.prox_target == "weights_only"
    assert math.isclose(algorithm.prox_operator.lam, 3.0, rel_tol=1e-9)


def test_legacy_l1_reg_still_builds_l1_prox():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--algorithm",
            "robust_proxnaggs",
            "--l1-reg",
            "0.25",
        ]
    )
    config = config_from_args(args)
    algorithm = build_algorithm(nn.Linear(2, 2), config)

    assert algorithm.prox_name == "l1"
    assert math.isclose(algorithm.prox_operator.lam, 0.25, rel_tol=1e-9)
