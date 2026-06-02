from __future__ import annotations

import torch
import torch.nn as nn

from rpnaggs.algorithms.baselines import OptimizerAlgorithm
from rpnaggs.algorithms.robust_proxnaggs import RobustProxNAGGSAlgorithm
from rpnaggs.config import ExperimentConfig
from rpnaggs.optim.prox import make_prox_operator


def available_algorithms() -> list[str]:
    return ["adamw", "sgd", "clipped_sgd", "robust_proxnaggs"]


def build_algorithm(model: nn.Module, config: ExperimentConfig, total_training_steps=None):
    name = config.algorithm_name.lower()

    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        return OptimizerAlgorithm(name="adamw", optimizer=optimizer)

    if name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        return OptimizerAlgorithm(name="sgd", optimizer=optimizer)

    if name == "clipped_sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        return OptimizerAlgorithm(
            name="clipped_sgd",
            optimizer=optimizer,
            grad_clip_mode=config.robust_map,
            grad_clip_threshold=config.clip_threshold,
        )

    if name == "robust_proxnaggs":
        prox_operator = make_prox_operator(
            name=config.prox_name,
            lam=config.prox_lam,
            l1=config.prox_l1,
            l2=config.prox_l2,
            group=config.prox_group,
            lower=config.prox_lower,
            upper=config.prox_upper,
        )
        if config.l1_reg > 0 and config.prox_name == "none":
            prox_operator = make_prox_operator("l1", lam=config.l1_reg)
        return RobustProxNAGGSAlgorithm(
            model=model,
            a=config.pnaggs_a,
            mu_hat=config.pnaggs_mu_hat,
            robust_map=config.robust_map,
            threshold=config.clip_threshold,
            prox_operator=prox_operator,
            prox_target=config.prox_target,
            use_warmup=config.use_warmup,
            warmup_fraction=config.warmup_fraction,
            warmup_steps=config.warmup_steps,
            total_training_steps=total_training_steps,
        )

    raise ValueError(f"Unsupported algorithm: {config.algorithm_name}")
