from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Optional

from rpnaggs.config import ExperimentConfig
from rpnaggs.optim.transforms import flatten_grads_from_params, get_trainable_params, robust_transform_vector


def make_reference_subset_loader(
    train_eval_loader: DataLoader,
    ref_size: int = 512,
    batch_size: int = 128,
    seed: int = 123,
) -> DataLoader:
    dataset = train_eval_loader.dataset
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=min(ref_size, n), replace=False)
    ref_subset = Subset(dataset, indices.tolist())
    return DataLoader(
        ref_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def gradient_vector_average(
    model: nn.Module,
    batches,
    device: torch.device,
    total_samples: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    params = get_trainable_params(model)
    for p in params:
        p.grad = None

    criterion_sum = nn.CrossEntropyLoss(reduction="sum")
    materialized_batches = list(batches)
    if total_samples is None:
        total_samples = sum(x.shape[0] for x, _ in materialized_batches)

    for x, y in materialized_batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion_sum(logits, y) / total_samples
        loss.backward()

    grad = flatten_grads_from_params(params).detach().clone()
    for p in params:
        p.grad = None
    return grad


def gradient_vector_one_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    model.eval()
    params = get_trainable_params(model)
    for p in params:
        p.grad = None
    criterion = nn.CrossEntropyLoss(reduction="mean")
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    grad = flatten_grads_from_params(params).detach().clone()
    for p in params:
        p.grad = None
    return grad, loss.detach().item()


def estimate_threshold_from_reference_gradient(
    model: nn.Module,
    ref_loader: DataLoader,
    robust_map: str,
    multiplier: float,
    device: torch.device,
) -> float:
    g_ref = gradient_vector_average(model, ref_loader, device=device)
    if robust_map == "coord_clip":
        base = torch.max(torch.abs(g_ref)).item()
    else:
        base = torch.linalg.norm(g_ref).item()
    return multiplier * max(base, 1e-8)


def collect_gradient_error_statistics(
    model: nn.Module,
    train_eval_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    tag: str,
):
    ref_loader = make_reference_subset_loader(
        train_eval_loader,
        ref_size=config.ref_size,
        batch_size=config.batch_size,
        seed=config.seed + 17,
    )
    ref_batches = list(ref_loader)
    total_ref = sum(x.shape[0] for x, _ in ref_batches)
    print(f"Computing reference gradient on {total_ref} samples...")
    g_ref = gradient_vector_average(model, ref_batches, device=device, total_samples=total_ref)

    d = g_ref.numel()
    rng = np.random.default_rng(config.seed + 99)
    coord_idx = torch.tensor(
        rng.choice(d, size=min(config.n_coord_probe, d), replace=False),
        device=g_ref.device,
        dtype=torch.long,
    )

    probe_loader = DataLoader(
        train_eval_loader.dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    raw_norms = []
    transformed_norms = []
    batch_losses = []
    coord_errors = []

    iterator = iter(probe_loader)
    for _ in range(config.n_probe_batches):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(probe_loader)
            x, y = next(iterator)

        g_b, loss_value = gradient_vector_one_batch(model, x, y, device=device)
        e = g_b - g_ref
        d_b = robust_transform_vector(g_b, config.robust_map, config.clip_threshold)
        e_trans = d_b - g_ref

        raw_norms.append(torch.linalg.norm(e).item())
        transformed_norms.append(torch.linalg.norm(e_trans).item())
        batch_losses.append(loss_value)
        coord_errors.append(e[coord_idx].detach().cpu().numpy())

    return {
        "tag": tag,
        "g_ref_norm": torch.linalg.norm(g_ref).item(),
        "g_ref_inf": torch.max(torch.abs(g_ref)).item(),
        "raw_norms": np.array(raw_norms),
        "transformed_norms": np.array(transformed_norms),
        "batch_losses": np.array(batch_losses),
        "coord_errors": np.concatenate(coord_errors, axis=0),
    }
