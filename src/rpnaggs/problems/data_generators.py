from __future__ import annotations

from typing import Optional, Tuple

import torch


def generate_sparse_ground_truth(
    d: int,
    sparsity: float = 0.2,
    amplitude: float = 1.0,
    seed: int = 123,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x_true = torch.zeros(d, device=device)
    nnz = max(1, int(round(d * sparsity)))
    support = torch.randperm(d, generator=generator, device=device)[:nnz]
    signs = torch.where(
        torch.rand(nnz, generator=generator, device=device) < 0.5,
        -torch.ones(nnz, device=device),
        torch.ones(nnz, device=device),
    )
    x_true[support] = amplitude * signs
    return x_true


def generate_gaussian_linear_data(
    n: int,
    d: int,
    noise_std: float = 0.1,
    seed: int = 123,
    x_true: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if x_true is None:
        x_true = torch.randn(d, generator=generator, device=device)
    A = torch.randn(n, d, generator=generator, device=device)
    noise = noise_std * torch.randn(n, generator=generator, device=device)
    b = A @ x_true + noise
    return A, b, x_true


def generate_student_t_linear_data(
    n: int,
    d: int,
    df: float = 3.0,
    noise_std: float = 0.1,
    seed: int = 123,
    x_true: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if x_true is None:
        x_true = torch.randn(d, generator=generator, device=device)
    student = torch.distributions.StudentT(df=df)
    A = student.sample((n, d)).to(device=device)
    noise = noise_std * torch.randn(n, generator=generator, device=device)
    b = A @ x_true + noise
    return A, b, x_true


def generate_leverage_mixture_linear_data(
    n: int,
    d: int,
    leverage_fraction: float = 0.05,
    leverage_scale: float = 10.0,
    noise_std: float = 0.1,
    seed: int = 123,
    x_true: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if x_true is None:
        x_true = torch.randn(d, generator=generator, device=device)
    A = torch.randn(n, d, generator=generator, device=device)
    mask = torch.rand(n, generator=generator, device=device) < leverage_fraction
    A[mask] = leverage_scale * torch.randn(mask.sum().item(), d, generator=generator, device=device)
    noise = noise_std * torch.randn(n, generator=generator, device=device)
    b = A @ x_true + noise
    return A, b, x_true
