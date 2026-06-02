from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TheoryConfig:
    output_dir: str = "./robust_proxnaggs_outputs/theory_verification"
    tests: List[str] = field(default_factory=lambda: ["1", "3", "2_mild"])
    seeds: List[int] = field(default_factory=lambda: [123])
    iterations: int = 300
    log_every: int = 1
    n: int = 5000
    d_values: List[int] = field(default_factory=lambda: [50])
    mu_reg_values: List[float] = field(default_factory=lambda: [1e-1])
    batch_sizes_test1: List[int] = field(default_factory=lambda: [64, 256])
    batch_sizes_test2: List[int] = field(default_factory=lambda: [64, 256])
    batch_sizes_test3: List[int] = field(default_factory=lambda: [64, 256])
    test2_robust_maps: List[str] = field(default_factory=lambda: ["identity", "coord_clip", "norm_clip", "tanh"])
    heavy_tail_kind: str = "student_t"
    heavy_tail_df: float = 5.0
    heavy_tail_leverage_fraction: float = 0.05
    heavy_tail_leverage_scale: float = 10.0
    threshold_quantiles: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95, 0.99])
    main_threshold_quantile: float = 0.95
    a_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4])
    mu_hat_factors: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0])
    test3_local_a_values: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5])
    test3_local_h_values: List[float] = field(default_factory=lambda: [0.04, 0.05, 0.06, 0.08])
    sgd_lr: float = 0.05
    lasso_lambda: float = 0.05
    lasso_sparsity: float = 0.2
    noise_std: float = 0.1
    deterministic_sanity_a: float = 0.2
    deterministic_sanity_mu_hat_factor: float = 2.0
    calibration_iterations: int = 25
    explosion_objective_gap: float = 1e12
    explosion_distance_v: float = 1e6
    explosion_grad_norm: float = 1e12

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
