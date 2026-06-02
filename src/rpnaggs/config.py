from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    model_name: str = "small_cifar_cnn"
    data_root: str = "./data"
    output_dir: str = "./robust_proxnaggs_outputs"
    batch_size: int = 128
    num_workers: int = 2
    label_noise_rate: float = 0.20
    use_train_subset: bool = True
    train_subset_size: int = 4096
    epochs: int = 1
    algorithm_name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    robust_map: str = "norm_clip"
    clip_threshold: float = 1.0
    auto_set_threshold: bool = True
    threshold_multiplier: float = 1.5
    pnaggs_a: float = 0.20
    pnaggs_mu_hat: float = 40.0
    use_warmup: bool = False
    warmup_fraction: float = 0.05
    warmup_steps: Optional[int] = None
    l1_reg: float = 0.0
    prox_name: str = "none"
    prox_lam: float = 0.0
    prox_l1: float = 0.0
    prox_l2: float = 0.0
    prox_group: float = 0.0
    prox_lower: float = -1.0
    prox_upper: float = 1.0
    prox_target: str = "all"
    run_diagnostics: bool = True
    ref_size: int = 512
    n_probe_batches: int = 20
    n_coord_probe: int = 128
    seed: int = 123

    @property
    def optimizer_name(self) -> str:
        return self.algorithm_name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local robust Prox-NAG-GS experiment.")
    parser.add_argument("--algorithm", default="adamw", choices=["adamw", "sgd", "clipped_sgd", "robust_proxnaggs"])
    parser.add_argument("--optimizer", dest="algorithm", help="Alias for --algorithm.")
    parser.add_argument("--model", default="small_cifar_cnn")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--robust-map", default="norm_clip", choices=["identity", "norm_clip", "coord_clip", "tanh"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--label-noise-rate", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--clip-threshold", type=float, default=1.0)
    parser.add_argument("--threshold-multiplier", type=float, default=1.5)
    parser.add_argument("--pnaggs-a", type=float, default=0.20)
    parser.add_argument("--pnaggs-mu-hat", type=float, default=40.0)
    parser.add_argument("--pnaggs-eta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--use-warmup", action="store_true", help="Enable linear warmup for the effective step size.")
    parser.add_argument("--warmup-fraction", type=float, default=0.05, help="Warmup length as a fraction of total training steps.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Explicit number of warmup steps. Overrides --warmup-fraction when set.")
    parser.add_argument("--l1-reg", type=float, default=0.0)
    parser.add_argument(
        "--prox-name",
        default="none",
        choices=["none", "l1", "l2", "elastic_net", "group_lasso", "sparse_group_lasso", "nonnegative", "box"],
        help="Proximal operator applied in robust Prox-NAG-GS.",
    )
    parser.add_argument("--prox-lam", type=float, default=0.0, help="Shared lambda parameter for l1 or l2 prox.")
    parser.add_argument("--prox-l1", type=float, default=0.0, help="L1 coefficient for elastic-net or sparse-group-lasso prox.")
    parser.add_argument("--prox-l2", type=float, default=0.0, help="L2 coefficient for elastic-net prox.")
    parser.add_argument("--prox-group", type=float, default=0.0, help="Group coefficient for group-lasso style prox.")
    parser.add_argument("--prox-lower", type=float, default=-1.0, help="Lower bound used by the box prox.")
    parser.add_argument("--prox-upper", type=float, default=1.0, help="Upper bound used by the box prox.")
    parser.add_argument(
        "--prox-target",
        default="all",
        choices=["all", "weights_only", "classifier", "conv"],
        help="Subset of parameters to which the prox operator is applied.",
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./robust_proxnaggs_outputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ref-size", type=int, default=512)
    parser.add_argument("--n-probe-batches", type=int, default=20)
    parser.add_argument("--n-coord-probe", type=int, default=128)
    parser.add_argument("--train-subset-size", type=int, default=4096)
    parser.add_argument("--full-train", action="store_true", help="Use the full training set instead of a subset.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip heavy-tail diagnostics to reduce runtime.")
    parser.add_argument(
        "--no-auto-threshold",
        action="store_true",
        help="Use the provided clip threshold directly instead of estimating it from a reference gradient.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    pnaggs_mu_hat = args.pnaggs_mu_hat
    if args.pnaggs_eta is not None:
        pnaggs_mu_hat = args.pnaggs_a / args.pnaggs_eta
    return ExperimentConfig(
        dataset=args.dataset,
        model_name=args.model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_noise_rate=args.label_noise_rate,
        use_train_subset=not args.full_train,
        train_subset_size=args.train_subset_size,
        epochs=args.epochs,
        algorithm_name=args.algorithm,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        robust_map=args.robust_map,
        clip_threshold=args.clip_threshold,
        auto_set_threshold=not args.no_auto_threshold,
        threshold_multiplier=args.threshold_multiplier,
        pnaggs_a=args.pnaggs_a,
        pnaggs_mu_hat=pnaggs_mu_hat,
        use_warmup=args.use_warmup,
        warmup_fraction=args.warmup_fraction,
        warmup_steps=args.warmup_steps,
        l1_reg=args.l1_reg,
        prox_name=args.prox_name,
        prox_lam=args.prox_lam,
        prox_l1=args.prox_l1,
        prox_l2=args.prox_l2,
        prox_group=args.prox_group,
        prox_lower=args.prox_lower,
        prox_upper=args.prox_upper,
        prox_target=args.prox_target,
        run_diagnostics=not args.no_diagnostics,
        ref_size=args.ref_size,
        n_probe_batches=args.n_probe_batches,
        n_coord_probe=args.n_coord_probe,
        seed=args.seed,
    )
