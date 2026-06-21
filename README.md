# Robust Prox-NAG-GS

`Robust Prox-NAG-GS` is a local PyTorch research environment for studying robust stochastic optimization in PyTorch, with CIFAR-10/MNIST training pipelines and controlled theory-verification tests. The repository provides:

- baseline training methods;
- the `robust_proxnaggs` optimizer;
- heavy-tail diagnostics for stochastic gradient errors;
- publication-style PDF figures;
- tools to compare runs and prepare a compact analysis bundle.

The default setup is CPU-first and uses an isolated virtual environment. The codebase is organized so that models, training methods, diagnostics, and experiment orchestration remain separate and extensible.

## Current capabilities

### Training methods

- `adamw`
- `sgd`
- `clipped_sgd`
- `robust_proxnaggs`

### Robust maps

- `identity`
- `norm_clip`
- `coord_clip`
- `tanh`

### Default model

The default training setup is `small_cifar_cnn`, a compact CNN for CIFAR-10:

- default dataset: `cifar10`
- default model: `small_cifar_cnn`

An MNIST path is also available through the registered pair:

- dataset: `mnist`
- model: `vgg7_mini_mnist`

The synthetic strongly convex theory tests are run through a separate entry point, `rpnaggs-theory`, and do not use image models.

## Installation

This project is designed to be installed from the command line in a local virtual environment.

### Requirements

Before installing, make sure you have:

- `git`
- Python `3.9` or newer
- internet access to download Python packages and the CIFAR-10 dataset

Check the tools first:

On Linux/macOS:

```bash
git --version
python3 --version
```

On Windows PowerShell:

```powershell
git --version
py -3 --version
```

### First-time installation from GitHub

If you do not have the repository yet, clone it first.

On Linux/macOS:

```bash
git clone https://github.com/YOUR_USERNAME/robust-prox-nag-gs.git
cd robust-prox-nag-gs
python3 -m venv RPNAGGS
source RPNAGGS/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
git clone https://github.com/vleplat/robust-prox-nag-gs.git
cd robust-prox-nag-gs
py -3 -m venv RPNAGGS
.\RPNAGGS\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the package in editable mode and keeps all dependencies inside `RPNAGGS/` instead of modifying the system Python installation.

### Update an existing local copy

If you already cloned the repository and just want the latest version:

On Linux/macOS:

```bash
cd /path/to/robust-prox-nag-gs
git pull
source RPNAGGS/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
cd C:\path\to\robust-prox-nag-gs
git pull
.\RPNAGGS\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Re-running `python -m pip install -e ".[dev]"` after `git pull` is recommended when the code or dependencies may have changed.

### Leave and reactivate the environment

To leave the virtual environment:

```bash
deactivate
```

To reactivate it later:

On Linux/macOS:

```bash
cd /path/to/robust-prox-nag-gs
source RPNAGGS/bin/activate
```

On Windows PowerShell:

```powershell
cd C:\path\to\robust-prox-nag-gs
.\RPNAGGS\Scripts\Activate.ps1
```

## Quick start

Run a small smoke test:

On Linux/macOS:

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm adamw \
  --robust-map identity \
  --epochs 1 \
  --train-subset-size 2048
```

On Windows PowerShell:

```powershell
.\RPNAGGS\Scripts\Activate.ps1
rpnaggs-run `
  --algorithm adamw `
  --robust-map identity `
  --epochs 1 `
  --train-subset-size 2048
```

This command:

- downloads CIFAR-10 if needed;
- runs one short training job;
- writes outputs to `robust_proxnaggs_outputs/`.

If the smoke test succeeds, the installation is ready and you can move on to the commands in the rest of this README.

## What can you run?

The repository currently has six main entry points:

1. `rpnaggs-run`: run one image-classification experiment (CIFAR-10 or MNIST).
2. `python scripts/run_method_suite.py`: run several CIFAR-10 methods sequentially and optionally generate comparison figures.
3. `python scripts/run_mnist_method_suite.py`: same four-method suite for MNIST + `vgg7_mini_mnist` with MNIST-tuned hyperparameters.
4. `rpnaggs-compare`: compare existing `history_*.csv` files and regenerate multi-run comparison PDFs.
5. `rpnaggs-bundle`: collect the minimal CSV bundle for downstream analysis.
6. `rpnaggs-optuna`: tune the training methods with Optuna under a shared budget.
7. `rpnaggs-theory`: run the controlled convex theory-verification experiments.

Use:

```bash
rpnaggs-run --help
rpnaggs-compare --help
rpnaggs-bundle --help
rpnaggs-optuna --help
rpnaggs-theory --help
python scripts/run_method_suite.py --help
python scripts/run_mnist_method_suite.py --help
```

Use `rpnaggs-run` for CIFAR-10 or MNIST training, and use `rpnaggs-theory` for the synthetic strongly convex tests. These are separate pipelines with separate outputs.

## Single-run experiments

The main entry point is:

```bash
rpnaggs-run [options]
```

Display the full command reference:

```bash
rpnaggs-run --help
```

Each `rpnaggs-run` execution writes one training history CSV, training-curve PDFs, optional heavy-tail diagnostics, and the exact resolved configuration JSON into `--output-dir`.

### Important options

- `--algorithm`: training method (`adamw`, `sgd`, `clipped_sgd`, `robust_proxnaggs`)
- `--model`: model name; current default is `small_cifar_cnn`
- `--dataset`: dataset name (`cifar10` or `mnist`; pair with a registered `--model`)
- `--robust-map`: robust transformation (`identity`, `norm_clip`, `coord_clip`, `tanh`)
- `--epochs`: number of epochs
- `--batch-size`: mini-batch size
- `--full-train`: use the full CIFAR-10 training set
- `--train-subset-size`: use a subset of the training set
- `--label-noise-rate`: inject symmetric label noise
- `--num-workers`: data loader worker count
- `--no-diagnostics`: disable heavy-tail diagnostics
- `--pnaggs-a`: coupling parameter for `robust_proxnaggs`
- `--pnaggs-mu-hat`: algorithmic curvature parameter for `robust_proxnaggs`
- effective step size: `step_size = a / mu_hat`
- `--use-warmup`: enable linear warmup of the effective step size
- `--warmup-fraction`: warmup length as a fraction of total training steps
- `--warmup-steps`: explicit warmup length; overrides `--warmup-fraction`
- `--prox-name`: proximal operator used by `robust_proxnaggs`
- `--prox-target`: which parameters receive the prox operator
- `--prox-lam`, `--prox-l1`, `--prox-l2`, `--prox-group`: proximal regularization strengths
- `--prox-lower`, `--prox-upper`: bounds used by the box projection
- `--output-dir`: output directory

### Proximal operators

The `robust_proxnaggs` implementation now uses a modular proximal system, and it is exposed through the public experiment config and CLI. The prox operators live in:

```text
src/rpnaggs/optim/prox.py
```

The current default is:

- prox operator: `none`
- regularizer: `r(x) = 0`
- practical effect: the update reduces to the robust gradient step with no additional proximal regularization

The robust update uses the same step size in both places:

```text
step_size = a / mu_hat
```

This `step_size` is used:

- for the robust gradient step;
- for the proximal operator.

### Available prox operators

Use `--prox-name` with one of:

- `none`
- `l1`
- `l2`
- `elastic_net`
- `group_lasso`
- `sparse_group_lasso`
- `nonnegative`
- `box`

### Available prox targets

`--prox-target` accepts **only the fixed keywords below** — not arbitrary PyTorch layer
names (for example, there is no `--prox-target fc2`).

| `--prox-target` | What gets the prox |
|-----------------|-------------------|
| `all` | Every trainable parameter |
| `weights_only` | All tensors with `ndim >= 2` (weights, not biases) |
| `classifier` | Last `nn.Linear` in the model (weight **and** bias) |
| `conv` | All `nn.Conv2d` weight tensors |

Notes:

- **`classifier`** is the usual choice for L1 on the **final FC / logits layer** (e.g. `10`-class head on CIFAR or MNIST). It does **not** apply to earlier MLP layers in `vgg7_mini_mnist`.
- **`weights_only`** applies prox to weights across the whole model, but skips all bias vectors.
- Proximal regularization is applied only when using **`--algorithm robust_proxnaggs`** (not AdamW / SGD / clipped SGD).

To inspect which parameter tensors a target selects:

```bash
source RPNAGGS/bin/activate
python - <<'EOF'
from rpnaggs.config import ExperimentConfig
from rpnaggs.models.registry import build_model
from rpnaggs.algorithms.robust_proxnaggs import _find_target_param_indices
from rpnaggs.optim.transforms import get_trainable_params

cfg = ExperimentConfig(dataset="mnist", model_name="vgg7_mini_mnist")
model = build_model(cfg)
params = get_trainable_params(model)
for target in ["all", "weights_only", "classifier", "conv"]:
    idxs = _find_target_param_indices(model, params, target)
    print(f"\nprox-target={target}:")
    for i in idxs:
        print(f"  [{i}] shape={tuple(params[i].shape)}")
EOF
```

### Parameter meaning

- `--prox-lam`: shared scalar used by `l1` and `l2`
- `--prox-l1`: L1 coefficient used by `elastic_net` and `sparse_group_lasso`
- `--prox-l2`: L2 coefficient used by `elastic_net`
- `--prox-group`: group coefficient used by `group_lasso` and `sparse_group_lasso`
- `--prox-lower`, `--prox-upper`: lower and upper box limits used by `box`

### Backward compatibility

The legacy `--l1-reg` option is still accepted. If `--l1-reg > 0` and `--prox-name none`, the code automatically uses the modular `l1` prox internally.

### Practical examples

No proximal regularization, which is also the current default:

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm robust_proxnaggs \
  --robust-map tanh \
  --pnaggs-a 0.4 \
  --pnaggs-mu-hat 1.0 \
  --clip-threshold 0.02 \
  --no-auto-threshold \
  --prox-name none \
  --prox-target all \
  --epochs 5 \
  --full-train
```

Lasso only on weights, not on biases:

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm robust_proxnaggs \
  --robust-map tanh \
  --pnaggs-a 0.4 \
  --pnaggs-mu-hat 1.0 \
  --clip-threshold 0.02 \
  --no-auto-threshold \
  --prox-name l1 \
  --prox-lam 1e-5 \
  --prox-target weights_only \
  --epochs 5 \
  --full-train
```

L1 (Lasso) on the final FC layer only (`--prox-target classifier`):

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm robust_proxnaggs \
  --robust-map tanh \
  --pnaggs-a 0.4 \
  --pnaggs-mu-hat 1.0 \
  --clip-threshold 0.02 \
  --no-auto-threshold \
  --prox-name l1 \
  --prox-lam 1e-5 \
  --prox-target classifier \
  --epochs 5 \
  --full-train
```

Elastic net on the classifier only:

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm robust_proxnaggs \
  --robust-map tanh \
  --pnaggs-a 0.4 \
  --pnaggs-mu-hat 1.0 \
  --clip-threshold 0.02 \
  --no-auto-threshold \
  --prox-name elastic_net \
  --prox-l1 1e-5 \
  --prox-l2 1e-5 \
  --prox-target classifier \
  --epochs 5 \
  --full-train
```

Nonnegativity projection on convolutional weights:

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm robust_proxnaggs \
  --robust-map tanh \
  --pnaggs-a 0.4 \
  --pnaggs-mu-hat 1.0 \
  --clip-threshold 0.02 \
  --no-auto-threshold \
  --prox-name nonnegative \
  --prox-target conv \
  --epochs 5 \
  --full-train
```

### What is logged

When running `robust_proxnaggs`, the history and optimizer diagnostics now include:

- `a`
- `mu_hat`
- `step_size`
- `robust_map`
- `threshold`
- `prox_name`
- `prox_target`
- `data_loss`
- `regularization_penalty`
- `total_objective`
- `gradient_norm`
- `transformed_gradient_norm`
- `x-v` distance
- clipping statistics

### Recommended baseline run

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm adamw \
  --model small_cifar_cnn \
  --robust-map identity \
  --label-noise-rate 0.2 \
  --epochs 5 \
  --full-train
```

### Recommended robust Prox-NAG-GS run (CIFAR polish winner)

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm robust_proxnaggs \
  --model small_cifar_cnn \
  --robust-map tanh \
  --label-noise-rate 0.2 \
  --pnaggs-a 0.4 \
  --pnaggs-mu-hat 0.625 \
  --clip-threshold 0.06 \
  --no-auto-threshold \
  --seed 55 \
  --epochs 5 \
  --full-train
```

## Sequential benchmark suite

To run the currently available methods sequentially and build comparison plots automatically:

```bash
source RPNAGGS/bin/activate
python scripts/run_method_suite.py \
  --model small_cifar_cnn \
  --epochs 5 \
  --batch-size 128 \
  --label-noise-rate 0.2 \
  --full-train \
  --compare-tag first_full_comparison
```

By default, `AdamW`, `SGD`, and `clipped_sgd` use the best hyperparameters from
`robust_proxnaggs_outputs/optuna_first_pass/` (written by `rpnaggs-optuna` when that
folder exists). The script reloads `adamw/best_trial.json`, `sgd/best_trial.json`, and
`clipped_sgd/best_trial.json` on startup and prints the values it will use. Override
with `--no-optuna` and explicit flags such as `--adamw-lr`, or point to another study
with `--optuna-dir`.

First-pass competitor settings (from the 10-trial / 3-epoch / 10k-subset study):

| Method | `lr` | `momentum` | `weight_decay` | clipping |
|--------|------|------------|----------------|----------|
| AdamW | `0.00103` | — | `0.00635` | `identity` |
| SGD | `0.108` | `0.643` | `6.1e-6` | `identity` |
| clipped SGD | `0.0832` | `0.824` | `1.3e-6` | `coord_clip`, threshold `0.02` |

By default, `robust_proxnaggs` uses the best hyperparameters from
`robust_proxnaggs_outputs/optuna_rproxnaggs_polish/` when that folder exists (same
reload pattern as the competitors). Baked-in fallbacks match the polish study:

- `robust_map = tanh`
- `a = 0.4`
- `mu_hat = 0.625`
- `clip_threshold = 0.06`
- `warmup_fraction = 0.0` (no warmup)
- `--no-auto-threshold`

Previous deep-study defaults (`optuna_rproxnaggs_deep`: `mu_hat = 0.75`, `clip_threshold = 0.05`, 53.62% val acc) remain commented in `scripts/run_method_suite.py` for reference.

Override with `--no-optuna-robust`, `--optuna-robust-dir`, or explicit flags such as
`--pnaggs-mu-hat`.

By default, all four methods use **`--seed 55`** (the seed from the polish study’s best
`robust_proxnaggs` trial). Override with `--seed` or per-method flags such as `--adamw-seed`.

The current suite includes:

1. `adamw`
2. `sgd`
3. `clipped_sgd`
4. `robust_proxnaggs`

If the machine has issues with PyTorch worker processes, use:

```bash
source RPNAGGS/bin/activate
python scripts/run_method_suite.py \
  --model small_cifar_cnn \
  --epochs 5 \
  --batch-size 128 \
  --label-noise-rate 0.2 \
  --full-train \
  --num-workers 0 \
  --compare-tag first_full_comparison
```

## Reference commands for the four current methods

Activate the environment first:

```bash
source RPNAGGS/bin/activate
```

### AdamW

```bash
rpnaggs-run \
  --algorithm adamw \
  --model small_cifar_cnn \
  --robust-map identity \
  --label-noise-rate 0.2 \
  --epochs 5 \
  --full-train
```

### SGD

```bash
rpnaggs-run \
  --algorithm sgd \
  --model small_cifar_cnn \
  --robust-map identity \
  --label-noise-rate 0.2 \
  --epochs 5 \
  --full-train
```

### Clipped SGD

The example below uses the same coordinate-clipping choice as the first-pass tuned setting.

```bash
rpnaggs-run \
  --algorithm clipped_sgd \
  --model small_cifar_cnn \
  --robust-map coord_clip \
  --clip-threshold 0.02 \
  --no-auto-threshold \
  --label-noise-rate 0.2 \
  --epochs 5 \
  --full-train
```

### Robust Prox-NAG-GS

Same hyperparameters as the polish Optuna winner (`optuna_rproxnaggs_polish`):

```bash
rpnaggs-run \
  --algorithm robust_proxnaggs \
  --model small_cifar_cnn \
  --robust-map tanh \
  --label-noise-rate 0.2 \
  --pnaggs-a 0.4 \
  --pnaggs-mu-hat 0.625 \
  --clip-threshold 0.06 \
  --no-auto-threshold \
  --seed 55 \
  --epochs 5 \
  --full-train
```

## Run comparison

If several `history_*.csv` files already exist, compare them with:

```bash
source RPNAGGS/bin/activate
rpnaggs-compare \
  robust_proxnaggs_outputs/history_adamw_identity.csv \
  robust_proxnaggs_outputs/history_sgd_identity.csv \
  robust_proxnaggs_outputs/history_clipped_sgd_coord_clip.csv \
  robust_proxnaggs_outputs/history_robust_proxnaggs_tanh.csv \
  --labels "AdamW" "SGD" "Clipped SGD" "Robust Prox-NAG-GS" \
  --tag first_full_comparison
```

Display the full comparison command reference:

```bash
rpnaggs-compare --help
```

## Analysis bundle

After a suite run, prepare the minimal upload-ready bundle for an analysis agent:

```bash
source RPNAGGS/bin/activate
rpnaggs-bundle
```

This creates:

```text
robust_proxnaggs_outputs/analysis_bundle/
```

The bundle contains the core CSV files needed for downstream analysis:

- `history_*.csv`
- `tail_summary_*.csv`
- `hill_raw_norms_*.csv`
- `hill_coord_errors_*.csv`

To include exact run configurations as well:

```bash
source RPNAGGS/bin/activate
rpnaggs-bundle --include-configs
```

To bundle only selected runs:

```bash
source RPNAGGS/bin/activate
rpnaggs-bundle \
  robust_proxnaggs_outputs/history_adamw_identity.csv \
  robust_proxnaggs_outputs/history_robust_proxnaggs_tanh.csv
```

Display the full bundling command reference:

```bash
rpnaggs-bundle --help
```

## Optuna hyperparameter tuning

The repository now includes an Optuna-based tuner to optimize each available method under the same budget.

Primary entry point:

```bash
rpnaggs-optuna [options]
```

The tuner runs one Optuna study per method and stores:

- per-trial metrics in `trials.csv`
- the best trial in `best_trial.json`
- a replayable training command in `best_command.txt`
- a cross-method summary in `summary.csv` and `summary.md`

### Recommended first Optuna run

```bash
source RPNAGGS/bin/activate
rpnaggs-optuna \
  --methods adamw sgd clipped_sgd robust_proxnaggs \
  --trials 10 \
  --epochs 3 \
  --train-subset-size 10000 \
  --batch-size 128 \
  --num-workers 0 \
  --output-dir robust_proxnaggs_outputs/optuna_first_pass
```

This command gives every method:

- the same number of trials;
- the same number of epochs per trial;
- the same dataset subset;
- the same batch size;
- the same random-seed policy.

### Search spaces used by the Optuna tuner

- `adamw`: `lr`, `weight_decay`
- `sgd`: `lr`, `momentum`, `weight_decay`
- `clipped_sgd`: `lr`, `momentum`, `weight_decay`, `robust_map`, `clip_threshold`
- `robust_proxnaggs`: `a`, `mu_hat`, `robust_map`, `clip_threshold`, `use_warmup`, `warmup_fraction`

At the moment, the Optuna tuner does not yet scan the new prox choices or prox targets automatically. Prox selection is currently a user-controlled experimental choice through `rpnaggs-run`.

For focused robust-only studies, the Optuna CLI also supports explicit narrowing of the `robust_proxnaggs` search region through:

- `--rpnaggs-a-values`
- `--rpnaggs-mu-hat-values`
- `--rpnaggs-robust-maps`
- `--rpnaggs-coord-threshold-values`
- `--rpnaggs-norm-threshold-values`
- `--rpnaggs-warmup-fractions`

### Metrics logged for each Optuna trial

- final and best validation proxy accuracy;
- final and best validation proxy loss;
- final train loss;
- divergence flag;
- gradient norm;
- transformed-gradient norm;
- clipping ratio;
- coordinate clipping ratio when applicable;
- `x-v` distance when applicable.

The current implementation uses the run's `test_acc` and `test_loss` as the validation proxy for solver comparison. This keeps the protocol simple and reproducible while remaining fully aligned across methods.

### Optuna on MNIST with `vgg7_mini_mnist`

The training runner already supports the modular pair `dataset=mnist` and `model=vgg7_mini_mnist`
(see [Tutorial: add or change the model](#tutorial-add-or-change-the-model)). To run the **same
Optuna protocol** on this setup, pass those two flags to `rpnaggs-optuna` in addition to the usual
study options. Use a **separate output directory** so CIFAR Optuna artifacts stay untouched, and
**do not reuse CIFAR-tuned hyperparameters** — MNIST and VGG-7-Mini need their own study.

Prerequisites:

- registered model pair: `mnist` / `vgg7_mini_mnist`
- Optuna flags: `--dataset mnist` and `--model vgg7_mini_mnist` (forwarded to each trial and to `best_command.txt`)
- same four methods and search spaces as the CIFAR first pass (`adamw`, `sgd`, `clipped_sgd`, `robust_proxnaggs`)

Recommended first-pass study (all four methods, 10k MNIST subset, same budget as CIFAR):

```bash
source RPNAGGS/bin/activate
rpnaggs-optuna \
  --dataset mnist \
  --model vgg7_mini_mnist \
  --methods adamw sgd clipped_sgd robust_proxnaggs \
  --trials 10 \
  --epochs 3 \
  --train-subset-size 10000 \
  --batch-size 128 \
  --num-workers 0 \
  --output-dir robust_proxnaggs_outputs/optuna_mnist_first_pass
```

Robust Prox-NAG-GS only (longer trials; MNIST is cheap on CPU):

```bash
source RPNAGGS/bin/activate
rpnaggs-optuna \
  --dataset mnist \
  --model vgg7_mini_mnist \
  --methods robust_proxnaggs \
  --trials 20 \
  --epochs 5 \
  --train-subset-size 10000 \
  --batch-size 128 \
  --num-workers 0 \
  --output-dir robust_proxnaggs_outputs/optuna_mnist_rproxnaggs
```

Example polished `robust_proxnaggs`-only pass (tighter grid, after a first MNIST study):

```bash
source RPNAGGS/bin/activate
rpnaggs-optuna \
  --dataset mnist \
  --model vgg7_mini_mnist \
  --methods robust_proxnaggs \
  --trials 20 \
  --epochs 5 \
  --train-subset-size 10000 \
  --batch-size 128 \
  --num-workers 0 \
  --rpnaggs-a-values 0.35 0.40 0.45 \
  --rpnaggs-mu-hat-values 0.625 0.75 0.875 1.0 \
  --rpnaggs-robust-maps tanh coord_clip \
  --rpnaggs-coord-threshold-values 0.04 0.05 0.06 \
  --rpnaggs-warmup-fractions 0.0 0.03 \
  --output-dir robust_proxnaggs_outputs/optuna_mnist_rproxnaggs_polish
```

After tuning, run the four-method MNIST suite. `scripts/run_mnist_method_suite.py` applies
hardcoded winners from `optuna_mnist_first_pass/` (competitors) and
`optuna_mnist_rproxnaggs_deep/` (robust Prox-NAG-GS):

| Method | Source | Key settings |
|--------|--------|--------------|
| AdamW | `optuna_mnist_first_pass` | `lr=0.00146`, `weight_decay=1.46e-5` |
| SGD | `optuna_mnist_first_pass` | `lr=0.108`, `momentum=0.643`, `weight_decay=6.1e-6` |
| clipped SGD | `optuna_mnist_first_pass` | `lr=0.0743`, `momentum=0.550`, `norm_clip`, threshold `1.0` |
| robust_proxnaggs | `optuna_mnist_rproxnaggs_deep` | `a=0.6`, `mu_hat=1.25`, `coord_clip`, threshold `0.03`, warmup `0.1` |

Full MNIST training set, 10 epochs:

```bash
source RPNAGGS/bin/activate
python scripts/run_mnist_method_suite.py \
  --full-train \
  --epochs 10 \
  --batch-size 128 \
  --label-noise-rate 0.2 \
  --num-workers 0 \
  --compare-tag mnist_full_10ep
```

Results land under `robust_proxnaggs_outputs/optuna_mnist_first_pass/` with the same per-method
layout as CIFAR (`trials.csv`, `best_trial.json`, `best_command.txt`, plus root `summary.csv` /
`summary.md`).

## Suggested workflow

For a first heavy-tail analysis before evaluating the robust method, the following sequence is recommended:

1. Run `adamw` on noisy CIFAR-10.
2. Inspect the diagnostic PDF figures and tail CSV files.
3. Run `clipped_sgd` and `robust_proxnaggs` under the same conditions.
4. Compare the resulting training histories.
5. Build the compact analysis bundle for external review.

## Theory Verification Tests

The repository also includes a separate theory-oriented experiment stack for controlled strongly convex finite-sum problems. These tests are distinct from the CIFAR-10 and MNIST training pipelines:

- they do not use CNNs;
- they do not use image classification accuracy;
- they do not add artificial Gaussian gradient noise;
- they are designed to verify the stochastic theory in simple convex settings;
- they save raw iteration-level CSV files and vector PDF figures.

Primary entry point:

```bash
rpnaggs-theory [options]
```

Display the full theory command reference:

```bash
rpnaggs-theory --help
```

### Common finite-sum setup

All theory tests are based on finite-sum objectives. The smooth part has the form

```text
f(x) = (1/n) sum_i f_i(x).
```

At iteration `k`, the code samples a mini-batch

```text
B_k subset {1,...,n}
```

and replaces the full gradient by the mini-batch gradient

```text
g_B(x) = (1/|B_k|) sum_{i in B_k} grad f_i(x).
```

For the least-squares tests, the data are rows `a_i^T` of a matrix `A` and labels `b_i`. The smooth objective is

```text
f(x) = (1/(2n)) ||Ax-b||^2 + (mu_reg/2)||x||^2.
```

The full gradient is

```text
grad f(x) = (1/n) A^T(Ax-b) + mu_reg x.
```

The mini-batch gradient is

```text
g_B(x) = (1/|B_k|) A_B^T(A_B x - b_B) + mu_reg x
       = (1/|B_k|) sum_{i in B_k} a_i(a_i^T x-b_i) + mu_reg x.
```

For robust Prox-NAG-GS, the update does not use `g_B(x)` directly. It uses the transformed gradient

```text
Phi_lambda(g_B(x)).
```

The theory-relevant error is therefore the transformed oracle error

```text
e_tr = Phi_lambda(g_B(x_next)) - grad f(x_next),
```

not only the raw mini-batch error

```text
e_raw = g_B(x_next) - grad f(x_next).
```

Both quantities are logged.

### What is implemented

The theory stack currently includes four main tests:

1. stable stochastic strongly convex finite-sum quadratic with `r = 0`;
2. stochastic strongly convex quadratic with heavy-tailed finite-sum gradients induced by the data distribution;
3. stochastic strongly convex composite Lasso with a true proximal step;
4. residual-floor verification for the Lyapunov recursion.

Concrete `--test` choices are:

- `1`: stable finite-sum quadratic baseline;
- `2_mild`: mild heavy-tail robustness check;
- `2_strong`: stronger heavy-tail robustness audit;
- `3`: composite Lasso test with a true prox step;
- `3_local`: local parameter sweep around the best Test 3 region;
- `4_floor_vs_theory`: residual-floor test for the Lyapunov recursion;
- `all`: convenience preset for the default combined sweep.

Important note:

- `--test all` currently expands to `1`, `3`, and `2_mild`;
- it does **not** include `2_strong`;
- it does **not** include `4_floor_vs_theory`;
- `2_strong` and `4_floor_vs_theory` are intended to be launched explicitly as separate controlled checks.

The implementation uses:

- exact closed-form solutions for the ridge least-squares model;
- a deterministic high-accuracy FISTA reference solver for the Lasso model;
- stochastic mini-batch gradients for the smooth finite-sum part;
- robust maps `identity`, `coord_clip`, `norm_clip`, and `tanh`;
- vector-space versions of SGD, clipped SGD, Prox-SGD, clipped Prox-SGD, Prox-NAG-GS, and robust Prox-NAG-GS.

### Test 1: stable strongly convex quadratic

Test 1 solves the smooth ridge least-squares problem

```text
f(x) = (1/(2n)) ||Ax-b||^2 + (mu_reg/2)||x||^2.
```

There is no nonsmooth term:

```text
r(x) = 0.
```

Therefore the proximal operator is the identity.

The exact solution is available:

```text
x_star = ((1/n) A^T A + mu_reg I)^(-1) (1/n) A^T b.
```

The only stochasticity comes from mini-batch sampling. No external Gaussian gradient noise is added.

Purpose of the test:

- check that the method is stable in the clean strongly convex stochastic setting;
- verify objective-gap decay;
- verify distance to the exact solution;
- check the Lyapunov quantities;
- observe that smaller batches produce a larger stochastic residual floor.

The main quantities are:

```text
objective_gap = f(v_k) - f(x_star)
distance_v    = ||v_k - x_star||
distance_x    = ||x_k - x_star||
G_k           = f(v_k) - f(x_star)
V_k           = ||v_k - x_star||^2
X_k           = ||x_k - x_star||^2
lyapunov      = G_k + (b-s)V_k + cX_k
```

### Test 2: heavy-tailed finite-sum quadratic

Test 2 solves the same smooth ridge least-squares problem as Test 1:

```text
f(x) = (1/(2n)) ||Ax-b||^2 + (mu_reg/2)||x||^2,
r(x) = 0.
```

The mini-batch gradient has the same expression:

```text
g_B(x) = (1/|B_k|) A_B^T(A_B x - b_B) + mu_reg x.
```

The difference is the way the data matrix `A` is generated. The goal is to create rare samples with unusually large gradient contributions, while keeping the objective strongly convex.

Two heavy-tail mechanisms are available.

#### Student-t features

With

```bash
--heavy-tail-kind student_t
--heavy-tail-df 5.0
```

the entries or rows of `A` are sampled from a Student-t distribution. A Student-t distribution has heavier tails than a Gaussian distribution. This means most rows are moderate, but some rows can be much larger.

The degrees of freedom control how heavy the tails are:

- large degrees of freedom: closer to Gaussian;
- small degrees of freedom: heavier tails;
- `df = 5` gives a mild heavy-tailed setting.

#### Leverage mixture

With

```bash
--heavy-tail-kind leverage_mixture
```

the data are generated as a mixture:

```text
most rows: standard Gaussian rows
rare rows: Gaussian rows multiplied by a large scale
```

For example, the strong test uses

```text
95% standard rows
5% rows multiplied by 30
```

These rare rows are called high-leverage rows. If one of them enters a mini-batch, the mini-batch gradient can be much larger than usual.

No external Gaussian gradient noise is added. The stochasticity still comes only from mini-batch sampling.

Purpose of the test:

- check whether robust maps reduce rare large stochastic-gradient events;
- compare the raw mini-batch error and the transformed oracle error;
- quantify the clipping bias versus robustness trade-off;
- verify that heavier tails make the robust map more active.

The key diagnostics are:

```text
raw_error_norm          = ||g_B(x_next) - grad f(x_next)||
transformed_error_norm  = ||Phi_lambda(g_B(x_next)) - grad f(x_next)||
batch_grad_norm         = ||g_B(x_next)||
transformed_grad_norm   = ||Phi_lambda(g_B(x_next))||
clipping_ratio          = ||Phi_lambda(g_B)-g_B|| / (||g_B|| + 1e-12)
deterministic_bias_norm = ||Phi_lambda(grad f(x_next)) - grad f(x_next)||
```

### Test 3: strongly convex composite Lasso

Test 3 solves a genuinely proximal problem:

```text
F(x) = f(x) + r(x),
```

where

```text
f(x) = (1/(2n)) ||Ax-b||^2 + (mu_reg/2)||x||^2
```

and

```text
r(x) = lambda_l1 ||x||_1.
```

The mini-batch gradient is applied only to the smooth part `f`:

```text
g_B(x) = (1/|B_k|) A_B^T(A_B x - b_B) + mu_reg x.
```

The L1 term is not included in the stochastic gradient. It is handled by the proximal operator.

For Lasso, the prox is soft-thresholding:

```text
prox_{h lambda_l1 ||.||_1}(u)_j
    = sign(u_j) max(|u_j| - h lambda_l1, 0).
```

The robust Prox-NAG-GS step has the form

```text
u = z_{k+1} - h Phi_lambda(g_B(x_{k+1}))
v_{k+1} = prox_{h r}(u),
h = a / mu_hat.
```

Purpose of the test:

- verify that the algorithm works in a composite nonsmooth setting;
- check the true proximal mechanism, not only the smooth gradient step;
- compare Prox-NAG-GS with Prox-SGD and clipped Prox-SGD;
- measure sparsity and support recovery.

The key quantities are:

```text
objective_gap    = F(v_k) - F_star
distance_v       = ||v_k - x_star||
distance_x       = ||x_k - x_star||
sparsity         = fraction of zero or near-zero entries in v_k
support_recovery = agreement with the true sparse support, when available
```

This test mainly validates the composite proximal behavior. It is not the main test for heavy-tail robustness. In the Lasso setting, generally

```text
grad f(x_star) != 0,
```

because the optimality condition is

```text
0 in grad f(x_star) + partial r(x_star).
```

Therefore Test 3 is less direct than Test 4 for checking the vanishing residual-floor theorem.

### Test 4: residual floor versus theory

Test 4 focuses on the Lyapunov residual-floor prediction.

It uses the same smooth strongly convex ridge problem as Test 2, typically with the stronger high-leverage data:

```text
f(x) = (1/(2n)) ||Ax-b||^2 + (mu_reg/2)||x||^2,
r(x) = 0.
```

The test is focused mainly on robust Prox-NAG-GS with coordinate clipping.

The theory predicts a recursion of the form

```text
E[L_{k+1}] <= q E[L_k] + residual term.
```

If the transformed oracle error stays nonzero, the method should converge to a neighborhood. If the transformed oracle error vanishes, the residual floor should also vanish.

Test 4 has two parts.

#### Part A: fixed batch size

The method is run with fixed batch sizes such as

```text
|B| = 64
|B| = 256
```

Expected behavior:

- the Lyapunov function decreases first;
- it then stabilizes around a nonzero floor;
- the floor is lower for larger batch sizes.

#### Part B: increasing batch size

The batch size is increased along the run, for example

```text
64 -> 128 -> 256 -> n.
```

At the final full-batch stage,

```text
g_B(x) = grad f(x),
```

so the mini-batch stochastic error disappears.

Expected behavior:

- the residual proxy decreases;
- the Lyapunov floor moves downward;
- when the final phase is effectively full-batch, the floor becomes very small.

This test is the most direct numerical check of the statement:

```text
constant transformed-oracle error  -> nonzero residual floor
vanishing transformed-oracle error -> vanishing floor
```

It also motivates an annealed robustification strategy:

```text
early iterations: stronger clipping to suppress rare large gradients
later iterations: relaxed threshold and larger batch size
```

The goal is to be robust early, but asymptotically unbiased late.

### Theory parameters and their meaning

The theory tests use the paper notation.

#### Algorithmic curvature

The algorithmic curvature is `mu_hat`. It controls the effective step size

```text
h = a / mu_hat.
```

For the theory-safe runs, the code uses

```text
mu_hat = mu_hat_factor * L,
```

with typical values

```text
mu_hat_factor in {2, 5, 10}.
```

This ensures

```text
mu_hat >= L.
```

Early unstable runs showed that violating this condition can lead to explosions. The stable theory runs therefore report whether `mu_hat >= L`.

#### Coupling parameter

The parameter `a` is the coupling parameter in Prox-NAG-GS. It appears in the coupled update and also affects the effective step size

```text
h = a / mu_hat.
```

Typical stable values used in the theory sweeps are

```text
a in {0.1, 0.2, 0.4}.
```

Later focused tests often use `a = 0.4` with a safe `mu_hat_factor`.

#### Lyapunov parameters

The parameters `s` and `c` are used to build the Lyapunov function. They are not optimizer hyperparameters. The code logs

```text
G_k = F(v_k) - F_star
V_k = ||v_k - x_star||^2
X_k = ||x_k - x_star||^2
lyapunov = G_k + (b-s)V_k + cX_k.
```

The coefficient `c` is selected in the admissible interval from the theory, usually by taking the midpoint. If the interval is invalid, the Lyapunov value is marked invalid for that parameter choice.

#### Clipping threshold quantile

In the experiments, `q = 0.90`, `0.95`, or `0.99` is a threshold quantile used to choose the clipping level. This is not the same as the theoretical contraction factor often also denoted by `q`.

For coordinate clipping:

- `q = 0.90` gives more aggressive clipping;
- `q = 0.95` is the safest default in the current experiments;
- `q = 0.99` is almost unbiased, but gives a weaker robustness effect.

The current empirical conclusion is:

```text
coord_clip with q = 0.95
```

is the best default compromise in the mild and strong heavy-tail tests.

### Default output structure

Theory outputs are saved under:

```text
robust_proxnaggs_outputs/theory_verification/
```

with subfolders:

```text
csv/
figures/
configs/
```

Main saved files include:

- `csv/all_iterations.csv`
- `csv/summary_by_run.csv`
- `csv/summary_by_method.csv`
- `figures/fig_1_*.pdf`
- `figures/fig_2_*.pdf`
- `figures/fig_3_*.pdf`
- `figures/fig_4_*.pdf`
- `configs/theory_config.json`

### Logged quantities

The theory runner logs the key quantities requested for stochastic verification, including:

- `objective_gap`
- `distance_v`
- `distance_x`
- `raw_error_norm`
- `transformed_error_norm`
- `G_k`, `V_k`, `X_k`
- `lyapunov`
- `batch_grad_norm`
- `transformed_grad_norm`
- `clipping_ratio`
- `coordinate_activation_ratio_batch`
- `coordinate_activation_ratio_full`
- `deterministic_bias_norm`
- `sparsity`
- `support_recovery`
- `mu_f`, `mu_F`, `L`
- whether `mu_hat >= L`
- run `status`
- explosion diagnostics such as `first_bad_iteration`, `max_objective_gap`, `max_distance_v`, `max_batch_grad_norm`, and `max_transformed_grad_norm`

### Example commands

Run the default combined theory sweep:

```bash
source RPNAGGS/bin/activate
rpnaggs-theory --test all
```

This convenience mode is a good first pass, but it does not include the stronger heavy-tail stress test or the residual-floor test. Run `--test 2_strong` and `--test 4_floor_vs_theory` explicitly when needed.

Run the current stable Test 1 sweep:

```bash
source RPNAGGS/bin/activate
rpnaggs-theory \
  --test 1 \
  --output-dir robust_proxnaggs_outputs/theory_verification_test1_stable \
  --iterations 150 \
  --seeds 0 \
  --n 5000 \
  --d-values 50 \
  --mu-reg-values 0.1 \
  --batch-sizes-test1 64 256 \
  --a-values 0.1 0.2 0.4 \
  --mu-hat-factors 2 5 10
```

This generates:

- `csv/all_iterations.csv`
- `csv/summary_by_run.csv`
- `csv/summary_by_method.csv`
- `figures/fig_1_mean_lyapunov_vs_iterations.pdf`
- `figures/fig_1_objective_gap_vs_iterations.pdf`
- `figures/fig_1_final_lyapunov_vs_batch_size.pdf`
- `figures/fig_1_gradient_error_survival_by_batch_size.pdf`

Final statement for Test 1:

- the stabilized theory-safe regime should show no explosions;
- larger batch sizes should reduce the stochastic residual floor;
- the Lyapunov and objective-gap plots should decay coherently on a log scale;
- robust maps can be compared against SGD and clipped SGD under the same finite-sum quadratic problem.

Run the current stable Test 3 sweep:

```bash
source RPNAGGS/bin/activate
rpnaggs-theory \
  --test 3 \
  --output-dir robust_proxnaggs_outputs/theory_verification_test3_stable \
  --iterations 150 \
  --seeds 0 \
  --n 5000 \
  --d-values 50 \
  --mu-reg-values 0.1 \
  --batch-sizes-test3 64 256 \
  --a-values 0.1 0.2 0.4 \
  --mu-hat-factors 2 5 10
```

This generates:

- `csv/all_iterations.csv`
- `csv/summary_by_run.csv`
- `csv/summary_by_method.csv`
- `figures/fig_3_lasso_objective_gap.pdf`
- `figures/fig_3_lasso_distance_to_solution.pdf`
- `figures/fig_3_lasso_support_recovery.pdf`
- `figures/fig_3_lasso_sparsity_vs_iterations.pdf`
- additional tuned-analysis figures such as:
  - `figures/fig_3_best_by_objective_gap.pdf`
  - `figures/fig_3_best_by_distance_v.pdf`
  - `figures/fig_3_heatmap_final_objective_gap.pdf`
  - `figures/fig_3_heatmap_final_distance_v.pdf`
  - `figures/fig_3_heatmap_final_sparsity.pdf`
  - `figures/fig_3_heatmap_final_support_recovery.pdf`

Final statement for Test 3:

- this test validates the composite proximal behavior rather than a heavy-tail robustness gain;
- the key quantities are objective gap, distance to the reference solution, sparsity, and support recovery;
- Prox-NAG-GS should be compared against Prox-SGD and clipped Prox-SGD in a stable Lasso regime;
- the parameter heatmaps help identify the best effective-step region, which is often more informative than a single arbitrary display choice.

Run the current stable mild heavy-tail Test 2 sweep:

```bash
source RPNAGGS/bin/activate
rpnaggs-theory \
  --test 2_mild \
  --output-dir robust_proxnaggs_outputs/theory_verification_test2_mild_stable \
  --iterations 150 \
  --seeds 0 \
  --n 5000 \
  --d-values 50 \
  --mu-reg-values 0.1 \
  --batch-sizes-test2 64 256 \
  --a-values 0.1 0.2 0.4 \
  --mu-hat-factors 2 5 10 \
  --heavy-tail-kind student_t \
  --heavy-tail-df 5.0
```

Run the mild heavy-tail threshold-sensitivity sweep:

```bash
source RPNAGGS/bin/activate
rpnaggs-theory \
  --test 2_mild \
  --output-dir robust_proxnaggs_outputs/theory_verification_test2_mild_threshold_sweep \
  --iterations 150 \
  --seeds 0 \
  --n 5000 \
  --d-values 50 \
  --mu-reg-values 0.1 \
  --batch-sizes-test2 64 256 \
  --a-values 0.2 0.4 \
  --mu-hat-factors 2 5 \
  --heavy-tail-kind student_t \
  --heavy-tail-df 5.0 \
  --threshold-quantiles 0.70 0.80 0.90 0.95 0.99 \
  --test2-robust-maps identity coord_clip tanh
```

This threshold sweep keeps the stable mild reference run unchanged and adds dedicated calibration figures such as:

- `figures/fig_2_mild_threshold_final_objective_gap_vs_quantile.pdf`
- `figures/fig_2_mild_threshold_final_distance_vs_quantile.pdf`
- `figures/fig_2_mild_threshold_mean_transformed_error_vs_quantile.pdf`
- `figures/fig_2_mild_threshold_max_transformed_grad_vs_quantile.pdf`
- `figures/fig_2_mild_threshold_mean_bias_vs_quantile.pdf`
- `figures/fig_2_mild_threshold_mean_clipping_ratio_vs_quantile.pdf`

Current mild-regime conclusion:

- `coord_clip` is the best-calibrated robust map in this regime;
- the most useful threshold range is `q` in `0.90-0.95`;
- `q = 0.95` is the safest default mild-regime choice;
- lower thresholds reduce rare large transformed gradients more aggressively, but introduce noticeably more deterministic bias and objective-gap degradation;
- very high thresholds such as `q = 0.99` preserve optimization quality, but recover only a weak robustness effect.

Run the current stable strong heavy-tail Test 2 sweep:

```bash
source RPNAGGS/bin/activate
rpnaggs-theory \
  --test 2_strong \
  --output-dir robust_proxnaggs_outputs/theory_verification_test2_strong_stable \
  --iterations 150 \
  --seeds 0 \
  --n 5000 \
  --d-values 50 \
  --mu-reg-values 0.1 \
  --batch-sizes-test2 256 64 \
  --a-values 0.2 0.4 \
  --mu-hat-factors 5 10 \
  --heavy-tail-kind leverage_mixture \
  --heavy-tail-leverage-fraction 0.05 \
  --heavy-tail-leverage-scale 30 \
  --noise-std 0.0 \
  --threshold-quantiles 0.90 0.95 \
  --test2-robust-maps identity coord_clip
```

This controlled strong run saves:

- `csv/all_iterations.csv`
- `csv/summary_by_run.csv`
- `csv/summary_by_method.csv`
- `figures/fig_2_strong_objective_gap_heavytail_comparison.pdf`
- `figures/fig_2_strong_distance_to_solution_over_iterations.pdf`
- `figures/fig_2_strong_raw_error_survival.pdf`
- `figures/fig_2_strong_transformed_error_survival.pdf`
- `figures/fig_2_strong_batch_grad_survival.pdf`
- `figures/fig_2_strong_transformed_grad_survival.pdf`
- `figures/fig_2_strong_clipping_ratio_over_iterations.pdf`
- `figures/fig_2_strong_lyapunov_components.pdf`

Current strong-regime conclusion:

- `coord_clip` reduces extreme transformed gradients much more clearly than in `test2_mild`;
- the effect is strongest for the smaller mini-batch regime;
- `q = 0.95` remains the safer threshold choice;
- in the current small safe grid, this stronger clipping benefit does not yet translate into a better final objective gap than `identity`.

Run the residual-floor Test 4:

```bash
source RPNAGGS/bin/activate
rpnaggs-theory \
  --test 4_floor_vs_theory \
  --output-dir robust_proxnaggs_outputs/theory_verification_test4_floor \
  --iterations 2000 \
  --seeds 0 1 2 3 4 \
  --n 5000 \
  --d-values 50 \
  --mu-reg-values 0.1 \
  --batch-sizes-test2 64 256 \
  --mu-hat-factors 5 10 \
  --heavy-tail-kind leverage_mixture \
  --heavy-tail-leverage-fraction 0.05 \
  --heavy-tail-leverage-scale 30 \
  --noise-std 0.0 \
  --test2-robust-maps identity coord_clip
```

This focused Test 4 run saves:

- `csv/all_iterations.csv`
- `csv/summary_by_run.csv`
- `csv/summary_by_method.csv`
- `figures/fig_4_lyapunov_vs_iterations_constant_vs_increasing.pdf`
- `figures/fig_4_objective_gap_vs_iterations_constant_vs_increasing.pdf`
- `figures/fig_4_transformed_error_sq_vs_iterations.pdf`
- `figures/fig_4_tail_empirical_floor_vs_batch_size.pdf`
- `figures/fig_4_empirical_floor_vs_residual_proxy.pdf`
- `figures/fig_4_deterministic_bias_norm_vs_iterations.pdf`
- `figures/fig_4_clipping_ratio_vs_iterations.pdf`

Current Test 4 conclusion:

- with fixed batch size, the Lyapunov and objective gap decrease first and then stabilize around a nonzero residual floor;
- larger constant batch sizes give a smaller empirical floor and a smaller residual proxy;
- increasing the batch size drives the observed floor sharply downward;
- when the last phase becomes effectively full-batch, the transformed stochastic error becomes numerically negligible and the empirical floor becomes much smaller;
- in the current safe `q = 0.95` regime, clipping mainly suppresses extreme transformed gradients, while the dominant floor mechanism is still finite mini-batch stochasticity.

### Interpretation

These experiments are not intended to outperform AdamW on deep learning benchmarks. Their role is to verify the expected theory-level messages:

- smaller mini-batches create a stronger residual stochastic regime;
- increasing the batch size lowers the stochastic floor;
- heavy-tailed finite-sum data produces rare extreme stochastic gradients;
- robust maps reduce the upper tail of stochastic gradient errors;
- the threshold introduces a bias-variance tradeoff;
- the proximal Lasso setting creates sparsity and validates the composite update.

## Output files

By default, all outputs are written to:

```text
robust_proxnaggs_outputs/
```

There are two main output families:

- CIFAR-10 training outputs written directly under `robust_proxnaggs_outputs/`
- theory-verification outputs written under a dedicated subfolder such as `robust_proxnaggs_outputs/theory_verification_test2_strong_stable/`

### Training artifacts

- `history_*.csv`: epoch-level train/test metrics, including optimizer diagnostics such as `step_size`, `prox_name`, `data_loss`, `regularization_penalty`, `total_objective`, `gradient_norm`, `transformed_gradient_norm`, `x-v` distance, and clipping statistics
- `training_curves_*.pdf`: combined loss and accuracy plots
- `loss_curves_*.pdf`: train/test loss versus epoch
- `accuracy_curves_*.pdf`: train/test accuracy versus epoch
- `comparison_curves_*.pdf`: multi-run comparison figure
- `comparison_test_loss_*.pdf`: comparison of test loss
- `comparison_test_accuracy_*.pdf`: comparison of test accuracy
- `tail_summary_*.csv`: aggregate tail summaries
- `hill_raw_norms_*.csv`: Hill estimates for gradient-error norms
- `hill_coord_errors_*.csv`: Hill estimates for coordinate-wise errors
- `hist_*.pdf`, `survival_*.pdf`, `hill_*.pdf`: diagnostic figures
- `used_config_*.json`: exact run configuration

### Theory artifacts

Each theory run creates:

- `csv/all_iterations.csv`: per-iteration records for every run in the sweep
- `csv/summary_by_run.csv`: one final summary row per run
- `csv/summary_by_method.csv`: aggregated summaries grouped by method and batch size
- `figures/*.pdf`: theory figures for the selected test
- `configs/theory_config.json`: exact theory CLI configuration

If you only want the strongest robustness check, run `test2_strong` directly and inspect its dedicated folder.

## Test suite

Run the lightweight unit tests with:

```bash
source RPNAGGS/bin/activate
pytest
```

## Tutorial: add or change the model

The project is structured so that model changes are localized. In most cases, adding a new model requires changes only in the model definition layer and the model registry.

For `robust_proxnaggs`, the public optimizer notation follows the paper:

```text
a
mu_hat
step_size = a / mu_hat
```

The proximal part is now modular as well: the optimizer computes the robust-gradient proposal with `step_size = a / mu_hat`, then applies a prox operator with that same step size. So when adapting models or regularizers, recommended parameter tables should preferably be written in terms of `a` and `mu_hat`, not `eta`.

### Files involved

- `src/rpnaggs/models/small_cnn.py`: current default CIFAR model
- `src/rpnaggs/models/vgg7_mini.py`: optional MNIST VGG-7-Mini model
- `src/rpnaggs/models/registry.py`: model selection logic (`dataset`, `model` pairs)
- `src/rpnaggs/data/registry.py`: dataset loader dispatch
- `src/rpnaggs/data/cifar10.py`: CIFAR-10 transforms and loaders
- `src/rpnaggs/data/mnist.py`: MNIST transforms and loaders
- `src/rpnaggs/config.py`: CLI argument definition and defaults
- `src/rpnaggs/experiments/runner.py`: uses the data registry (no dataset-specific logic)

Existing CIFAR-10 code paths are unchanged. New datasets/models are added by registering
new pairs without modifying the old ones.

### Registered model / dataset pairs

| Dataset | Model name | Module |
|---------|------------|--------|
| `cifar10` | `small_cifar_cnn` | `models/small_cnn.py` |
| `mnist` | `vgg7_mini_mnist` | `models/vgg7_mini.py` |

### Procedure

1. Create a new model file in `src/rpnaggs/models/`, for example:

```text
src/rpnaggs/models/vgg.py
```

or:

```text
src/rpnaggs/models/vit.py
```

2. Implement the PyTorch model class in that file.

3. Register the new `(dataset, model)` pair in `src/rpnaggs/models/registry.py`.

4. If you introduce a new dataset, add a loader in `src/rpnaggs/data/` and register it in
   `src/rpnaggs/data/registry.py`.

5. Run a short smoke test before launching long experiments.

### Smoke test for a new model

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --model your_new_model_name \
  --algorithm adamw \
  --epochs 1 \
  --train-subset-size 512 \
  --num-workers 0 \
  --no-diagnostics
```

This verifies that:

- the model builds correctly;
- the forward pass works;
- the output dimension matches the CIFAR-10 labels;
- the training loop accepts the new model.

### MNIST + VGG-7-Mini smoke test

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --dataset mnist \
  --model vgg7_mini_mnist \
  --algorithm adamw \
  --epochs 1 \
  --train-subset-size 512 \
  --num-workers 0 \
  --no-diagnostics
```

### Run all four methods on MNIST

Use the dedicated MNIST suite wrapper (same four algorithms as CIFAR, MNIST-tuned hyperparameters
hardcoded from Optuna):

```bash
source RPNAGGS/bin/activate
python scripts/run_mnist_method_suite.py \
  --epochs 5 \
  --batch-size 128 \
  --label-noise-rate 0.2 \
  --full-train \
  --num-workers 0 \
  --compare-tag mnist_first_comparison
```

Full training set, 10 epochs:

```bash
source RPNAGGS/bin/activate
python scripts/run_mnist_method_suite.py \
  --full-train \
  --epochs 10 \
  --batch-size 128 \
  --label-noise-rate 0.2 \
  --num-workers 0 \
  --compare-tag mnist_full_10ep
```

Or pass `--dataset mnist --model vgg7_mini_mnist` to `scripts/run_method_suite.py` directly
(CIFAR Optuna defaults apply unless you override flags manually).

### Architectural guidance

#### VGG or ResNet

These are the easiest next steps. They usually require:

- a new model class;
- an updated classifier head for 10 classes;
- modest hyperparameter tuning.

In many cases, CIFAR-10 can remain the starting dataset without major pipeline changes.

#### Vision Transformer

This is feasible, but it is a larger modification. A ViT often requires:

- resized inputs;
- larger memory budget;
- lower batch size;
- more hyperparameter tuning;
- preferably a GPU environment.

For this reason, a CNN, VGG, or ResNet is the recommended first extension.

### Practical rule

When introducing a new model family:

1. implement the model;
2. register it;
3. validate it on a small smoke test;
4. only then run full experiments or the suite.

## Tutorial: move the project to GPU

The training code already detects the available device and moves the model and mini-batches accordingly. In practice, migrating to GPU mainly involves environment preparation and parameter tuning rather than code refactoring.

### Step 1. Prepare the environment on the GPU machine

```bash
cd /path/to/Robust_Prox_NAG_GS
python3 -m venv RPNAGGS
source RPNAGGS/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If the default installation does not expose the GPU, install the appropriate GPU-enabled PyTorch build for the target platform and CUDA setup.

### Step 2. Verify GPU visibility

```bash
source RPNAGGS/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If the first value is `True` and the second is greater than `0`, PyTorch can see at least one GPU.

### Step 3. Run a small smoke test

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm adamw \
  --model small_cifar_cnn \
  --robust-map identity \
  --epochs 1 \
  --train-subset-size 512 \
  --no-diagnostics
```

Inspect the terminal summary. If the reported device is `cuda`, the run is using the GPU.

### Step 4. Tune batch size and workers

On GPU, two parameters typically change first:

- `--batch-size`
- `--num-workers`

Example:

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm adamw \
  --model small_cifar_cnn \
  --robust-map identity \
  --epochs 1 \
  --train-subset-size 2048 \
  --batch-size 256 \
  --num-workers 4
```

If out-of-memory errors occur, reduce `--batch-size`.

### Step 5. Run the normal experiments

Single-run example:

```bash
source RPNAGGS/bin/activate
rpnaggs-run \
  --algorithm robust_proxnaggs \
  --model small_cifar_cnn \
  --robust-map tanh \
  --label-noise-rate 0.2 \
  --pnaggs-a 0.4 \
  --pnaggs-mu-hat 1.0 \
  --clip-threshold 0.02 \
  --no-auto-threshold \
  --epochs 5 \
  --full-train \
  --batch-size 256 \
  --num-workers 4
```

Suite example:

```bash
source RPNAGGS/bin/activate
python scripts/run_method_suite.py \
  --model small_cifar_cnn \
  --epochs 5 \
  --batch-size 256 \
  --label-noise-rate 0.2 \
  --full-train \
  --num-workers 4 \
  --compare-tag gpu_suite
```

### What typically changes on GPU

The following usually remain unchanged:

- optimizer implementations;
- diagnostics;
- comparison tooling;
- bundle creation.

The following usually require adjustment:

- batch size;
- worker count;
- runtime expectations;
- occasionally the learning rate when the batch size changes substantially.

### Common GPU issues

#### Device still reports `cpu`

Possible causes:

- a CPU-only PyTorch installation;
- missing or incompatible GPU drivers;
- no GPU access on the current machine.

#### Out-of-memory errors

Typical mitigations:

- reduce `--batch-size`;
- use a smaller model;
- reduce concurrent workload.

#### Training becomes unstable

This is often caused by changed training conditions rather than a GPU-specific defect. Typical causes include:

- a larger model;
- a larger batch size;
- hyperparameters that were tuned for CPU defaults.

## Repository structure

```text
scripts/       # helper scripts such as the sequential benchmark suite
src/rpnaggs/
  algorithms/   # training methods and registries
  data/         # datasets and dataloaders
  diagnostics/  # heavy-tail analysis and plotting
  experiments/  # training loop and evaluation
  models/       # model definitions
  optim/        # robust transforms
  problems/     # synthetic strongly convex problem definitions
  theory/       # vector-space solvers and Lyapunov tools
  utils/        # reproducibility helpers
```
