# Robust Prox-NAG-GS

`Robust Prox-NAG-GS` is a local PyTorch research environment for studying robust stochastic optimization on CIFAR-10. The repository provides:

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

The current default model is `small_cifar_cnn`, a compact CNN for CIFAR-10. At the moment, the public training pipeline supports:

- dataset: `cifar10`
- built-in model: `small_cifar_cnn`

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
git clone https://github.com/YOUR_USERNAME/robust-prox-nag-gs.git
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

1. `rpnaggs-run`: run one CIFAR-10 training experiment.
2. `python scripts/run_method_suite.py`: run several training methods sequentially and optionally generate comparison figures.
3. `rpnaggs-compare`: compare existing `history_*.csv` files and regenerate multi-run comparison PDFs.
4. `rpnaggs-bundle`: collect the minimal CSV bundle for downstream analysis.
5. `rpnaggs-optuna`: tune the training methods with Optuna under a shared budget.
6. `rpnaggs-theory`: run the controlled convex theory-verification experiments.

Use:

```bash
rpnaggs-run --help
rpnaggs-compare --help
rpnaggs-bundle --help
rpnaggs-optuna --help
rpnaggs-theory --help
python scripts/run_method_suite.py --help
```

Use `rpnaggs-run` for CIFAR-10 training, and use `rpnaggs-theory` for the synthetic strongly convex tests. These are separate pipelines with separate outputs.

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
- `--dataset`: dataset name; the current public implementation supports `cifar10`
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

Use `--prox-target` with one of:

- `all`: apply prox to all trainable parameters
- `weights_only`: apply prox only to tensors with dimension at least 2
- `classifier`: apply prox only to the final `nn.Linear` layer
- `conv`: apply prox only to convolutional weights

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

### Recommended robust Prox-NAG-GS run

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

By default, the suite now uses the current tuned recommendation for the `robust_proxnaggs` run:

- `robust_map = tanh`
- `a = 0.4`
- `mu_hat = 1.0`
- `clip_threshold = 0.02`
- `--no-auto-threshold`

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

```bash
rpnaggs-run \
  --algorithm clipped_sgd \
  --model small_cifar_cnn \
  --robust-map norm_clip \
  --label-noise-rate 0.2 \
  --epochs 5 \
  --full-train
```

### Robust Prox-NAG-GS

```bash
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
  --full-train
```

## Run comparison

If several `history_*.csv` files already exist, compare them with:

```bash
source RPNAGGS/bin/activate
rpnaggs-compare \
  robust_proxnaggs_outputs/history_adamw_identity.csv \
  robust_proxnaggs_outputs/history_sgd_identity.csv \
  robust_proxnaggs_outputs/history_clipped_sgd_norm_clip.csv \
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

## Suggested workflow

For a first heavy-tail analysis before evaluating the robust method, the following sequence is recommended:

1. Run `adamw` on noisy CIFAR-10.
2. Inspect the diagnostic PDF figures and tail CSV files.
3. Run `clipped_sgd` and `robust_proxnaggs` under the same conditions.
4. Compare the resulting training histories.
5. Build the compact analysis bundle for external review.

## Theory Verification Tests

The repository also includes a separate theory-oriented experiment stack for controlled strongly convex finite-sum problems. These tests are distinct from the CIFAR-10 pipeline:

- they do not use CNNs;
- they do not use image classification accuracy;
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

### What is implemented

The theory stack currently includes the following test modes:

1. stochastic strongly convex finite-sum quadratic with `r = 0`;
2. stochastic strongly convex quadratic with heavy-tailed finite-sum gradients induced by the data distribution;
3. stochastic strongly convex composite Lasso with a true proximal step.

Concrete `--test` choices are:

- `1`: stable finite-sum quadratic baseline
- `2_mild`: mild heavy-tail robustness check
- `2_strong`: stronger heavy-tail robustness audit
- `3`: composite Lasso test with a true prox step
- `3_local`: local parameter sweep around the best Test 3 region
- `all`: convenience preset for the default combined sweep

Important note:

- `--test all` currently expands to `1`, `3`, and `2_mild`
- it does **not** include `2_strong`
- `2_strong` is intended to be launched explicitly as a separate controlled robustness check

The implementation uses:

- exact closed-form solutions for the ridge least-squares model;
- a deterministic high-accuracy FISTA reference solver for the Lasso model;
- stochastic mini-batch gradients for the smooth finite-sum part;
- robust maps `identity`, `coord_clip`, `norm_clip`, and `tanh`;
- vector-space versions of SGD, clipped SGD, Prox-SGD, clipped Prox-SGD, Prox-NAG-GS, and robust Prox-NAG-GS.

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

This convenience mode is a good first pass, but it does not include the stronger heavy-tail stress test. Run `--test 2_strong` explicitly when you want the final robustness audit.

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

- `src/rpnaggs/models/small_cnn.py`: current default model
- `src/rpnaggs/models/registry.py`: model selection logic
- `src/rpnaggs/config.py`: CLI argument definition and defaults
- `src/rpnaggs/data/cifar10.py`: dataset transforms

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

3. Register the new model name in `src/rpnaggs/models/registry.py`.

4. Run a short smoke test before launching long experiments.

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
