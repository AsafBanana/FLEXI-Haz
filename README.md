# FLEXI-Haz Simulation Code

This repository contains simulation code for the FLEXI-Haz survival model. The model estimates a partially linear continuous-time hazard of the form

```text
h(t | X, Z) = exp(g(t, X) + theta^T Z)
```

where `g(t, X)` is learned by a neural network and `theta` is the interpretable low-dimensional linear component.

The repository currently has two main entry-point files:

| File | Purpose | Main entry point |
|---|---|---|
| `simlib.py` | Continuous-covariate simulation focused mainly on estimating `theta`. | `run_sample(n, nn_config)` |
| `simlib_CI.py` | Discrete-X cross-fitted one-step confidence-interval simulation for cumulative hazard / survival curves. | `run_sample(...)`, `run_local_batch(...)` |

## Installation

The code is written in Python and uses TensorFlow/Keras for the neural network estimator.

A minimal environment should include:

```bash
pip install numpy pandas scipy scikit-learn tensorflow lifelines matplotlib
```

For saving outputs as parquet in the CI code, also install one parquet backend:

```bash
pip install pyarrow
```

If `pyarrow` is not available, the code falls back to compressed CSV for saved data frames.

## Neural-network configuration

Most runs are controlled by a dictionary such as:

```python
NN_CONFIG = {
    "hidden_layers_nodes": 64,
    "n_hidden_layers": 4,
    "learning_rate": 0.001,
    "activation": "relu",
    "optimizer": "adam",
    "batch_size": 10000,
    "patience": 50,
    "dropout": 0.0,
    "lmbd_L1": 0.0,
    "lmbd_cali": 0.0,
    "lmbd_cor": 0.0,
    "epochs": 2000,
    "jit_compile": False,
}
```

Notes:

- `hidden_layers_nodes`: number of neurons in each hidden layer.
- `n_hidden_layers`: number of hidden layers in the network for `g(t, X)`.
- `learning_rate`: optimizer learning rate.
- `batch_size`: training batch size. Large batches are useful on GPU.
- `patience`: early-stopping patience.
- `lmbd_L1`: L1 regularization strength.
- `lmbd_cali`, `lmbd_cor`: optional calibration/correlation penalty weights used by the custom loss.
- `jit_compile`: enables TensorFlow XLA compilation when supported.

## 1. Continuous simulation for theta estimation: `simlib.py`

Use this file when the goal is to run one simulated dataset and evaluate estimation of the parametric component `theta`.

### Entry point

```python
from simlib import run_sample

NN_CONFIG = {
    "hidden_layers_nodes": 64,
    "n_hidden_layers": 4,
    "learning_rate": 0.001,
    "activation": "relu",
    "optimizer": "adam",
    "batch_size": 10000,
    "patience": 50,
    "dropout": 0.0,
    "lmbd_L1": 0.0,
    "lmbd_cali": 0.0,
    "lmbd_cor": 0.0,
    "epochs": 2000,
    "jit_compile": False,
}

result = run_sample(n=6000, nn_config=NN_CONFIG)
```

### Returned object

`run_sample` returns a tuple:

```python
val_score, beta_hat, cov_theta, in_ci_90, in_ci_95, r2, beta_oracle, data = result
```

where:

- `val_score`: validation loss / likelihood score under the reference setting.
- `beta_hat`: estimated linear coefficient vector for `Z`.
- `cov_theta`: estimated covariance matrix for `beta_hat`.
- `in_ci_90`: indicator vector for whether the true theta lies in the 90% confidence interval.
- `in_ci_95`: indicator vector for whether the true theta lies in the 95% confidence interval.
- `r2`: event-time validation R² for the learned nuisance component.
- `beta_oracle`: beta obtained when optimizing with the true nuisance component.
- `data`: raw simulated subject-level data.

### Typical Monte Carlo loop

```python
import numpy as np
import pandas as pd
from simlib import run_sample

rows = []
for seed in range(20):
    np.random.seed(seed)
    val_score, beta_hat, cov_theta, in_ci_90, in_ci_95, r2, beta_oracle, data = run_sample(
        n=6000,
        nn_config=NN_CONFIG,
    )
    rows.append({
        "seed": seed,
        "beta0": beta_hat[0],
        "beta1": beta_hat[1],
        "cover90_beta0": in_ci_90[0],
        "cover90_beta1": in_ci_90[1],
        "cover95_beta0": in_ci_95[0],
        "cover95_beta1": in_ci_95[1],
        "r2": r2,
    })

summary = pd.DataFrame(rows)
print(summary.mean(numeric_only=True))
```

## 2. Discrete-X confidence-interval simulation: `simlib_CI.py`

Use this file when the goal is to evaluate cross-fitted one-step confidence intervals for cumulative hazard and survival curves in the discrete-X setting.

This code performs:

1. simulation of a training dataset,
2. outer cross-fitting over subjects,
3. nuisance fitting for `g(t, X)`,
4. one-step correction for cumulative hazard,
5. evaluation of survival curves for newly sampled patients,
6. optional saving of run outputs.

### Single run entry point

```python
from simlib_CI import run_sample, DEFAULT_NN_CONFIG, DEFAULT_RESIDUAL_CONFIG

result = run_sample(
    n=6000,
    nn_config=DEFAULT_NN_CONFIG,
    residual_config=DEFAULT_RESIDUAL_CONFIG,
    seed=0,
    n_new_samples=200,
    crossfit_k=5,
    time_scale=30.0,
    output_dir="saved_runs/discreteX_seed_00000",
)
```

### Important arguments

- `n`: number of training subjects in the Monte Carlo dataset.
- `nn_config`: neural-network configuration for the main FLEXI-Haz model.
- `residual_config`: smaller neural-network configuration for residual nuisance fits.
- `seed`: random seed for reproducibility.
- `n_new_samples`: number of newly sampled patients used for curve evaluation.
- `crossfit_k`: number of outer cross-fitting folds.
- `time_scale`: time normalization constant used when feeding time into the neural network.
- `output_dir`: if provided, saves result files to disk.
- `save_full_run_bundle`: if `True`, saves the trained fold models and full cross-fit object. This can use substantial disk space.
- `use_cpu_only`: set to `True` to force CPU execution.
- `enable_xla`: set to `True` to try TensorFlow XLA compilation.
- `enable_mixed_precision`: set to `True` to try mixed precision on GPU.

### Returned object

`run_sample` returns a dictionary with:

```python
{
    "seed": ...,
    "elapsed_sec": ...,
    "n_subjects": ...,
    "n_new_samples": ...,
    "crossfit_k": ...,
    "time_scale": ...,
    "nn_config": ...,
    "residual_config": ...,
    "run_summary": ...,
    "curves_df": ...,
    "patient_summary_df": ...,
}
```

The most important outputs are:

- `curves_df`: long-format table containing true and estimated cumulative hazard / survival curves, confidence intervals, standard errors, and pointwise coverage indicators.
- `patient_summary_df`: one row per new patient, with average curve coverage and final-time summaries.
- `run_summary`: compact metadata and mean patient-level coverage summary.

When `output_dir` is provided, the code saves:

```text
output_dir/
├── config.json
├── run_summary.json
├── survival_curves.parquet      # or survival_curves.csv.gz
└── patient_summary.parquet      # or patient_summary.csv.gz
```

If `save_full_run_bundle=True`, it also saves a model bundle under:

```text
output_dir/bundle/
```

### Local batch entry point

For several Monte Carlo seeds, use `run_local_batch`:

```python
from simlib_CI import run_local_batch, DEFAULT_NN_CONFIG, DEFAULT_RESIDUAL_CONFIG

summary_df = run_local_batch(
    seeds=range(10),
    output_root="saved_runs/discreteX_batch",
    n_subjects=6000,
    n_new_samples=200,
    nn_config=DEFAULT_NN_CONFIG,
    residual_config=DEFAULT_RESIDUAL_CONFIG,
    crossfit_k=5,
    time_scale=30.0,
    use_cpu_only=False,
    enable_xla=False,
    enable_mixed_precision=False,
    save_full_run_bundle=False,
)

print(summary_df)
```

This creates one directory per seed:

```text
saved_runs/discreteX_batch/
├── seed_00000/
├── seed_00001/
├── seed_00002/
└── run_summary_all.parquet      # or run_summary_all.csv.gz
```

## Samplers

Both files define simulation classes for nonlinear survival data. The default setting uses a nonlinear non-proportional-hazards simulator:

```python
SimStudyNonLinearNonPH
```

In `simlib_CI.py`, the sampler can be selected through `sampler_name`:

```python
run_sample(..., sampler_name="nonlinear_nonph")
```

Available names:

- `"nonlinear_nonph"` or `"default"`: nonlinear non-proportional hazards.
- `"nlph"`: nonlinear proportional-ish hazard variant.
- `"nonlinear_ph"`: nonlinear proportional hazards variant.

## Recommended workflow

For theta-only experiments:

1. Edit `NN_CONFIG`.
2. Run `simlib.run_sample(n, NN_CONFIG)`.
3. Aggregate `beta_hat`, `cov_theta`, and coverage indicators across seeds.

For discrete-X CI experiments:

1. Use `simlib_CI.DEFAULT_NN_CONFIG` and `DEFAULT_RESIDUAL_CONFIG` as starting points.
2. Run a small test first, for example `n=500`, `n_new_samples=10`, `crossfit_k=2`, and fewer epochs.
3. Scale to the intended Monte Carlo size.
4. Use `run_local_batch` for multiple seeds.
5. Aggregate `patient_summary` or `run_summary_all` to report coverage.

## Performance notes

- GPU is recommended for large runs.
- Large `batch_size` values, such as `10000`, are used because the likelihood expansion creates many interval rows per subject.
- `enable_xla=True` may improve speed, but should be checked for numerical stability.
- `save_full_run_bundle=True` is useful for debugging or plotting later, but can require substantial disk space.

## Minimal smoke tests

Theta simulation:

```python
from simlib import run_sample

cfg = NN_CONFIG.copy()
cfg["epochs"] = 5
cfg["patience"] = 2
out = run_sample(100, cfg)
print(out[1])  # beta_hat
```

Discrete-X CI simulation:

```python
from simlib_CI import run_sample, DEFAULT_NN_CONFIG, DEFAULT_RESIDUAL_CONFIG

nn_cfg = DEFAULT_NN_CONFIG.copy()
res_cfg = DEFAULT_RESIDUAL_CONFIG.copy()
nn_cfg["epochs"] = 5
nn_cfg["patience"] = 2
res_cfg["epochs"] = 5
res_cfg["patience"] = 2

out = run_sample(
    n=200,
    nn_config=nn_cfg,
    residual_config=res_cfg,
    seed=0,
    n_new_samples=5,
    crossfit_k=2,
)
print(out["run_summary"])
```

## Citation / paper context

This code supports simulation experiments for the FLEXI-Haz model: a flexible partially linear deep-learning framework for continuous-time survival analysis. The main focus is separating the interpretable linear effect `theta^T Z` from the flexible nuisance component `g(t, X)`, while using the full likelihood rather than a Cox partial likelihood.
