# Reproducing the NCTL Split-MNIST Runs

All commands are written for the repository root:

```bash
cd /path/to/neural_combinatorial_transfer_learning
```

## 1. Environment

Use Python 3.10+ and a CUDA-capable PyTorch build.  The development environment
used PyTorch `2.9.0+cu128` with locally precompiled CUDA extensions.  See
`scripts/nctl_bench/cuda/BUILD.md` for CUDA compiler details and workarounds.

Minimal Python dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch pytest ruff optuna mlflow
```

If you only want to run the direct benchmark, `optuna` and `mlflow` are optional.

## 2. Data

Download/preprocess MNIST and Fashion-MNIST into the expected binary format:

```bash
python3 scripts/download_mnist.py
```

Expected files are documented in `data/README.md`.

## 3. CUDA Kernels

Build the kernels before GPU benchmark runs:

```bash
bash scripts/nctl_bench/cuda/build_kernels.sh
```

The Python implementation loads cached shared objects for:

- `ggm_cuda` (standalone/reference path)
- `fmn_mixture_cuda` (single-level fallback / selected-level path)
- `fmn_mixture_multilevel_cuda` (current PTW-DP hot path)

The active strict NCTL path uses the fused FMN mixture kernels, which inline the
Gated Geometric Mixer (GGM) context, prediction, and update math.

## 4. Smoke Test

Run a single strict task-free seed:

```bash
python3 scripts/run_split_mnist.py mnist \
  --nodes 50-25-1 --lr 0.001 \
  --min-segment 512 --pool 80 --pool-reservoir 10 \
  --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
  --pool-evict-policy age-diversity-oldest-floor --pool-oldest-floor 6 \
  --active-state per-level --prediction-mode ptw_dp \
  --chunk-size 1024 --posterior-temp 1.0 \
  --adapt-n 50 \
  --seed 1 --json-out scripts/sweep_results/smoke_seed1.json
```

The run should print a startup line showing that the multilevel CUDA extension
loaded and has `forward_update_with_resets`.

The recorded floor-6 seed-1 run in `run-03-06-2026-17:47:27.json` produced
`avg_accuracy = 95.19998391137133` and
`avg_forgetting = 4.316164392373658`.

## 5. Five-Seed Strict Candidate

The floor-6 candidate-only rerun uses the recorded seed-1 recipe and runs seeds
1-5 without rerunning FIFO:

```bash
bash scripts/sweep_results/phase5o_age_diversity_oldest_floor/run_floor6_nofifo.sh
```

The summary is written to:

```text
scripts/sweep_results/phase5o_age_diversity_oldest_floor/summary_floor6_nofifo.json
```

Expected headline:

```text
floor6_candidate.avg_accuracy_mean ~= 95.2479
floor6_candidate.avg_forgetting_mean ~= 4.2419
floor6_delta_to_paper_target ~= +0.1779
```

The older floor-4 candidate-only rerun remains available as:

```bash
bash scripts/sweep_results/phase5o_age_diversity_oldest_floor/run_ab_nofifo.sh
```

## 6. Full FIFO vs Candidate A/B

To rerun both the FIFO control and the floor-2
`age-diversity-oldest-floor` candidate:

```bash
bash scripts/sweep_results/phase5o_age_diversity_oldest_floor/run_ab.sh
```

The summary is written to:

```text
scripts/sweep_results/phase5o_age_diversity_oldest_floor/summary_ab.json
```

## 7. Diagnostic Retention Sweeps

These sweeps document the path to the current retention policy:

```bash
bash scripts/sweep_results/phase5m_age_diversity/run_ab.sh
bash scripts/sweep_results/phase5n_age_bucket_floor/run_ab.sh
```

`task-floor` diagnostics are useful for understanding the remaining gap but are
not paper-faithful because they use task IDs.

## 8. Tests

Fast CPU checks:

```bash
ruff check scripts
python3 -m pytest scripts/tests/test_pool_evict_policy.py -q
python3 -m pytest scripts/tests/test_phase5m_age_diversity_summary.py -q
python3 -m pytest scripts/tests/test_phase5n_age_bucket_floor_summary.py -q
python3 -m pytest scripts/tests/test_phase5o_age_diversity_oldest_floor_summary.py -q
python3 -m pytest scripts/tests/test_phase5o_age_diversity_oldest_floor_nofifo_summary.py -q
```

Full suite:

```bash
python3 -m pytest scripts/tests -q
```

## 9. Interpreting Results

Always inspect:

- `avg_accuracy`
- `avg_forgetting`
- final per-task accuracy from `acc_matrix`
- pool provenance histograms
- per-seed values, not just means

The Split-MNIST failure mode that drove this replication effort was early-task
starvation in the bounded model pool.  A candidate can look healthy on mean
accuracy while still under-retaining task 1.
