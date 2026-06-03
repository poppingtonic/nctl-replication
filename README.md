# Neural Combinatorial Transfer Learning (NCTL) Replication

This repository is a release-oriented artifact for the GPU Split-MNIST /
Split-Fashion-MNIST Neural Combinatorial Transfer Learning (NCTL) replication
effort.  It contains the NCTL implementation, CUDA kernels, benchmark runners,
evaluation/sweep harnesses, focused tests, and documentation needed to rerun the
strict task-free replication experiments.

The target paper result for Split-MNIST is:

```text
95.07% average accuracy
```

The strongest strict task-free five-seed result captured by this artifact is:

```text
95.248% average accuracy
4.242% average forgetting
age-diversity-oldest-floor, pool_oldest_floor=6
0.178 percentage points above the paper target
```

The earlier floor-4 sweep reached `95.043%` average accuracy, 0.027 percentage
points below the paper target.

Diagnostic task-ID-assisted retention reaches about `95.46%`, identifying model
pool retention as the key lever. Because that diagnostic consumes benchmark
task IDs, it is not a paper-faithful result.

## Repository Layout

```text
scripts/
  run_split_mnist.py                  # direct benchmark entry point
  download_mnist.py                   # dataset bootstrap helper
  optuna_mlflow_sweep.py              # Optuna/MLflow search wrapper
  inspect_pool_provenance.py          # provenance inspection helper
  nctl_bench/                         # NCTL implementation package
  tests/                              # CPU/CUDA/summary tests
  sweep_results/                      # reproducible sweep harnesses and optional artifacts
docs/
  nctl-architecture.md                # Mermaid architecture diagrams
  nctl-parameter-reference.md         # runner/search parameter reference
data/
  README.md                           # expected dataset files
REPRODUCE.md                          # exact reproduction commands
RELEASE_MANIFEST.md                   # release contents and artifact boundaries
```

The implementation front page is `scripts/nctl_bench/README.md`.

## Quick Start

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch pytest ruff optuna mlflow

python3 scripts/download_mnist.py
bash scripts/nctl_bench/cuda/build_kernels.sh

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

For a full reproduction workflow, see `REPRODUCE.md`.

## What Is Paper-Faithful Here?

Paper-faithful or intended candidate settings:

- `active_state=per-level`
- `prediction_mode=ptw_dp`
- `pool_update_policy=paper`
- `posterior_temp=1.0`
- no task-ID-based training-time eviction

Diagnostic settings:

- `pool_evict_policy=task-floor` uses task IDs and is not paper-faithful.
- `close_task_boundary=1` uses task-boundary information and is only valid when
  the target environment exposes boundaries.
- `prediction_mode=selected_level` and `active_state=flat` are legacy
  approximations useful for regression checks.

## Testing

CPU-focused checks:

```bash
ruff check scripts
python3 -m pytest scripts/tests/test_pool_evict_policy.py -q
python3 -m pytest scripts/tests/test_phase5o_age_diversity_oldest_floor_summary.py -q
python3 -m pytest scripts/tests/test_phase5o_age_diversity_oldest_floor_nofifo_summary.py -q
```

Full test suite:

```bash
python3 -m pytest scripts/tests -q
```

CUDA tests are marked and should be run on a host with the precompiled kernels
available.  CPU-only hosts skip CUDA-required tests where the test helper marks
them appropriately.

## Release Status

This is best treated as a reproducible research artifact, not a general-purpose
Python package.  The CUDA build path is documented and tested on the development
environment, but users should expect to adapt `scripts/nctl_bench/cuda/BUILD.md`
for other CUDA/PyTorch/nvcc combinations.

Generated logs, MLflow directories, and exploratory sweep dumps are not part of
the clean source surface.  We keep source/harnesses in git and publish large result
bundles as attached release artifacts when needed.

## Based On
@article{Wang2020ACP,
  title={A Combinatorial Perspective on Transfer Learning},
  author={Jianan Wang and Eren Sezener and David Budden and Marcus Hutter and Joel Veness},
  journal={ArXiv},
  year={2020},
  volume={abs/2010.12268},
  url={https://api.semanticscholar.org/CorpusID:225062550}
}