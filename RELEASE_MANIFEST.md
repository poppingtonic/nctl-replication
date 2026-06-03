# Release Manifest

This manifest defines the clean source surface for the Neural Combinatorial
Transfer Learning (NCTL) replication artifact.

## Source Surface

Implementation:

- `scripts/nctl_bench/`
- `scripts/run_split_mnist.py`
- `scripts/download_mnist.py`
- `scripts/optuna_mlflow_sweep.py`
- `scripts/inspect_pool_provenance.py`

Tests:

- `scripts/tests/test_pool_evict_policy.py`
- `scripts/tests/test_evaluate_task_eval_offset.py`
- `scripts/tests/test_profile.py`
- `scripts/tests/test_optuna_mlflow_sweep.py`
- `scripts/tests/test_cuda_fmn_kernel.py`
- `scripts/tests/test_gaussian_head.py`
- `scripts/tests/test_gaussian_head_deepmind_parity.py`
- `scripts/tests/test_phase5m_age_diversity_summary.py`
- `scripts/tests/test_phase5n_age_bucket_floor_summary.py`
- `scripts/tests/test_phase5o_age_diversity_oldest_floor_summary.py`
- `scripts/tests/test_phase5o_age_diversity_oldest_floor_nofifo_summary.py`

Documentation:

- `README.md`
- `REPRODUCE.md`
- `RELEASE_MANIFEST.md`
- `CITATION.cff`
- `docs/nctl-architecture.md`
- `docs/nctl-parameter-reference.md`
- `scripts/nctl_bench/README.md`
- `scripts/nctl_bench/cuda/BUILD.md`
- `data/README.md`

Benchmark harnesses:

- `scripts/sweep_results/phase5m_age_diversity/README.md`
- `scripts/sweep_results/phase5m_age_diversity/run_ab.sh`
- `scripts/sweep_results/phase5m_age_diversity/summarise.py`
- `scripts/sweep_results/phase5n_age_bucket_floor/README.md`
- `scripts/sweep_results/phase5n_age_bucket_floor/run_ab.sh`
- `scripts/sweep_results/phase5n_age_bucket_floor/summarise.py`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/README.md`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/run_ab.sh`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/run_ab_nofifo.sh`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/run_floor6_nofifo.sh`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/summarise.py`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/summarise_nofifo.py`

Optional result artifacts:

- `scripts/sweep_results/phase5m_age_diversity/summary_ab.json`
- `scripts/sweep_results/phase5n_age_bucket_floor/summary_ab.json`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/summary_ab.json`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/summary_nofifo.json`
- `scripts/sweep_results/phase5o_age_diversity_oldest_floor/summary_floor6_nofifo.json`
- per-seed JSON files used to produce those summaries

## Excluded From Source Releases

Do not include these in source-only release archives:

- raw `.log` files
- exploratory sweep dumps outside the selected phase5m/5n/5o harnesses
- `mlruns/`
- Python caches and test caches
- local CUDA build artifacts (`*.so`, object files, build directories)
- downloaded datasets

Publish large generated outputs as separate release assets if needed.

## Release Checks

Before tagging:

```bash
ruff check scripts
python3 -m pytest scripts/tests/test_pool_evict_policy.py -q
python3 -m pytest scripts/tests/test_phase5o_age_diversity_oldest_floor_summary.py -q
python3 -m pytest scripts/tests/test_phase5o_age_diversity_oldest_floor_nofifo_summary.py -q
python3 -m pytest scripts/tests/test_optuna_mlflow_sweep.py -q
```

On a CUDA host:

```bash
bash scripts/nctl_bench/cuda/build_kernels.sh
python3 -m pytest scripts/tests/test_cuda_fmn_kernel.py -q
```

## Suggested Tag

Use semantic research-artifact tags, for example:

```text
nctl-replication-v0.1
```

The release notes should state that the strict task-free floor-6 five-seed run
reaches `95.248%` average accuracy with `4.242%` average forgetting, crossing
the paper target by 0.178 percentage points. The floor-4 five-seed sweep remains
a near replication (`95.043%` vs `95.07%`).
