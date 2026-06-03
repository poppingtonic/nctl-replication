# phase5n_age_bucket_floor — bd 30.13

## Purpose
A/B compare the default FIFO eviction against the task-free
`age-bucket-floor` candidate at the canonical 30.9 recipe (pool=80).
The candidate uses only snapshot insertion indices: when the pool is
full, it groups retained snapshots by log-age bucket and protects at
least one survivor per occupied bucket before falling back to FIFO.

## Context
`phase5k_task_floor` showed retention quality is the lever:
`task-floor` reproduced the FIFO baseline on its control arm and lifted
mean accuracy by about +4.52 pp.  Because `task-floor` leaks task
identity, it remains diagnostic only.  `age-bucket-floor` is the closest
task-free analogue: protect temporal strata rather than benchmark task
IDs, matching the FMN/PTW binary time hierarchy targeted by the NCTL
paper's bounded `UPDATEMODELPOOL` replacement step.

## Execution (GPU host, synchronous unified_exec session)
`bash run_ab.sh` runs 5 seeds x 2 policies = 10 runs at roughly the
same cost as `phase5k_task_floor` (~70 min total on the prior host).
The script is idempotent: per-arm JSONs that already exist are skipped.

## Required validation
The `fifo` arm must reproduce 90.94% +/- 0.31 mean accuracy from
`scripts/sweep_results/phase5k_task_floor/summary_ab.json`
(`fifo.avg_accuracy_mean = 90.94496938959395`).  If it does not, debug
the current code path or environment before interpreting the
`age-bucket-floor` delta.

## Acceptance (from bd 30.13)
CPU unit tests cover `age-bucket-floor` invariants; `run_split_mnist.py`
exposes `--pool-evict-policy age-bucket-floor`; this 5-seed A/B records
FIFO vs `age-bucket-floor` accuracy, forgetting, final per-task accuracy,
and pool histograms.  Document whether `age-bucket-floor` closes the
retention gap without task IDs.
