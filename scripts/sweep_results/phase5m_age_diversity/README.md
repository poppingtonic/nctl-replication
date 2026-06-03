# phase5m_age_diversity — bd 30.13

## Purpose
A/B compare the default FIFO eviction against the task-free
`age-diversity` candidate at the canonical 30.9 recipe (pool=80).
The candidate uses only snapshot insertion indices: when the pool is
full, it evicts the slot whose insertion index is closest to another
survivor, preserving broader temporal coverage without consuming
benchmark `task_id`.

## Context
`phase5k_task_floor` showed retention quality is the lever:
`task-floor` reproduced the FIFO baseline on its control arm and lifted
mean accuracy by about +4.52 pp.  Because `task-floor` leaks task
identity, it remains diagnostic only.  This sweep tests whether a
task-free temporal-retention rule can recover the same older-task lift.

## Execution (GPU host, synchronous unified_exec session)
`bash run_ab.sh` runs 5 seeds x 2 policies = 10 runs at roughly the
same cost as `phase5k_task_floor` (~70 min total on the prior host).
The script is idempotent: per-arm JSONs that already exist are skipped.

## Required validation
The `fifo` arm must reproduce 90.94% +/- 0.31 mean accuracy from
`scripts/sweep_results/phase5k_task_floor/summary_ab.json`
(`fifo.avg_accuracy_mean = 90.94496938959395`).  If it does not, debug
the current code path or environment before interpreting the
`age-diversity` delta.

## Acceptance (from bd 30.13)
CPU unit tests cover `age-diversity` invariants; `run_split_mnist.py`
exposes `--pool-evict-policy age-diversity`; this 5-seed A/B records
FIFO vs `age-diversity` accuracy, forgetting, final per-task accuracy,
and pool histograms.  Document whether `age-diversity` closes the
retention gap without task IDs.
