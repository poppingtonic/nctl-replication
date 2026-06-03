# phase5o_age_diversity_oldest_floor — bd 30.13 follow-up

## Purpose
A/B compare the default FIFO eviction against the task-free
`age-diversity-oldest-floor` candidate at the canonical 30.9 recipe
(pool=80).  The candidate protects the `--pool-oldest-floor` oldest
snapshots (slots `[0:floor]`, the earliest insertion indices), then runs
plain `age-diversity` redundancy eviction among the remaining slots. It
uses only snapshot insertion indices and never consumes benchmark
`task_id`.

## Context
- `phase5k_task_floor`: `task-floor` lifted mean accuracy ~+4.52 pp by
  protecting the sole survivor of each task, but it leaks task identity
  and is diagnostic only.
- `phase5m_age_diversity`: the first task-free rule reached 94.23% (+3.28
  pp over FIFO) but the earliest task (T1) still starved — its oldest
  snapshots were flushed because they looked temporally redundant.
- This sweep tests whether reserving a small oldest floor closes the
  residual T1 gap and crosses the paper's 95.07% without task IDs.

## Execution (GPU host, synchronous unified_exec session)
`bash run_ab.sh` runs 5 seeds x 2 policies = 10 runs at roughly the same
cost as `phase5k_task_floor` (~70 min total on the prior host). The floor
defaults to 2; override with `POOL_OLDEST_FLOOR=<k> bash run_ab.sh`. The
script is idempotent: per-arm JSONs that already exist are skipped.

## Required validation
The `fifo` arm must reproduce 90.94% +/- 0.31 mean accuracy from
`scripts/sweep_results/phase5k_task_floor/summary_ab.json`
(`fifo.avg_accuracy_mean = 90.94496938959395`). If it does not, debug the
current code path or environment before interpreting the candidate delta.

## Acceptance (from bd 30.13)
CPU unit tests cover `age-diversity-oldest-floor` invariants (oldest-floor
protection, FIFO fallback when the floor covers the pool, task-id
independence, state-dict roundtrip); `run_split_mnist.py` exposes
`--pool-evict-policy age-diversity-oldest-floor` and `--pool-oldest-floor`.
This 5-seed A/B records FIFO vs candidate accuracy, forgetting, final
per-task accuracy (especially T1), and pool histograms. Document whether
the oldest floor recovers the T1 deficit and crosses the paper 95.07%
without task IDs, and compare against `phase5m_age_diversity` (94.23%).
