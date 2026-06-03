# phase5k_task_floor — bd 30.11

## Purpose
A/B compare the default FIFO eviction against the opt-in `task-floor`
diagnostic at the canonical 30.9 recipe (pool=80).  The diagnostic
walks slots oldest-first and protects the sole surviving snapshot of
each task on a node.

## Paper-faithfulness caveat (Agent 2 review)
`task-floor` consumes the benchmark `task_id` directly.  COOKBOOK's
scope-lock forbids task-identity signals in the paper-replication
critical path.  Therefore:
- Run task-floor as a DIAGNOSTIC only.
- Do NOT promote it to the default in run_split_mnist.py even if it
  lifts accuracy.  If retention-quality emerges as the lever, the
  follow-up work is to design a task-FREE retention policy (age
  buckets, segment-age floor, snapshot-reservoir, or posterior
  diversity) and only that becomes the candidate default.

## Execution (GPU host, synchronous unified_exec session)
`bash run_ab.sh` runs 5 seeds × 2 policies = 10 runs at ~425 s each =
~70 min total.  Idempotent: per-arm JSONs that already exist are
skipped.

## Required validation
The `fifo` arm must reproduce 90.94% ± 0.31 mean accuracy
(phase5i_c_5seed/summary_relaunched_20260525.json).  If it doesn't,
the 30.11 refactor introduced drift and must be debugged before
reading the task-floor delta.

## Acceptance (from bd 30.11)
Implementation + CPU unit tests + 5-seed comparison at pool=80 vs
FIFO baseline; A/B shows task-floor lifts T1/T2/T3 final accuracy
without harming T4/T5; commit references the +/- delta in COOKBOOK.
