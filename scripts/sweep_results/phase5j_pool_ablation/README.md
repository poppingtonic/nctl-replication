# phase5j_pool_ablation — bd 30.10

## Purpose
Test whether pool capacity > 80 closes the residual 4.13 pp gap between the
30.9 acceptance (5-seed mean 90.94%) and the paper's 95.07%.  Phase 5I-D
unblocked --pool > 80 by making `evaluate_task` in-place.

## Baseline (parent commit)
- 5-seed pool=80 mean 90.94% ± 0.31 (recipe-locked).
- Per-task after T5: T1=80, T2=83, T3=85, T4=97, T5=97 (diagonal decay).

## Execution order (GPU host, synchronous unified_exec session)
1. `bash run_single_seed.sh 100 1`  (~7-8 min)
2. `bash run_single_seed.sh 120 1`  (~9-10 min)
3. If seed=1 pool=120 >= 92%, run `python3 run_pool160_safety_check.py`;
   if it passes, `bash run_single_seed.sh 160 1`  (~12-13 min).
4. Promote the SMALLEST pool whose seed=1 reaches >= 92% to 5-seed gate:
   `bash run_5seed.sh POOL`  (~36-50 min total).
5. `python3 summarise.py POOL` aggregates the per-seed JSONs.

## Acceptance
5-seed mean >= 93% with all five task IDs surviving the end-of-run
`pool_provenance.task_histogram`.

## Decision tree (Agent 1 review)
- pool=100 < 91.5% AND pool=120 < 92%   -> stop, capacity is sub-linear;
                                            pivot to 30.11 task-floor diag.
- pool=120 >= 92%                       -> run pool=160 safety + single-seed.
- 5-seed at winning pool >= 93%         -> close 30.10, ratchet epic 30.
- 5-seed at winning pool 91-93%         -> partial close + escalate 30.11/30.12.

## Files
- run_single_seed.sh           — one (pool, seed) cell.
- run_5seed.sh                 — 5-seed gate at one pool.
- summarise.py                 — aggregate seed JSONs -> summary_pool${POOL}.json.
- run_pool160_safety_check.py  — GPU-mem pre-flight before pool=160.
