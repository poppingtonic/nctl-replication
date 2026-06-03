# phase5l_adapt_temp — bd 30.12

## Purpose
Sweep `adapt_n` x `posterior_temp` on the canonical 30.9 recipe to test
whether the adapt window or posterior temperature is the residual lever
on T1/T2/T3 (the diagonal-decay signature).

## Fairness fix (Agent 2 review)
All cells evaluate on `test[1000:]` via `--eval-suffix-from 1000`.
Without this, larger `adapt_n` values would evaluate on a different
test subset and the comparison would be biased.  `eval_suffix_from`
is the max of the adapt_n grid.

## Execution (GPU host, synchronous unified_exec session)
`bash run_sweep.sh [SEED]` runs the 12-cell grid for one seed.
Default seed=1.  Cost: ~85 min/seed.

A more efficient Option A (one training run + 12 cheap replays via
save_snapshot/load_snapshot) is intentionally NOT implemented yet:
the simpler retrain-per-cell harness lets us see whether the lever
exists before paying the larger refactor cost.  If GPU time becomes
the binding constraint we'll file a follow-up sub-task and add the
save/load CLI to run_split_mnist.py.

## Acceptance (from bd 30.12)
Per-(adapt_n, posterior_temp) per-task accuracy table written to
COOKBOOK; default adapt_n updated in run_split_mnist.py if a
Pareto-better cell emerges; 5-seed re-confirmation under the new
default still meets >= 90% mean.
