#!/usr/bin/env bash
# bd 30.12: adapt_n x posterior_temp sweep with common-suffix fairness.
#
# Sweeps the 4x3 grid {adapt_n=10,50,200,1000} x {posterior_temp=0.5,1.0,2.0}
# on the canonical 30.9 recipe.  All cells evaluate on test[1000:] so the
# comparison is unbiased by which adapt window the cell consumed.  Single
# seed first; promote a Pareto-better cell to 5-seed under the same
# eval_suffix_from=1000 setting if one emerges.
#
# Cost: 12 cells x ~425 s = ~85 min single seed.  No save/load yet; this is
# the simpler implementation.  If GPU time becomes the binding constraint
# we'll fall back to Option A (save trained state + cheap replay).
set -euo pipefail
cd "$(dirname "$0")/../../../.."
RESULT_DIR="rust-fmn/scripts/sweep_results/phase5l_adapt_temp"
mkdir -p "$RESULT_DIR"
SEED=${1:-1}
EVAL_OFFSET=1000  # max of the adapt_n grid; all cells eval on test[1000:].
for adapt in 10 50 200 1000; do
  for temp in 0.5 1.0 2.0; do
    tag="adapt${adapt}_temp${temp}"
    OUT="$RESULT_DIR/seed${SEED}_${tag}.json"
    LOG="${OUT%.json}.log"
    if [[ -s "$OUT" ]]; then
      echo "[$(date -Is)] seed=${SEED} ${tag} skip (exists)" \
        | tee -a "$RESULT_DIR/progress.log"
      continue
    fi
    echo "[$(date -Is)] seed=${SEED} ${tag} start" \
      | tee -a "$RESULT_DIR/progress.log"
    python rust-fmn/scripts/run_split_mnist.py mnist \
      --nodes 50-25-1 --lr 0.001 \
      --min-segment 512 --pool 80 --pool-reservoir 10 \
      --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
      --active-state per-level --prediction-mode ptw_dp \
      --chunk-size 1024 \
      --posterior-temp "$temp" \
      --adapt-n "$adapt" --eval-suffix-from "$EVAL_OFFSET" \
      --seed "$SEED" --json-out "$OUT" > "$LOG" 2>&1
    grep -E "^(Average Accuracy|Average Forgetting|Total time)" "$LOG" \
      | tee -a "$RESULT_DIR/progress.log"
    echo "[$(date -Is)] seed=${SEED} ${tag} done" \
      | tee -a "$RESULT_DIR/progress.log"
  done
done
python3 rust-fmn/scripts/sweep_results/phase5l_adapt_temp/summarise.py "$SEED"
