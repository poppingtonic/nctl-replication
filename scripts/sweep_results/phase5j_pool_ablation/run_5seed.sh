#!/usr/bin/env bash
# 30.10 5-seed gate run at the winning pool.
#
# Usage: bash run_5seed.sh POOL
#
# Promotes the winning single-seed pool to a 5-seed sweep and emits
# summary_pool${POOL}_5seed.json.  Must run synchronously under a
# unified_exec session; nohup & is killed by the sandbox.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
POOL=${1:?usage: bash run_5seed.sh POOL}
RESULT_DIR="rust-fmn/scripts/sweep_results/phase5j_pool_ablation"
mkdir -p "$RESULT_DIR"
for seed in 1 2 3 4 5; do
  OUT="$RESULT_DIR/seed${seed}_pool${POOL}.json"
  LOG="${OUT%.json}.log"
  if [[ -s "$OUT" ]]; then
    echo "[$(date -Is)] pool=${POOL} seed=${seed} skip (exists)" \
      | tee -a "$RESULT_DIR/progress.log"
    continue
  fi
  echo "[$(date -Is)] pool=${POOL} seed=${seed} start" \
    | tee -a "$RESULT_DIR/progress.log"
  python rust-fmn/scripts/run_split_mnist.py mnist \
    --nodes 50-25-1 --lr 0.001 \
    --min-segment 512 --pool "$POOL" --pool-reservoir 10 \
    --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
    --active-state per-level --prediction-mode ptw_dp \
    --chunk-size 1024 --posterior-temp 1.0 \
    --seed "$seed" --json-out "$OUT" > "$LOG" 2>&1
  grep -E "^(Average Accuracy|Average Forgetting|Total time)" "$LOG" \
    | tee -a "$RESULT_DIR/progress.log"
  echo "[$(date -Is)] pool=${POOL} seed=${seed} done" \
    | tee -a "$RESULT_DIR/progress.log"
done
python3 rust-fmn/scripts/sweep_results/phase5j_pool_ablation/summarise.py "$POOL"
