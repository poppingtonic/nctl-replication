#!/usr/bin/env bash
# 30.13 A/B sweep: fifo (regression parity) vs age-bucket-floor retention.
#
# Holds the 30.9 canonical recipe fixed at pool=80 so the only difference
# between arms is the pool eviction policy.  The fifo arm must reproduce
# phase5k_task_floor's 90.94% +/- 0.31 control mean before the
# age-bucket-floor delta is interpreted.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
RESULT_DIR="scripts/sweep_results/phase5n_age_bucket_floor"
mkdir -p "$RESULT_DIR"
for policy in fifo age-bucket-floor; do
  for seed in 1 2 3 4 5; do
    OUT="$RESULT_DIR/${policy}_seed${seed}.json"
    LOG="${OUT%.json}.log"
    if [[ -s "$OUT" ]]; then
      echo "[$(date -Is)] policy=${policy} seed=${seed} skip (exists)" \
        | tee -a "$RESULT_DIR/progress.log"
      continue
    fi
    echo "[$(date -Is)] policy=${policy} seed=${seed} start" \
      | tee -a "$RESULT_DIR/progress.log"
    python3 scripts/run_split_mnist.py mnist \
      --nodes 50-25-1 --lr 0.001 \
      --min-segment 512 --pool 80 --pool-reservoir 10 \
      --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
      --pool-evict-policy "$policy" \
      --active-state per-level --prediction-mode ptw_dp \
      --chunk-size 1024 --posterior-temp 1.0 \
      --seed "$seed" --json-out "$OUT" > "$LOG" 2>&1
    grep -E "^(Average Accuracy|Average Forgetting|Total time)" "$LOG" \
      | tee -a "$RESULT_DIR/progress.log" || true
    echo "[$(date -Is)] policy=${policy} seed=${seed} done" \
      | tee -a "$RESULT_DIR/progress.log"
  done
done
python3 scripts/sweep_results/phase5n_age_bucket_floor/summarise.py
