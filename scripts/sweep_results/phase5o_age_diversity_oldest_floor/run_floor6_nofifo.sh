#!/usr/bin/env bash
# Five-seed floor=6 follow-up for the task-free age-diversity-oldest-floor
# candidate. This matches the recorded seed-1 recipe from
# run-03-06-2026-17:47:27.json and skips any per-seed JSON that already exists.
set -euo pipefail

if [[ ! -f scripts/run_split_mnist.py ]]; then
  echo "error: run this script from the repository root" >&2
  exit 2
fi

RESULT_DIR="scripts/sweep_results/phase5o_age_diversity_oldest_floor"
CANDIDATE="age-diversity-oldest-floor"
OLDEST_FLOOR=6

mkdir -p "$RESULT_DIR"

for seed in 1 2 3 4 5; do
  OUT="$RESULT_DIR/${CANDIDATE}_floor${OLDEST_FLOOR}_seed${seed}.json"
  LOG="${OUT%.json}.log"
  if [[ -s "$OUT" ]]; then
    echo "[$(date -Is)] policy=${CANDIDATE} seed=${seed} floor=${OLDEST_FLOOR} skip (exists)" \
      | tee -a "$RESULT_DIR/progress_floor${OLDEST_FLOOR}.log"
    continue
  fi

  echo "[$(date -Is)] policy=${CANDIDATE} seed=${seed} floor=${OLDEST_FLOOR} start" \
    | tee -a "$RESULT_DIR/progress_floor${OLDEST_FLOOR}.log"
  python3 scripts/run_split_mnist.py mnist \
    --nodes 50-25-1 --lr 0.001 \
    --min-segment 512 --pool 80 --pool-reservoir 10 \
    --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
    --pool-evict-policy "$CANDIDATE" --pool-oldest-floor "$OLDEST_FLOOR" \
    --active-state per-level --prediction-mode ptw_dp \
    --chunk-size 1024 --posterior-temp 1.0 \
    --adapt-n 50 \
    --seed "$seed" --json-out "$OUT" > "$LOG" 2>&1
  grep -E "^(Average Accuracy|Average Forgetting|Total time)" "$LOG" \
    | tee -a "$RESULT_DIR/progress_floor${OLDEST_FLOOR}.log" || true
  echo "[$(date -Is)] policy=${CANDIDATE} seed=${seed} floor=${OLDEST_FLOOR} done" \
    | tee -a "$RESULT_DIR/progress_floor${OLDEST_FLOOR}.log"
done

POOL_OLDEST_FLOOR="$OLDEST_FLOOR" \
  python3 scripts/sweep_results/phase5o_age_diversity_oldest_floor/summarise_nofifo.py
