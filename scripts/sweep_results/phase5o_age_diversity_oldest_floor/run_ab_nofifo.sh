#!/usr/bin/env bash
# 30.13 follow-up A/B sweep: fifo (regression parity) vs
# age-diversity-oldest-floor task-free retention.
#
# Holds the 30.9 canonical recipe fixed at pool=80 so the only difference
# between arms is the pool eviction policy.  The fifo arm must reproduce
# phase5k_task_floor's 90.94% +/- 0.31 control mean before the candidate
# delta is interpreted.  The candidate protects the POOL_OLDEST_FLOOR oldest
# snapshots, then runs age-diversity among the remainder -- targeting the
# earliest-task starvation that plain age-diversity (phase5m, 94.23%) left.

# floor=4 without fifo
set -euo pipefail
cd "$(dirname "$0")/../../../.."
for seed in 1 2 3 4 5; do
  OUT="scripts/sweep_results/phase5o_age_diversity_oldest_floor/age-diversity-oldest-floor_floor4_seed${seed}.json"
  LOG="${OUT%.json}.log"
  python3 scripts/run_split_mnist.py mnist \
    --nodes 50-25-1 --lr 0.001 \
    --min-segment 512 --pool 80 --pool-reservoir 10 \
    --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
    --pool-evict-policy age-diversity-oldest-floor --pool-oldest-floor 4 \
    --active-state per-level --prediction-mode ptw_dp \
    --chunk-size 1024 --posterior-temp 1.0 \
    --seed "$seed" --json-out "$OUT" > "$LOG" 2>&1
  grep -E "^(Average Accuracy|Average Forgetting|Total time)" "$LOG" || true
done
python3 scripts/sweep_results/phase5o_age_diversity_oldest_floor/summarise_nofifo.py
