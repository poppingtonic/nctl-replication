#!/usr/bin/env bash
# 30.10 single-seed pool ablation.
#
# Usage: bash run_single_seed.sh POOL [SEED]
#
# Recipe identical to phase5i_c_5seed/run_relaunch_pool80_20260525.sh except
# --pool changes.  Acceptance: seed=1 pool=120 >=92% promotes to 5-seed.
# Pool=160 only after pool=120 promising AND post-allocation GPU memory
# audit passes (see run_pool160_safety_check.py).
set -euo pipefail
cd "$(dirname "$0")/../../../.."
POOL=${1:?usage: bash run_single_seed.sh POOL [SEED]}
SEED=${2:-1}
RESULT_DIR="rust-fmn/scripts/sweep_results/phase5j_pool_ablation"
mkdir -p "$RESULT_DIR"
OUT="$RESULT_DIR/seed${SEED}_pool${POOL}.json"
LOG="${OUT%.json}.log"
echo "[$(date -Is)] pool=${POOL} seed=${SEED} start" \
  | tee -a "$RESULT_DIR/progress.log"
python rust-fmn/scripts/run_split_mnist.py mnist \
  --nodes 50-25-1 --lr 0.001 \
  --min-segment 512 --pool "$POOL" --pool-reservoir 10 \
  --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
  --active-state per-level --prediction-mode ptw_dp \
  --chunk-size 1024 --posterior-temp 1.0 \
  --seed "$SEED" --json-out "$OUT" > "$LOG" 2>&1
grep -E "^(Average Accuracy|Average Forgetting|Total time|Pool:)" "$LOG" \
  | tee -a "$RESULT_DIR/progress.log"
echo "[$(date -Is)] pool=${POOL} seed=${SEED} done" \
  | tee -a "$RESULT_DIR/progress.log"
