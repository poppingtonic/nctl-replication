#!/usr/bin/env bash
# bd 30.12 selected-config 5-seed confirmation after the seed=1 grid.
#
# Usage:
#   bash run_confirm_selected.sh ADAPT_N POSTERIOR_TEMP [SEEDS...]
# Examples:
#   bash run_confirm_selected.sh 200 2.0 1 2 3 4 5
#   bash run_confirm_selected.sh 1000 2.0 1 2 3 4 5
#
# All cells use --eval-suffix-from 1000, matching the fair common-suffix
# grid that produced seed=1 96.65% at adapt=1000,temp=2.0.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
ADAPT=${1:?usage: bash run_confirm_selected.sh ADAPT_N POSTERIOR_TEMP [SEEDS...]}
TEMP=${2:?usage: bash run_confirm_selected.sh ADAPT_N POSTERIOR_TEMP [SEEDS...]}
shift 2
if [[ $# -eq 0 ]]; then
  set -- 1 2 3 4 5
fi
RESULT_DIR="rust-fmn/scripts/sweep_results/phase5l_adapt_temp"
EVAL_OFFSET=1000
for seed in "$@"; do
  tag="adapt${ADAPT}_temp${TEMP}"
  OUT="$RESULT_DIR/seed${seed}_${tag}.json"
  LOG="${OUT%.json}.log"
  if [[ -s "$OUT" ]]; then
    echo "[$(date -Is)] seed=${seed} ${tag} skip (exists)" \
      | tee -a "$RESULT_DIR/progress_confirm.log"
    continue
  fi
  echo "[$(date -Is)] seed=${seed} ${tag} start" \
    | tee -a "$RESULT_DIR/progress_confirm.log"
  python rust-fmn/scripts/run_split_mnist.py mnist \
    --nodes 50-25-1 --lr 0.001 \
    --min-segment 512 --pool 80 --pool-reservoir 10 \
    --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
    --active-state per-level --prediction-mode ptw_dp \
    --chunk-size 1024 \
    --posterior-temp "$TEMP" \
    --adapt-n "$ADAPT" --eval-suffix-from "$EVAL_OFFSET" \
    --seed "$seed" --json-out "$OUT" > "$LOG" 2>&1
  grep -E "^(Average Accuracy|Average Forgetting|Total time)" "$LOG" \
    | tee -a "$RESULT_DIR/progress_confirm.log"
  echo "[$(date -Is)] seed=${seed} ${tag} done" \
    | tee -a "$RESULT_DIR/progress_confirm.log"
done
python3 rust-fmn/scripts/sweep_results/phase5l_adapt_temp/summarise_selected.py \
  "$ADAPT" "$TEMP" "$@"
