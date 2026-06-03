#!/usr/bin/env bash
set -euo pipefail
cd /data/backup-src/src/aixi/aixictwxcode/forget-me-not
RESULT_DIR="rust-fmn/scripts/sweep_results/phase5i_c_5seed"
mkdir -p "$RESULT_DIR"
for seed in 1 2 3 4 5; do
  out="$RESULT_DIR/seed${seed}_pool80.json"
  log="$RESULT_DIR/seed${seed}_pool80.log"
  echo "[$(date -Is)] seed=${seed} start" | tee -a "$RESULT_DIR/relaunch_pool80_5seed_20260525.progress.log"
  python rust-fmn/scripts/run_split_mnist.py mnist \
    --nodes 50-25-1 --lr 0.001 \
    --min-segment 512 --pool 80 --pool-reservoir 10 \
    --pool-update-policy paper --pool-alpha 0.0 --pool-beta 0.0 \
    --active-state per-level --prediction-mode ptw_dp \
    --chunk-size 1024 --posterior-temp 1.0 \
    --seed "$seed" --json-out "$out" > "$log" 2>&1
  grep -E "^(Average Accuracy|Average Forgetting|Total time)" "$log" | tee -a "$RESULT_DIR/relaunch_pool80_5seed_20260525.progress.log"
  echo "[$(date -Is)] seed=${seed} done" | tee -a "$RESULT_DIR/relaunch_pool80_5seed_20260525.progress.log"
done
python3 - <<'PYEOF'
import json, pathlib, statistics, time
root=pathlib.Path('rust-fmn/scripts/sweep_results/phase5i_c_5seed')
files=sorted(root.glob('seed*_pool80.json'))
rows=[]
for f in files:
    d=json.loads(f.read_text())
    rows.append({
        'file': f.name,
        'seed': d.get('seed'),
        'avg_accuracy': d.get('avg_accuracy'),
        'avg_forgetting': d.get('avg_forgetting'),
        'total_time': d.get('total_time'),
        'task_histogram': (d.get('pool_provenance') or {}).get('task_histogram'),
        'event_counts': (d.get('pool_provenance') or {}).get('event_counts'),
    })
acc=[r['avg_accuracy'] for r in rows]
fgt=[r['avg_forgetting'] for r in rows]
summary={
    'experiment':'phase5i_c_5seed',
    'recipe':'pool80_ms512_temp1_ab00_in_place_eval_20260525',
    'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'n':len(rows),
    'results':rows,
    'avg_accuracy_mean':statistics.mean(acc) if acc else None,
    'avg_accuracy_std_pop':statistics.pstdev(acc) if len(acc)>1 else None,
    'avg_accuracy_min':min(acc) if acc else None,
    'avg_accuracy_max':max(acc) if acc else None,
    'avg_forgetting_mean':statistics.mean(fgt) if fgt else None,
    'avg_forgetting_std_pop':statistics.pstdev(fgt) if len(fgt)>1 else None,
}
(root/'summary_relaunched_20260525.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PYEOF
