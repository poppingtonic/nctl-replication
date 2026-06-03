# phase5i_c_5seed backup: pre_5id_clone_eval_20260525
This backup preserves the existing pool=80 seed outputs before relaunching the 5-seed sweep after Phase 5I-D.

## Why this backup exists
The existing files were produced before the in-place `evaluate_task` snapshot/restore path landed. They are valid diagnostic artifacts, but the relaunch should regenerate all seeds under the current code path for consistency.

## Backed up files
- `seed1_pool80.json`: seed=1, avg_accuracy=91.146700%, avg_forgetting=9.382769%, total_time=490.2s
- `seed2_pool80.json`: seed=2, avg_accuracy=91.370467%, avg_forgetting=9.101454%, total_time=483.6s

## Aggregate
- n=2
- avg_accuracy_mean=91.258584%
- avg_accuracy_std_pop=0.111883
- avg_forgetting_mean=9.242111%
- avg_forgetting_std_pop=0.140657

## Relaunch plan
After this backup, remove/regenerate the top-level `seed*_pool80.json` files in `phase5i_c_5seed` so the final 5-seed mean reflects the Phase 5I-D in-place evaluation path.

## 2026-05-25 detached relaunch attempt failed
A first relaunch attempt (`run_relaunch_pool80_20260525.sh` via `nohup
& disown`) was killed by the sandbox when the parent shell exited. PID
9338 vanished before seed=1 produced any output (0-byte log).
`failed_detached_attempt.log` and `failed_detached_attempt.nohup.log`
in this directory preserve what little was emitted. The actual 5-seed
sweep is being re-run synchronously per seed under the unified exec
session that keeps the parent process alive for the full duration.
