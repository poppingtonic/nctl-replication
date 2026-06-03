# Sweep Results and Harnesses

This directory contains both reproducible sweep harnesses and, in some copies of
the artifact, generated JSON/log outputs from prior runs.

Curated release harnesses:

- `phase5m_age_diversity/` - FIFO vs task-free age-diversity eviction.
- `phase5n_age_bucket_floor/` - FIFO vs log-age-bucket retention.
- `phase5o_age_diversity_oldest_floor/` - FIFO/floor-2, floor-4, and floor-6
  `age-diversity-oldest-floor` runs.

Diagnostic or historical directories are retained only when they help explain
the search path.  Treat `.log` files, MLflow outputs, and ad-hoc trial
directories as generated artifacts rather than source.

For the newest strict candidate workflow, start with:

```bash
bash scripts/sweep_results/phase5o_age_diversity_oldest_floor/run_floor6_nofifo.sh
```

That reruns the floor-6 five-seed candidate and writes
`summary_floor6_nofifo.json`. The older floor-4 candidate-only harness remains
available as `run_ab_nofifo.sh`.
