# 120 s large-scale experiment

## Frozen protocol

- Methods: `centralized`, `dual_layer`, `ci_trace`, `ci_det`
- Duration: 120 s
- Burn-in: 20 s
- Seeds: 1010–1039
- Unique conditions: 33
- Condition–seed tasks: 990
- Method runs: 3960
- Motion: bounded random 3-D motion with sustained 3-D target speed
- Output: block/component RMSE, RMTE and TMNNEES; full-state TMNNEES only
- Resources: theoretical communication, storage and FLOP counts; no measured runtime
- Sparse topology: fixed circular even-degree graph; all configured `knn_k` values are even

## Balanced class-B anchors

Class B still sees exactly two anchors per entity, but B entities no longer all
share `[0, 1]`.  The assignment is derived from anchor heights and balanced
across the four anchors.  For the current geometry the repeating cycle is:

```text
[0, 1], [3, 2], [0, 2], [3, 1]
```

Every pair contains one elevated and one lower anchor.  For six B entities the
anchor-use counts are exactly `[3, 3, 3, 3]`.  The exact map used by every run
is stored in `config.anchor_visibility_by_entity`.

## Run with 100 processes

```bash
python -u run_large_scale_parallel.py \
  --workers 100 \
  --output results_large_scale_120s
```

The worker count is capped at 100 processes.  Each task runs all four methods on one shared
condition/seed scenario.  Completed JSON files are skipped automatically when
the command is restarted.

The launcher forces BLAS/OpenMP thread counts to one, preventing 100 Python
processes from each creating their own multithreaded BLAS pool. Ensure the
server has enough RAM for 100 simultaneous condition/seed tasks; otherwise use
a smaller `--workers` value without changing the experiment protocol.

Before starting the full run:

```bash
python check_large_scale_revision.py
python -u run_large_scale_parallel.py --dry-run
```

For a scheduler smoke test, submit only one task.  This still uses the frozen
120 s duration:

```bash
python -u run_large_scale_parallel.py --workers 1 --max-tasks 1 \
  --output results_large_scale_smoke
```

## Output layout

```text
results_large_scale_120s/
├── experiment_manifest.json
└── runs/
    └── C001/
        ├── seed_1010.json
        ├── seed_1011.json
        └── ...
```

Each result is written through a temporary file and atomically renamed.  A
failed task is retained with its exception and traceback instead of being
silently discarded.

## Memory behaviour

The large-scale path sets `large_scale_summary_only=True`.  It streams the
post-burn-in metrics and does not retain truth, state, error or covariance
histories.  This is essential when many 120 s trials run concurrently.
