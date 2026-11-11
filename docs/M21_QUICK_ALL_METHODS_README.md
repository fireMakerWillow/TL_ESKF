# Quick all-method M21 comparison

This package runs one K=10 (2/6/2) operating point with all seven methods:

- centralized
- dual_layer
- local_only
- ci_trace
- ci_det
- imu_only
- uwb_only

The default run uses three seeds, bounded random 3-D motion, sustained 3-D
target speed, smooth x-y boundary recovery, and distinct per-entity z reference
heights.

The primary consistency metric is TMNNEES:

1. remove burn-in inside each seed;
2. average the NEES time series for that seed;
3. divide by the relevant DoF;
4. summarize the three independent seed values.

Final-epoch ANEES remains in the output only as an audit field.

Run:

    python -u run_m21_quick_all_methods.py

Example:

    python -u run_m21_quick_all_methods.py         --speed 5 --interval 0.05 --duration 20         --seed-base 1010 --runs 3
