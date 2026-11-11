# Experiment design v2 runbook

The protocol has 25 E1 grid points and 40 unique physical conditions. The
nominal condition is C008. E1 uses 50 seeds; all other conditions use 30.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r ../requirements.txt
python -m compileall -q .
python check_large_scale_revision.py
python l1_diagnostic.py
python l1_resource.py
python check_l3_estimators.py --skip-units --only=registry,T11,T16
python check_experiment_v2.py

python validate_maneuver_grid.py --seeds 1010 --duration 120 --burn-in 20 --strict --workers 1 --output maneuver_validation.json

python run_large_scale_parallel.py --dry-run --workers 1 --output results_experiment_v2
python run_large_scale_parallel.py --conditions C008 --seeds 1010 --duration 2 --burn-in 0.5 --workers 1 --output results_smoke
python collect_results.py --results-root results_smoke --output analysis_smoke --level core --bundle

python run_large_scale_parallel.py --workers 8 --output results_experiment_v2
python collect_results.py --results-root results_experiment_v2 --output analysis_tables --level core --bundle
python collect_results.py --results-root results_experiment_v2 --output analysis_tables_all_methods --level core --methods all --bundle
```
