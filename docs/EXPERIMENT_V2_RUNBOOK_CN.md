# 实验设计 v2：完整运行说明

本版本包含 25 个 E1 加速度—同步周期格点，去重后共 40 个物理条件。名义条件为 `C008`。E1 默认 50 个种子，其余条件默认 30 个种子；结构诊断只在配置允许的条件上对 `1010:1014` 五个种子执行。

## 1. 环境与自检

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
```

## 2. 运动学网格验证

```bash
python validate_maneuver_grid.py --seeds 1010 --duration 120 --burn-in 20 --strict --workers 1 --output maneuver_validation.json
```

## 3. 检查实验清单

```bash
python run_large_scale_parallel.py --dry-run --workers 1 --output results_experiment_v2
```

## 4. 名义条件短跑

```bash
python run_large_scale_parallel.py --conditions C008 --seeds 1010 --duration 2 --burn-in 0.5 --diag-stride 0.5 --workers 1 --output results_smoke
python collect_results.py --results-root results_smoke --output analysis_smoke --level core --bundle
```

## 5. 全量实验

```bash
python run_large_scale_parallel.py --workers 8 --output results_experiment_v2
```

同一命令可断点续跑。程序会跳过状态为 `complete` 的结果文件。

## 6. 数据汇总

论文主线数据，默认排除 CI-det：

```bash
python collect_results.py --results-root results_experiment_v2 --output analysis_tables --level core --bundle
```

保留每个坐标分量与 RMTE：

```bash
python collect_results.py --results-root results_experiment_v2 --output analysis_tables_full --level full --bundle
```

需要查看 CI-det 脚注数据时：

```bash
python collect_results.py --results-root results_experiment_v2 --output analysis_tables_all_methods --level core --methods all --bundle
```
