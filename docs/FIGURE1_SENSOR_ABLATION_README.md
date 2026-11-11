# Figure 1 传感器消融补跑

新增脚本：`run_figure1_sensor_ablation.py`

## 正式实验条件

- Condition: C008
- K = 10
- A/B/C = 2/6/2
- Target speed = 5 m/s
- Synchronization interval = 0.10 s
- Topology = full
- Duration = 120 s
- Burn-in = 20 s
- Seeds = 1010–1039
- Methods = `imu_only`, `uwb_only`

脚本直接复用 `run_large_scale_parallel.py` 中的 `build_conditions()` 和
`_build_config()`，因此锚点、类别随机排列、随机轨迹、噪声、初始误差、
数值配置和正式 C008 基准保持一致。每个 seed 内两种方法共享同一组
truth/IMU/UWB 数据。

## 推荐运行命令

在项目根目录运行：

```bash
python -u src/run_figure1_sensor_ablation.py --workers 30
```

默认输出目录：

```text
results_large_scale_120s/figure1_sensor_ablation/
```

如果在 `src` 目录内运行，也可直接执行：

```bash
python -u run_figure1_sensor_ablation.py --workers 30
```

默认路径按脚本位置解析，不受当前工作目录影响。

## 断点续跑

默认自动跳过已有且状态为 `complete` 的 seed。重新运行相同命令即可补跑失败或缺失任务。

强制重跑全部任务：

```bash
python -u src/run_figure1_sensor_ablation.py --workers 30 --no-resume
```

仅重新汇总已有结果：

```bash
python -u src/run_figure1_sensor_ablation.py --aggregate-only
```

查看任务而不运行：

```bash
python -u src/run_figure1_sensor_ablation.py --dry-run
```

## 输出文件

```text
figure1_sensor_ablation/
├── experiment_manifest.json
├── runs/
│   └── C008/
│       ├── seed_1010.json
│       └── ...
├── sensor_ablation_per_seed_metrics.csv
├── sensor_ablation_aggregate_metrics.csv
├── comparison_sensor_ablations.csv
├── comparison_results.json
├── reference_match_report.json
└── ablation_summary.json
```

`comparison_sensor_ablations.csv` 可以直接作为现有绘图程序的输入：

```bash
python -u src/plot_reduced_paper_figures.py   --ablation-file results_large_scale_120s/figure1_sensor_ablation/comparison_sensor_ablations.csv
```

## 配对检查

脚本默认读取已有正式结果：

```text
results_large_scale_120s/runs/C008/seed_XXXX.json
```

并检查：

- `config_hash` 是否一致；
- `motion_summary` 是否一致。

已有参考文件若出现不一致，程序返回非零退出码并写入
`reference_match_report.json`。参考文件不存在不会阻止补跑。

不进行参考检查：

```bash
python -u src/run_figure1_sensor_ablation.py --no-reference-check
```
