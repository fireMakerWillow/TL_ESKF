"""Run a lightweight all-method M21 comparison at one (speed, T) point.

Default experiment:
- K=10, class split A/B/C = 2/6/2;
- all 7 methods;
- bounded random 3-D motion;
- per-entity height references and z boundary control;
- sustained full-3D target speed;
- 3 independent seeds;
- per-seed post-burn-in time-mean normalized NEES.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from config import (
    MAIN_LINE_ESTIMATORS,
    SENSOR_ABLATION_ESTIMATORS,
    apply_entity_count,
    default_config,
    validate_config,
)
from l4_comparison import run_scenario


def build_config(args):
    cfg = default_config()
    apply_entity_count(cfg, 10, (2, 6, 2))

    # Use every architecture and sensor-ablation method.
    cfg["estimators"] = list(MAIN_LINE_ESTIMATORS)
    cfg["ablation_estimators"] = list(SENSOR_ABLATION_ESTIMATORS)
    cfg["comparison_require_full_set"] = True
    cfg["comparison_run_extension"] = False
    cfg["comparison_entity_counts"] = [10]
    cfg["comparison_class_sizes"] = {10: (2, 6, 2)}

    # One random-motion operating point.
    cfg["traj_type"] = "random_accel"
    cfg["speed"] = float(args.speed)
    cfg["target_speed"] = float(args.speed)
    cfg["target_speed_gain"] = float(args.target_speed_gain)
    cfg["sync_interval"] = float(args.interval)
    cfg["dt_imu"] = float(args.dt_imu)

    epochs = max(2, int(math.ceil(
        float(args.duration) / float(args.interval) - 1e-12)))
    cfg["sim_duration"] = epochs * cfg["sync_interval"]

    # Random maneuver and smooth x-y boundary control.
    cfg["max_accel"] = float(args.max_accel)
    cfg["max_turn_rate"] = float(args.max_turn_rate)
    cfg["random_hold_time"] = float(args.random_hold_time)
    cfg["random_vertical_scale"] = float(args.random_vertical_scale)
    cfg["xy_boundary_margin"] = float(args.xy_boundary_margin)
    cfg["xy_boundary_k"] = float(args.xy_boundary_k)
    cfg["xy_boundary_damping"] = float(args.xy_boundary_damping)
    cfg["random_velocity_damping"] = float(args.velocity_damping)

    # Diverse per-entity heights; do not collapse all entities to z=25 m.
    cfg["z_reference"] = float(args.z_reference)
    cfg["z_reference_spread"] = float(args.z_reference_spread)
    cfg["z_reference_jitter"] = float(args.z_reference_jitter)
    cfg["z_initial_spread"] = float(args.z_initial_spread)
    cfg["z_min"] = float(args.z_min)
    cfg["z_max"] = float(args.z_max)
    cfg["z_restore_k"] = float(args.z_restore_k)
    cfg["z_velocity_damping"] = float(args.z_velocity_damping)
    cfg["z_boundary_margin"] = float(args.z_boundary_margin)
    cfg["z_boundary_k"] = float(args.z_boundary_k)

    # Three independent seeds and time-mean NEES aggregation.
    cfg["mc_runs"] = int(args.runs)
    cfg["seed_base"] = int(args.seed_base)
    cfg["comparison_burn_in_frac"] = float(args.burn_in)
    cfg["nees_bootstrap_confidence"] = float(args.bootstrap_confidence)
    cfg["nees_bootstrap_resamples"] = int(args.bootstrap_resamples)
    cfg["nees_bootstrap_seed"] = int(args.bootstrap_seed)

    # Keep this quick.
    cfg["comparison_warmup"] = bool(args.warmup)
    cfg["runtime_repeats"] = 1
    cfg["save_plots"] = not args.no_plots
    cfg["run_preflight"] = False
    cfg["separation_check"] = False
    cfg["enable_runtime_profile"] = False
    cfg["output_dir"] = str(Path(args.output))

    # Keep the UWB-only random-walk tuning aligned with this operating point.
    cfg["uwb_only_pos_rw_psd"] = (
        cfg["target_speed"] ** 2 * cfg["sync_interval"]
    )

    validate_config(cfg)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Quick all-method comparison with 3-seed TMNNEES")

    parser.add_argument("--speed", type=float, default=5.0,
                        help="full-3D target speed in m/s")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="UWB/synchronization interval in seconds")
    parser.add_argument("--duration", type=float, default=50.0)
    parser.add_argument("--dt-imu", type=float, default=0.005)
    parser.add_argument("--seed-base", type=int, default=1010)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--burn-in", type=float, default=0.25)

    parser.add_argument("--max-accel", type=float, default=4.0)
    parser.add_argument("--max-turn-rate", type=float, default=0.8)
    parser.add_argument("--random-hold-time", type=float, default=0.50)
    parser.add_argument("--random-vertical-scale", type=float, default=0.20)
    parser.add_argument("--target-speed-gain", type=float, default=0.80)

    parser.add_argument("--xy-boundary-margin", type=float, default=20.0)
    parser.add_argument("--xy-boundary-k", type=float, default=0.20)
    parser.add_argument("--xy-boundary-damping", type=float, default=0.80)
    parser.add_argument("--velocity-damping", type=float, default=0.03)

    parser.add_argument("--z-reference", type=float, default=25.0)
    parser.add_argument("--z-reference-spread", type=float, default=30.0)
    parser.add_argument("--z-reference-jitter", type=float, default=1.0)
    parser.add_argument("--z-initial-spread", type=float, default=2.0)
    parser.add_argument("--z-min", type=float, default=5.0)
    parser.add_argument("--z-max", type=float, default=45.0)
    parser.add_argument("--z-restore-k", type=float, default=0.035)
    parser.add_argument("--z-velocity-damping", type=float, default=0.25)
    parser.add_argument("--z-boundary-margin", type=float, default=3.0)
    parser.add_argument("--z-boundary-k", type=float, default=0.60)

    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=24680)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output", default="results_m21_quick_all_methods")

    args = parser.parse_args(argv)

    if args.speed < 0:
        parser.error("speed must be non-negative")
    if args.interval <= 0 or args.duration <= 0 or args.dt_imu <= 0:
        parser.error("interval, duration and dt-imu must be positive")
    if args.runs < 1:
        parser.error("runs must be at least 1")
    if not 0 <= args.burn_in < 1:
        parser.error("burn-in must lie in [0,1)")
    if not 0 < args.bootstrap_confidence < 1:
        parser.error("bootstrap-confidence must lie in (0,1)")
    if args.bootstrap_resamples < 0:
        parser.error("bootstrap-resamples must be non-negative")

    cfg = build_config(args)
    methods = list(cfg["estimators"]) + list(cfg["ablation_estimators"])
    seeds = [cfg["seed_base"] + i for i in range(cfg["mc_runs"])]

    print(
        "[M21 quick] "
        f"K=10 classes=(2,6,2), methods={methods}, "
        f"target_speed_3d={cfg['target_speed']:.2f}m/s, "
        f"T={cfg['sync_interval']:.3f}s, "
        f"duration={cfg['sim_duration']:.1f}s, seeds={seeds}",
        flush=True,
    )

    payload = run_scenario(
        cfg, burn_in_frac=cfg["comparison_burn_in_frac"], verbose=True)

    print("\n=== Compact all-method summary ===")
    print(
        "  %-12s %9s %9s %9s %9s %9s" %
        ("method", "RMSE(m)", "RMTE(m)", "TMNNEES", "NNEESstd", "runtime(s)")
    )
    for name in methods:
        row = payload["methods"][name]
        print(
            "  %-12s %9.4f %9.4f %9.3f %9.3f %9.3f" %
            (
                name,
                row["pos_rmse"],
                row["rmte"],
                row["time_mean_nnees"],
                row["time_mean_nnees_std"],
                row["runtime_total_s"],
            )
        )

    print("\nPaired TMNNEES gaps versus centralized:")
    for name, row in payload[
            "paired_time_mean_nnees_vs_centralized"].items():
        lo, hi = row["bootstrap_ci"]
        print(
            f"  {name:12s}: mean={row['mean']:+.3f}, "
            f"std={row['std']:.3f}, bootstrap CI=[{lo:+.3f}, {hi:+.3f}]"
        )

    output = Path(cfg["output_dir"]) / "comparison_results.json"
    print(f"\n[M21 quick] results -> {output}")


if __name__ == "__main__":
    main()
