"""Sparse Scheme-A quick comparison with a fixed circular k-regular graph.

Entity indices form the circle 0,1,...,K-1,0.  For even k, each entity connects
to k/2 predecessors and k/2 successors, giving exact degree k and guaranteed
connectivity without topology repair.  A entities retain four anchor ranges, B
entities retain two, and C entities retain none.  All methods receive the same
noisy edge list.  Primary consistency metric is per-seed post-burn-in time-mean
normalized NEES.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from config import apply_entity_count, default_config, validate_config
from l4_comparison import run_scenario

RETAINED_ARCHITECTURES = ["centralized", "dual_layer", "ci_trace"]
SENSOR_ABLATIONS = ["imu_only", "uwb_only"]


def build_config(args):
    cfg = default_config()
    apply_entity_count(cfg, 10, (2, 6, 2))

    cfg["estimators"] = list(RETAINED_ARCHITECTURES)
    cfg["ablation_estimators"] = list(SENSOR_ABLATIONS)
    cfg["comparison_require_full_set"] = False
    cfg["comparison_run_extension"] = False
    cfg["comparison_entity_counts"] = [10]
    cfg["comparison_class_sizes"] = {10: (2, 6, 2)}

    cfg["entity_topology"] = "knn"
    cfg["knn_k"] = int(args.neighbors)
    cfg["anchor_visible_by_class"] = {
        "A": [0, 1, 2, 3],
        "B": [0, 1],
        "C": [],
    }

    cfg["traj_type"] = "random_accel"
    cfg["speed"] = float(args.speed)
    cfg["target_speed"] = float(args.speed)
    cfg["target_speed_gain"] = float(args.target_speed_gain)
    cfg["sync_interval"] = float(args.interval)
    cfg["dt_imu"] = float(args.dt_imu)
    epochs = max(
        2,
        int(math.ceil(float(args.duration) / float(args.interval) - 1e-12)),
    )
    cfg["sim_duration"] = epochs * cfg["sync_interval"]

    cfg["max_accel"] = float(args.max_accel)
    cfg["max_turn_rate"] = float(args.max_turn_rate)
    cfg["random_hold_time"] = float(args.random_hold_time)
    cfg["random_vertical_scale"] = float(args.random_vertical_scale)
    cfg["xy_boundary_margin"] = float(args.xy_boundary_margin)
    cfg["xy_boundary_k"] = float(args.xy_boundary_k)
    cfg["xy_boundary_damping"] = float(args.xy_boundary_damping)
    cfg["random_velocity_damping"] = float(args.velocity_damping)

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

    cfg["mc_runs"] = int(args.runs)
    cfg["seed_base"] = int(args.seed_base)
    cfg["comparison_burn_in_frac"] = float(args.burn_in)
    cfg["nees_bootstrap_confidence"] = float(args.bootstrap_confidence)
    cfg["nees_bootstrap_resamples"] = int(args.bootstrap_resamples)
    cfg["nees_bootstrap_seed"] = int(args.bootstrap_seed)

    cfg["comparison_warmup"] = bool(args.warmup)
    cfg["runtime_repeats"] = 1
    cfg["save_plots"] = not args.no_plots
    cfg["run_preflight"] = False
    cfg["separation_check"] = False
    cfg["enable_runtime_profile"] = bool(args.profile_runtime)
    cfg["output_dir"] = str(Path(args.output))
    cfg["uwb_only_pos_rw_psd"] = (
        cfg["target_speed"] ** 2 * cfg["sync_interval"]
    )

    validate_config(cfg)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sparse fixed-circular Scheme-A quick retained-method comparison")
    parser.add_argument("--neighbors", type=int, default=4)

    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.20)
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
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output", default="results_m21_quick_all_methods_sparse_knn4")
    args = parser.parse_args(argv)

    if args.neighbors < 2 or args.neighbors >= 10:
        parser.error("neighbors must lie in [2, 9]")
    if args.neighbors % 2 != 0:
        parser.error("neighbors must be even for the fixed circular topology")
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
        "[M21 sparse quick] "
        f"K=10 classes=(2,6,2), topology=fixed-circular-k-regular(k={cfg['knn_k']}), "
        f"anchors=A4/B2/C0, methods={methods}, "
        f"target_speed_3d={cfg['target_speed']:.2f}m/s, "
        f"T={cfg['sync_interval']:.3f}s, "
        f"duration={cfg['sim_duration']:.1f}s, seeds={seeds}",
        flush=True,
    )

    payload = run_scenario(
        cfg, burn_in_frac=cfg["comparison_burn_in_frac"], verbose=True)

    print("\n=== Compact retained-method summary ===")
    print(
        "  %-12s %9s %9s %9s %9s %10s %10s" %
        ("method", "RMSE(m)", "RMTE(m)", "TMNNEES", "NNEESstd",
         "meas(KiB)", "est(KiB)"))
    for name in methods:
        row = payload["methods"][name]
        print(
            "  %-12s %9.4f %9.4f %9.3f %9.3f %10.1f %10.1f" %
            (
                name,
                row["pos_rmse"],
                row["rmte"],
                row["time_mean_nnees"],
                row["time_mean_nnees_std"],
                row["comm_measurement_bytes"] / 1024.0,
                row["comm_estimation_bytes"] / 1024.0,
            ))

    print("\nFixed circular topology by seed:")
    for row in payload["metadata"]["motion_diagnostics"]:
        print(
            f"  seed={int(row['seed'])}: "
            f"inter edges mean/min/max="
            f"{row.get('inter_edges_mean', float('nan')):.2f}/"
            f"{row.get('inter_edges_min', float('nan')):.0f}/"
            f"{row.get('inter_edges_max', float('nan')):.0f}, "
            f"degree mean/min/max="
            f"{row.get('mean_degree_mean', float('nan')):.2f}/"
            f"{row.get('min_degree_min', float('nan')):.0f}/"
            f"{row.get('max_degree_max', float('nan')):.0f}")

    print("\nPaired TMNNEES gaps versus centralized:")
    for name, row in payload[
            "paired_time_mean_nnees_vs_centralized"].items():
        lo, hi = row["bootstrap_ci"]
        print(
            f"  {name:12s}: mean={row['mean']:+.3f}, "
            f"std={row['std']:.3f}, bootstrap CI=[{lo:+.3f}, {hi:+.3f}]")

    output = Path(cfg["output_dir"]) / "comparison_results.json"
    print(f"\n[M21 sparse quick] results -> {output}")


if __name__ == "__main__":
    main()
