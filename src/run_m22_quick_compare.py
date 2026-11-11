"""Fast one-point random-motion comparison: centralized versus dual-layer.

The quick probe is deliberately single-seed.  It visualises raw and moving-
average normalized NEES, while the JSON summary uses the post-burn-in time mean
for each method/class.  Formal multi-seed bootstrap reporting is implemented in
``l4_comparison.py`` and ``l4_failure_boundary.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()
for _p in (_THIS_DIR, _CWD):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import apply_entity_count, default_config, validate_config
from l3_runner import run_trial


def _serialisable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(v) for v in value]
    return value


def _finite(values):
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _moving_average(values, window):
    x = np.asarray(values, dtype=float).ravel()
    window = max(1, int(window))
    if window == 1:
        return x.copy()
    out = np.full(x.shape, np.nan, dtype=float)
    finite = np.isfinite(x)
    sums = np.convolve(np.where(finite, x, 0.0), np.ones(window), mode="valid")
    counts = np.convolve(finite.astype(float), np.ones(window), mode="valid")
    valid = counts > 0
    segment = np.full(sums.shape, np.nan)
    segment[valid] = sums[valid] / counts[valid]
    out[window - 1:] = segment
    return out


def build_config(args):
    cfg = default_config()
    apply_entity_count(cfg, 10, (2, 6, 2))

    cfg["traj_type"] = args.traj_type
    cfg["speed"] = float(args.speed)
    cfg["target_speed"] = float(args.speed)
    cfg["sync_interval"] = float(args.interval)
    cfg["dt_imu"] = float(args.dt_imu)

    epochs = max(2, int(math.ceil(args.duration / args.interval - 1e-12)))
    cfg["sim_duration"] = epochs * cfg["sync_interval"]
    cfg["m22_fixed_duration_s"] = cfg["sim_duration"]

    cfg["max_accel"] = float(args.max_accel)
    cfg["max_turn_rate"] = float(args.max_turn_rate)
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
    cfg["uwb_only_pos_rw_psd"] = cfg["speed"] ** 2 * cfg["sync_interval"]

    cfg["mc_runs"] = 1
    cfg["seed_base"] = int(args.seed)
    cfg["save_plots"] = False
    cfg["run_preflight"] = False
    cfg["separation_check"] = False
    cfg["enable_runtime_profile"] = False

    validate_config(cfg)
    return cfg, epochs


def pick_representative_entities(class_map):
    return {
        label: next(i for i, c in enumerate(class_map) if c == label)
        for label in ("A", "B", "C")
    }


def time_axis(config):
    epochs = int(round(config["sim_duration"] / config["sync_interval"]))
    return np.arange(1, epochs + 1, dtype=float) * float(config["sync_interval"])


def save_prediction_error_plot(outdir, label, entity_index, t,
                               truth_p, dual_p, central_p):
    """Save truth-trajectory context plus x/y/z prediction errors.

    The first two panels show the representative entity's true motion:
      1) x-y ground-track truth;
      2) true z (height) versus time.

    The remaining three panels compare dual-layer and centralized
    prediction errors along x, y, and z.
    """
    truth = np.asarray(truth_p[:, entity_index, :], dtype=float)
    dual_err = dual_p[:, entity_index, :] - truth
    central_err = central_p[:, entity_index, :] - truth

    fig = plt.figure(figsize=(8.2, 12.0))
    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[1.35, 1.0, 1.0, 1.0, 1.0],
        hspace=0.38,
    )

    # (1) True x-y ground track.
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xy.plot(truth[:, 0], truth[:, 1], label="truth")
    ax_xy.scatter(
        truth[0, 0], truth[0, 1],
        marker="o", s=36, label="start", zorder=3,
    )
    ax_xy.scatter(
        truth[-1, 0], truth[-1, 1],
        marker="x", s=48, label="end", zorder=3,
    )
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("True trajectory in x-y plane")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")
    ax_xy.set_aspect("equal", adjustable="datalim")

    # (2) True vertical motion.
    ax_z_truth = fig.add_subplot(gs[1, 0])
    ax_z_truth.plot(t, truth[:, 2], label="truth z")
    ax_z_truth.set_xlabel("time (s)")
    ax_z_truth.set_ylabel("z (m)")
    ax_z_truth.set_title("True vertical trajectory")
    ax_z_truth.grid(True, alpha=0.3)
    ax_z_truth.legend(loc="best")

    # (3-5) Prediction error along x/y/z.
    for k, component in enumerate(("x", "y", "z")):
        ax = fig.add_subplot(gs[k + 2, 0])
        ax.plot(t, dual_err[:, k], label="dual_layer")
        ax.plot(t, central_err[:, k], label="centralized")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylabel(f"{component} error (m)")
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(
        f"Class {label} representative truth and prediction error "
        f"(entity {entity_index})",
        y=0.995,
    )
    fig.subplots_adjust(top=0.955, bottom=0.055, left=0.11, right=0.97)

    path = outdir / f"prediction_error_class_{label}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_metric_plot(outdir, title, filename, t,
                     dual_nees, central_nees, dof,
                     dual_rmse, central_rmse, moving_window):
    dual_nnees = np.asarray(dual_nees, dtype=float) / float(dof)
    central_nnees = np.asarray(central_nees, dtype=float) / float(dof)
    dual_ma = _moving_average(dual_nnees, moving_window)
    central_ma = _moving_average(central_nnees, moving_window)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.8), sharex=True)
    axes[0].plot(t, dual_nnees, alpha=0.25, label="dual raw")
    axes[0].plot(t, central_nnees, alpha=0.25, label="central raw")
    axes[0].plot(t, dual_ma, label="dual moving mean")
    axes[0].plot(t, central_ma, label="central moving mean")
    axes[0].axhline(1.0, linestyle="--", linewidth=1.0, label="ideal = 1")
    axes[0].set_ylabel("normalized NEES")
    axes[0].set_title(f"{title}: normalized NEES")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", ncol=2)

    axes[1].plot(t, dual_rmse, label="dual_layer")
    axes[1].plot(t, central_rmse, label="centralized")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("RMSE (m)")
    axes[1].set_title(f"{title}: position RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    path = outdir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _single_seed_nees_summary(series, dof, burn):
    x = _finite(np.asarray(series, dtype=float)[burn:])
    if x.size == 0:
        return {
            "time_mean_nees": float("nan"),
            "time_mean_nnees": float("nan"),
            "time_median_nees": float("nan"),
            "time_p95_nees": float("nan"),
            "final_nees": float("nan"),
            "num_post_burn_epochs": 0,
        }
    original = np.asarray(series, dtype=float).ravel()
    return {
        "time_mean_nees": float(np.mean(x)),
        "time_mean_nnees": float(np.mean(x) / dof),
        "time_median_nees": float(np.median(x)),
        "time_p95_nees": float(np.quantile(x, 0.95)),
        "final_nees": float(original[-1]),
        "num_post_burn_epochs": int(x.size),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fast one-point bounded-random comparison: centralized vs dual_layer")
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=0.10)
    parser.add_argument("--duration", type=float, default=50.0)
    parser.add_argument("--dt-imu", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument(
        "--traj-type", default="random_accel",
        choices=["random_accel", "agile", "circle_formation", "static",
                 "const_vel", "circle", "figure8"])
    parser.add_argument("--max-accel", type=float, default=4.0)
    parser.add_argument("--max-turn-rate", type=float, default=0.8)

    parser.add_argument("--xy-boundary-margin", type=float, default=20.0)
    parser.add_argument("--xy-boundary-k", type=float, default=0.20)
    parser.add_argument("--xy-boundary-damping", type=float, default=0.80)
    parser.add_argument("--velocity-damping", type=float, default=0.03)
    parser.add_argument(
        "--z-reference", type=float, default=25.0,
        help="centre of the per-entity reference-height band")
    parser.add_argument(
        "--z-reference-spread", type=float, default=30.0,
        help="total vertical spread of per-entity reference heights")
    parser.add_argument(
        "--z-reference-jitter", type=float, default=1.0,
        help="small seed-dependent jitter of each reference height")
    parser.add_argument(
        "--z-initial-spread", type=float, default=2.0,
        help="initial z std around each entity's own reference height")
    parser.add_argument("--z-min", type=float, default=5.0)
    parser.add_argument("--z-max", type=float, default=45.0)
    parser.add_argument(
        "--z-restore-k", type=float, default=0.035,
        help="weak restoring gain toward each entity's own z reference")
    parser.add_argument("--z-velocity-damping", type=float, default=0.25)
    parser.add_argument("--z-boundary-margin", type=float, default=3.0)
    parser.add_argument("--z-boundary-k", type=float, default=0.60)

    parser.add_argument("--burn-in", type=float, default=0.25,
                        help="fraction excluded from time-mean NEES")
    parser.add_argument("--nees-window-s", type=float, default=2.0,
                        help="moving-average window used only for plotting")
    parser.add_argument("--output", default="results_m22_quick_compare_random")
    args = parser.parse_args(argv)

    if args.interval <= 0.0 or args.duration <= 0.0 or args.dt_imu <= 0.0:
        parser.error("interval, duration and dt-imu must be positive")
    if args.speed < 0.0:
        parser.error("speed must be non-negative")
    if not 0.0 <= args.burn_in < 1.0:
        parser.error("burn-in must lie in [0,1)")
    if args.nees_window_s <= 0.0:
        parser.error("nees-window-s must be positive")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    config, epochs = build_config(args)
    methods = ["centralized", "dual_layer"]
    burn = min(int(epochs * args.burn_in), epochs - 1)
    moving_window = max(1, int(round(args.nees_window_s / args.interval)))

    print(
        "[quick compare] "
        f"K=10 classes=(2,6,2), traj={config['traj_type']}, "
        f"speed={config['speed']:.3f}, T={config['sync_interval']:.3f}, "
        f"epochs={epochs}, duration={config['sim_duration']:.3f}s, "
        f"seed={args.seed}, burn={burn}, NEES-window={moving_window} epochs",
        flush=True)

    logs, _ = run_trial(config, args.seed, methods, run_preflight=False)
    class_map = list(config["entity_class_map"])
    reps = pick_representative_entities(class_map)
    t = time_axis(config)
    truth_p = logs["_meta"]["truth_history"]["p"]

    # Vertical-geometry diagnostics: if this span collapses toward zero, the
    # mobile formation is becoming nearly coplanar in z.
    z_span_t = np.ptp(np.asarray(truth_p, dtype=float)[:, :, 2], axis=1)
    z_std_t = np.std(np.asarray(truth_p, dtype=float)[:, :, 2], axis=1)
    print(
        "[vertical geometry] "
        f"z-span min/mean/final="
        f"{np.min(z_span_t):.2f}/{np.mean(z_span_t):.2f}/{z_span_t[-1]:.2f} m; "
        f"final z-std={z_std_t[-1]:.2f} m",
        flush=True,
    )

    dual_p = logs["dual_layer"]["nominal_history"]["p"]
    central_p = logs["centralized"]["nominal_history"]["p"]

    plot_paths = []
    for label, entity_index in reps.items():
        plot_paths.append(str(save_prediction_error_plot(
            outdir, label, entity_index, t, truth_p, dual_p, central_p)))
        dof = 3 * class_map.count(label)
        plot_paths.append(str(save_metric_plot(
            outdir, f"Class {label}", f"metrics_class_{label}.png", t,
            logs["dual_layer"][f"nees_{label}"],
            logs["centralized"][f"nees_{label}"], dof,
            logs["dual_layer"][f"rmse_{label}"],
            logs["centralized"][f"rmse_{label}"], moving_window)))

    plot_paths.append(str(save_metric_plot(
        outdir, "Overall position", "metrics_position_overall.png", t,
        logs["dual_layer"]["nees"], logs["centralized"]["nees"],
        3 * len(class_map), logs["dual_layer"]["rmse"],
        logs["centralized"]["rmse"], moving_window)))

    summary = {
        "warning": (
            "Single-seed quick probe. Formal uncertainty across seeds is "
            "reported by l4_comparison/l4_failure_boundary with bootstrap CIs."),
        "config": {
            "class_sizes": list(config["class_sizes"]),
            "traj_type": config["traj_type"],
            "speed": float(config["speed"]),
            "sync_interval": float(config["sync_interval"]),
            "dt_imu": float(config["dt_imu"]),
            "sim_duration": float(config["sim_duration"]),
            "seed": int(args.seed),
            "burn_in_fraction": float(args.burn_in),
            "burn_in_epoch": int(burn),
            "nees_moving_window_s": float(args.nees_window_s),
            "motion_bounds": {
                key: _serialisable(config[key]) for key in (
                    "xy_boundary_margin", "xy_boundary_k",
                    "xy_boundary_damping", "random_velocity_damping",
                    "z_reference", "z_reference_spread",
                    "z_reference_jitter", "z_initial_spread",
                    "z_min", "z_max", "z_restore_k",
                    "z_velocity_damping", "z_boundary_margin", "z_boundary_k")
            },
        },
        "representative_entities": reps,
        "vertical_geometry": {
            "z_span_min_m": float(np.min(z_span_t)),
            "z_span_mean_m": float(np.mean(z_span_t)),
            "z_span_final_m": float(z_span_t[-1]),
            "z_std_final_m": float(z_std_t[-1]),
            "initial_z_by_entity_m": np.asarray(truth_p, dtype=float)[0, :, 2],
            "final_z_by_entity_m": np.asarray(truth_p, dtype=float)[-1, :, 2],
        },
        "metrics": {},
        "plots": plot_paths,
    }

    dofs = {"overall": 3 * len(class_map)}
    dofs.update({label: 3 * class_map.count(label) for label in ("A", "B", "C")})
    for method in methods:
        method_summary = {
            "position": _single_seed_nees_summary(
                logs[method]["nees"], dofs["overall"], burn),
            "classes": {},
            "pos_rmse_time_mean": float(np.mean(logs[method]["rmse"][burn:])),
            "pos_rmse_final": float(logs[method]["rmse"][-1]),
            "rmte_time_mean": float(np.mean(logs[method]["rmte"][burn:])),
        }
        for label in ("A", "B", "C"):
            method_summary["classes"][label] = _single_seed_nees_summary(
                logs[method][f"nees_{label}"], dofs[label], burn)
            method_summary["classes"][label]["pos_rmse_time_mean"] = float(
                np.mean(logs[method][f"rmse_{label}"][burn:]))
        summary["metrics"][method] = method_summary

    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialisable(summary), handle, ensure_ascii=False, indent=2)

    compact = {
        method: {
            "position_time_mean_nnees": summary["metrics"][method]["position"]["time_mean_nnees"],
            "A_time_mean_nnees": summary["metrics"][method]["classes"]["A"]["time_mean_nnees"],
            "B_time_mean_nnees": summary["metrics"][method]["classes"]["B"]["time_mean_nnees"],
            "C_time_mean_nnees": summary["metrics"][method]["classes"]["C"]["time_mean_nnees"],
            "position_rmse_time_mean": summary["metrics"][method]["pos_rmse_time_mean"],
        }
        for method in methods
    }
    print(json.dumps(_serialisable(compact), ensure_ascii=False, indent=2))
    print(f"[quick compare] summary -> {summary_path}")
    for path in plot_paths:
        print(f"[quick compare] plot    -> {path}")


if __name__ == "__main__":
    main()
