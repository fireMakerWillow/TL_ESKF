"""M22 two-dimensional applicability-domain experiment.

M22 deliberately runs two complementary horizons:

``fixed_duration``
    Every grid point is observed for the same wall-clock duration.  This is the
    operational question (what happens over one mission segment), but it mixes
    per-cycle approximation error with the number of UWB cycles ``duration/T``
    and with the travelled angle ``v*duration/R``.

``fixed_angle``
    Every speed covers the same circular angle.  This normalises geometric
    excitation and reveals whether an apparent speed trend in the fixed-time
    table is merely caused by unequal path coverage.  The actual duration is
    rounded UP to an integer number of UWB periods and is stored in every row.

The experiment compares M19 against M15 on shared Monte-Carlo streams.  Accuracy
is pooled from post-burn-in error vectors (no mean-of-roots bias), consistency
uses one final-epoch NEES sample per independent trial, and M12 diagnostics are
read at their causal sample points: pre-UWB eta_dyn/channel shares, innovation
eta_S, and post-UWB structural/position-layer drift.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Iterable

import numpy as np

from config import ENTITY_CLASSES, IDX_P, apply_entity_count, validate_config
from l1_metrics import summarize_nees
from l3_runner import run_trial
from l4_results_io import save_config_snapshot, save_csv_rows, save_json

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - optional plotting dependency
    _HAVE_MPL = False

HORIZON_MODES = ("fixed_duration", "fixed_angle")


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _mean(values: Iterable[float]) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _max(values: Iterable[float]) -> float:
    arr = _finite(values)
    return float(np.max(arr)) if arr.size else float("nan")


def _pooled_rmse(run_rmses: Iterable[float]) -> float:
    arr = _finite(run_rmses)
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else float("nan")


def _rms_position(error_history: np.ndarray, indices=None) -> float:
    E = np.asarray(error_history, dtype=float)
    if E.ndim != 3:
        raise ValueError(f"error history must be (epochs,K,15), got {E.shape}")
    if indices is not None:
        E = E[:, list(indices), :]
    if E.shape[0] == 0 or E.shape[1] == 0:
        return float("nan")
    block = E[:, :, IDX_P]
    return float(np.sqrt(np.mean(np.sum(block ** 2, axis=-1))))


def _burn_index(num_epochs: int, fraction: float) -> int:
    if num_epochs <= 0:
        raise ValueError("M22 requires at least one UWB epoch")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("m22_burn_in_frac must lie in [0,1)")
    return min(int(num_epochs * fraction), num_epochs - 1)


def _series(log: dict, key: str, burn: int = 0) -> np.ndarray:
    value = np.asarray(log.get(key, []), dtype=float).ravel()
    return value[burn:]


def _horizon_config(config, v, T, horizon_mode, fixed_duration_s=None,
                    target_angle_rad=None):
    if horizon_mode not in HORIZON_MODES:
        raise ValueError(f"unknown M22 horizon mode {horizon_mode!r}")
    cfg = copy.deepcopy(config)
    cfg["speed"] = float(v)
    # In random/agile motion, the M22 v-axis is a sustained full 3-D target
    # speed magnitude, not merely the initial speed.
    if cfg.get("traj_type") in ("random_accel", "agile"):
        cfg["target_speed"] = float(v)
    cfg["sync_interval"] = float(T)
    cfg["uwb_only_pos_rw_psd"] = float(v) ** 2 * float(T)

    R = float(cfg["circle_radius"])
    if horizon_mode == "fixed_duration":
        requested_duration = float(
            cfg.get("m22_fixed_duration_s", cfg["sim_duration"])
            if fixed_duration_s is None else fixed_duration_s)
        if requested_duration <= 0.0:
            raise ValueError("fixed duration must be positive")
        epochs = max(1, int(math.ceil(requested_duration / float(T) - 1e-12)))
        target_angle = float(v) * requested_duration / R
    else:
        target_angle = float(
            cfg.get("m22_target_angle_rad", 0.5 * math.pi)
            if target_angle_rad is None else target_angle_rad)
        if target_angle <= 0.0:
            raise ValueError("target angular travel must be positive")
        requested_duration = target_angle * R / float(v)
        # Round UP: every point covers at least the requested angle.
        epochs = max(1, int(math.ceil(requested_duration / float(T) - 1e-12)))

    actual_duration = epochs * float(T)
    actual_angle = float(v) * actual_duration / R
    cfg["sim_duration"] = actual_duration
    cfg["m22_horizon_mode"] = horizon_mode
    cfg["m22_requested_duration_s"] = requested_duration
    cfg["m22_actual_duration_s"] = actual_duration
    cfg["m22_target_angle_rad"] = target_angle
    cfg["m22_actual_angle_rad"] = actual_angle
    validate_config(cfg)
    meta = {
        "horizon_mode": horizon_mode,
        "requested_duration_s": requested_duration,
        "actual_duration_s": actual_duration,
        "target_angle_rad": target_angle,
        "target_angle_deg": math.degrees(target_angle),
        "actual_angle_rad": actual_angle,
        "actual_angle_deg": math.degrees(actual_angle),
        "uwb_epochs": epochs,
        "imu_steps": epochs * int(cfg["imu_per_uwb"]),
        "target_speed_mps": (
            float(cfg["target_speed"])
            if cfg.get("target_speed", None) is not None
            else float(cfg["speed"])
        ),
    }
    return cfg, meta


def _blank_samples():
    methods = ("centralized", "dual_layer")
    return {
        "rmse": {name: [] for name in methods},
        "rmte": {name: [] for name in methods},
        "nees": {name: [] for name in methods},
        "class_rmse": {
            name: {label: [] for label in ENTITY_CLASSES} for name in methods},
        "class_rmte": {
            name: {label: [] for label in ENTITY_CLASSES} for name in methods},
        "class_nees": {
            name: {label: [] for label in ENTITY_CLASSES} for name in methods},
        "eta_dyn_mean": [], "eta_dyn_max": [], "eta_dyn_sum": [],
        "eta_S_mean": [], "eta_S_max": [], "eta_S_sum": [],
        "eta_struct_mean": [], "eta_abs_mean": [],
        "pos_layer_dev_rel_mean": [], "pos_layer_dev_rel_final": [],
        "kappa_mean": [], "kappa_plan_curve": [], "kappa_full_curve": [],
        "share_pos": [], "share_vel": [], "share_att": [],
        "share_acc_bias": [], "share_gyro_bias": [],
    }


def _append_trial(samples, per_run_rows, logs, config, seed, burn, horizon):
    class_map = list(config["entity_class_map"])
    class_indices = {
        label: [i for i, cls in enumerate(class_map) if cls == label]
        for label in ENTITY_CLASSES
    }
    run_row = {"seed": int(seed), **horizon}

    for name in ("centralized", "dual_layer"):
        log = logs[name]
        E = np.asarray(log["error_history"][burn:], dtype=float)
        value = _rms_position(E)
        samples["rmse"][name].append(value)
        run_row[f"{name}_rmse"] = value

        value = _mean(_series(log, "rmte", burn))
        samples["rmte"][name].append(value)
        run_row[f"{name}_rmte"] = value

        value = float(np.asarray(log["nees"], dtype=float)[-1])
        samples["nees"][name].append(value)
        run_row[f"{name}_nees_final"] = value

        for label in ENTITY_CLASSES:
            value = _rms_position(E, class_indices[label])
            samples["class_rmse"][name][label].append(value)
            run_row[f"{name}_rmse_{label}"] = value

            value = _mean(_series(log, f"rmte_{label}", burn))
            samples["class_rmte"][name][label].append(value)
            run_row[f"{name}_rmte_{label}"] = value

            value = float(np.asarray(log[f"nees_{label}"], dtype=float)[-1])
            samples["class_nees"][name][label].append(value)
            run_row[f"{name}_nees_{label}_final"] = value

    dual = logs["dual_layer"]
    diagnostic_keys = {
        "eta_dyn_mean": ("diag_pre_eta_dyn_mean", "mean"),
        "eta_dyn_max": ("diag_pre_eta_dyn_max", "max"),
        "eta_dyn_sum": ("diag_pre_eta_dyn_mean", "sum"),
        "eta_S_mean": ("diag_innovation_eta_S_mean", "mean"),
        "eta_S_max": ("diag_innovation_eta_S_max", "max"),
        "eta_S_sum": ("diag_innovation_eta_S_mean", "sum"),
        "eta_struct_mean": ("diag_post_eta_struct_mean", "mean"),
        "eta_abs_mean": ("diag_post_eta_abs_mean", "mean"),
        "pos_layer_dev_rel_mean": ("diag_post_pos_layer_dev_rel_mean", "mean"),
        "pos_layer_dev_rel_final": ("diag_post_pos_layer_dev_rel_mean", "final"),
        "kappa_mean": ("diag_pre_kappa_mean", "mean"),
        "kappa_plan_curve": ("diag_pre_kappa_plan_curve", "mean"),
        "kappa_full_curve": ("diag_pre_kappa_full_curve", "mean"),
        "share_pos": ("diag_pre_share_pos", "mean"),
        "share_vel": ("diag_pre_share_vel", "mean"),
        "share_att": ("diag_pre_share_att", "mean"),
        "share_acc_bias": ("diag_pre_share_acc_bias", "mean"),
        "share_gyro_bias": ("diag_pre_share_gyro_bias", "mean"),
    }
    for out_key, (source, reduction) in diagnostic_keys.items():
        arr = _finite(_series(dual, source, burn))
        if arr.size == 0:
            value = float("nan")
        elif reduction == "mean":
            value = float(np.mean(arr))
        elif reduction == "max":
            value = float(np.max(arr))
        elif reduction == "sum":
            value = float(np.sum(arr))
        elif reduction == "final":
            value = float(arr[-1])
        else:  # pragma: no cover
            raise RuntimeError(reduction)
        samples[out_key].append(value)
        run_row[out_key] = value

    angle = max(float(horizon["actual_angle_rad"]), 1e-15)
    run_row["eta_dyn_sum_per_rad"] = run_row["eta_dyn_sum"] / angle
    run_row["eta_S_sum_per_rad"] = run_row["eta_S_sum"] / angle
    per_run_rows.append(run_row)


def _aggregate(samples, config, horizon):
    K = int(config["num_entities"])
    confidence = float(config["nees_confidence"])
    dof = int(config["pos_dim"] * K)
    result = dict(horizon)

    nees_summary = {}
    for name in ("centralized", "dual_layer"):
        result[f"{name}_rmse"] = _pooled_rmse(samples["rmse"][name])
        result[f"{name}_rmte"] = _mean(samples["rmte"][name])
        result[f"{name}_rmse_rmte_ratio"] = (
            result[f"{name}_rmse"] / result[f"{name}_rmte"]
            if result[f"{name}_rmte"] > 0.0 else float("nan"))
        summary = summarize_nees(samples["nees"][name], dof, confidence)
        nees_summary[name] = summary
        result[f"{name}_anees"] = summary["anees"]
        result[f"{name}_anees_lower"] = summary["anees_bounds"][0]
        result[f"{name}_anees_upper"] = summary["anees_bounds"][1]
        result[f"{name}_anees_consistent"] = summary["anees_consistent"]
        result[f"{name}_anees_verdict"] = summary["verdict"]

        for label in ENTITY_CLASSES:
            n = list(config["entity_class_map"]).count(label)
            result[f"{name}_rmse_{label}"] = _pooled_rmse(
                samples["class_rmse"][name][label])
            result[f"{name}_rmte_{label}"] = _mean(
                samples["class_rmte"][name][label])
            class_summary = summarize_nees(
                samples["class_nees"][name][label],
                int(config["pos_dim"] * n), confidence)
            result[f"{name}_anees_{label}"] = class_summary["anees"]
            result[f"{name}_anees_{label}_lower"] = class_summary["anees_bounds"][0]
            result[f"{name}_anees_{label}_upper"] = class_summary["anees_bounds"][1]
            result[f"{name}_anees_{label}_consistent"] = (
                class_summary["anees_consistent"])

    c = max(result["centralized_rmse"], 1e-15)
    result["rmse_rel_gap"] = (result["dual_layer_rmse"] - c) / c
    cc = max(result["centralized_rmse_C"], 1e-15)
    result["rmse_C_rel_gap"] = (result["dual_layer_rmse_C"] - cc) / cc
    cr = max(result["centralized_rmte"], 1e-15)
    result["rmte_rel_gap"] = (result["dual_layer_rmte"] - cr) / cr

    dual_summary = nees_summary["dual_layer"]
    lo, hi = dual_summary["anees_bounds"]
    a = dual_summary["anees"]
    if not np.isfinite(a) or not np.isfinite(lo) or not np.isfinite(hi):
        result["nees_band_excess"] = float("nan")
    elif a > hi:
        result["nees_band_excess"] = (a - hi) / max(hi, 1e-15)
    elif a < lo:
        result["nees_band_excess"] = (lo - a) / max(lo, 1e-15)
    else:
        result["nees_band_excess"] = 0.0

    for key in (
        "eta_dyn_mean", "eta_dyn_max", "eta_dyn_sum",
        "eta_S_mean", "eta_S_max", "eta_S_sum",
        "eta_struct_mean", "eta_abs_mean", "pos_layer_dev_rel_mean",
        "pos_layer_dev_rel_final", "kappa_mean", "kappa_plan_curve",
        "kappa_full_curve", "share_pos", "share_vel", "share_att",
        "share_acc_bias", "share_gyro_bias"):
        result[key] = _mean(samples[key])

    angle = max(float(horizon["actual_angle_rad"]), 1e-15)
    duration = max(float(horizon["actual_duration_s"]), 1e-15)
    epochs = max(int(horizon["uwb_epochs"]), 1)
    result["eta_dyn_sum_per_rad"] = result["eta_dyn_sum"] / angle
    result["eta_S_sum_per_rad"] = result["eta_S_sum"] / angle
    result["eta_dyn_sum_per_second"] = result["eta_dyn_sum"] / duration
    result["eta_S_sum_per_second"] = result["eta_S_sum"] / duration
    result["eta_dyn_per_cycle"] = result["eta_dyn_sum"] / epochs
    result["eta_S_per_cycle"] = result["eta_S_sum"] / epochs
    return result


def _zone(row, config):
    """Classify a grid point using absolute consistency, not C-relative ANEES."""
    rmse_tol = float(config.get("zone_rmse_tol", 0.05))
    c_tol = float(config.get("zone_class_c_rmse_tol", 2.0 * rmse_tol))
    nees_tol = float(config.get("zone_nees_tol", 0.25))
    eta_tol = float(config.get("zone_eta_tol", 0.10))
    eta_s_tol = float(config.get("zone_eta_s_tol", 0.25))
    calibration_tol = float(config.get("zone_calibration_tol", 0.25))

    ratio = float(row["dual_layer_rmse_rmte_ratio"])
    calibration_error = abs(ratio - 1.0) if np.isfinite(ratio) else float("inf")
    values = {
        "rmse": float(row["rmse_rel_gap"]),
        "class_c": float(row["rmse_C_rel_gap"]),
        "nees": float(row["nees_band_excess"]),
        "eta_dyn": float(row["eta_dyn_mean"]),
        "eta_S": float(row["eta_S_mean"]),
        "calibration": calibration_error,
    }
    limits = {
        "rmse": rmse_tol, "class_c": c_tol, "nees": nees_tol,
        "eta_dyn": eta_tol, "eta_S": eta_s_tol,
        "calibration": calibration_tol,
    }
    row["calibration_error"] = calibration_error
    row["zone_failed_checks"] = [
        key for key in values
        if not np.isfinite(values[key]) or values[key] > limits[key]
    ]
    if not row["zone_failed_checks"]:
        return "recommended"
    if all(np.isfinite(values[key]) and values[key] <= 2.0 * limits[key]
           for key in values):
        return "transition"
    return "not_recommended"


def _plot_heatmaps(rows, v_values, T_values, config, out_dir, prefix):
    if not (_HAVE_MPL and bool(config.get("save_plots", True))):
        return []
    os.makedirs(out_dir, exist_ok=True)
    metrics = (
        ("rmse_rel_gap", "Relative RMSE gap"),
        ("rmse_C_rel_gap", "Class C relative RMSE gap"),
        ("nees_band_excess", "Dual-layer ANEES band excess"),
        ("eta_dyn_mean", "Mean dynamic residual"),
        ("eta_S_mean", "Mean innovation-variance residual"),
        ("pos_layer_dev_rel_final", "Final position-layer relative drift"),
    )
    index = {(float(r["speed"]), float(r["sync_interval"])): r for r in rows}
    paths = []
    for key, title in metrics:
        matrix = np.full((len(v_values), len(T_values)), np.nan)
        for iv, v in enumerate(v_values):
            for it, T in enumerate(T_values):
                matrix[iv, it] = float(index[(v, T)].get(key, np.nan))
        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        image = axis.imshow(matrix, aspect="auto", origin="lower")
        axis.set_xticks(np.arange(len(T_values)), [f"{x:g}" for x in T_values])
        axis.set_yticks(np.arange(len(v_values)), [f"{x:g}" for x in v_values])
        axis.set_xlabel("Synchronization interval T (s)")
        axis.set_ylabel("Speed v (m/s)")
        axis.set_title(title)
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        path = os.path.join(
            out_dir, f"{prefix}_{key}.{config.get('plot_format', 'png')}")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def run_stage(config, horizon_mode="fixed_duration", fixed_duration_s=None,
              target_angle_rad=None, output_dir=None, output_prefix=None,
              verbose=True):
    """Run one M22 horizon and return a machine-readable payload."""
    validate_config(config)
    v_values = [float(v) for v in config["sweep_v_values"]]
    T_values = [float(T) for T in config["sweep_T_values"]]
    burn_fraction = float(config.get("m22_burn_in_frac", 0.5))
    rows, per_run_rows = [], []

    stage_out = output_dir or str(config["output_dir"])
    prefix = output_prefix or (
        "applicability_fixed_duration" if horizon_mode == "fixed_duration"
        else "applicability_fixed_angle")

    for v in v_values:
        for T in T_values:
            cfg, horizon = _horizon_config(
                config, v, T, horizon_mode,
                fixed_duration_s=fixed_duration_s,
                target_angle_rad=target_angle_rad)
            samples = _blank_samples()
            for run_index in range(int(cfg["mc_runs"])):
                seed = int(cfg["seed_base"]) + run_index
                logs, _ = run_trial(
                    cfg, seed, ["centralized", "dual_layer"])
                num_epochs = len(logs["centralized"]["rmse"])
                burn = _burn_index(num_epochs, burn_fraction)
                _append_trial(samples, per_run_rows, logs, cfg, seed, burn,
                              {"speed": v, "sync_interval": T, **horizon})

            row = {
                "speed": v, "sync_interval": T,
                "mc_runs": int(cfg["mc_runs"]),
                "burn_in_fraction": burn_fraction,
                **_aggregate(samples, cfg, horizon),
            }
            row["zone"] = _zone(row, cfg)
            rows.append(row)
            if verbose:
                print(
                    "  [%s] v=%5.2f T=%5.3f angle=%6.1fdeg epochs=%5d "
                    "dRMSE=%+.3f Cgap=%+.3f ANEES=%7.2f eta/cycle=%.3e "
                    "etaS=%.3e %s" % (
                        horizon_mode, v, T, row["actual_angle_deg"],
                        row["uwb_epochs"], row["rmse_rel_gap"],
                        row["rmse_C_rel_gap"], row["dual_layer_anees"],
                        row["eta_dyn_per_cycle"], row["eta_S_mean"],
                        row["zone"]), flush=True)

    payload = {
        "horizon_mode": horizon_mode,
        "v_values": v_values,
        "T_values": T_values,
        "mc_runs": int(config["mc_runs"]),
        "burn_in_fraction": burn_fraction,
        "rows": rows,
    }
    os.makedirs(stage_out, exist_ok=True)
    save_json(payload, os.path.join(stage_out, f"{prefix}.json"))
    save_csv_rows(rows, os.path.join(stage_out, f"{prefix}.csv"))
    save_csv_rows(per_run_rows, os.path.join(stage_out, f"{prefix}_per_run.csv"))
    save_config_snapshot(
        config, stage_out, f"{prefix}_config_snapshot.json")
    payload["plots"] = _plot_heatmaps(
        rows, v_values, T_values, config, stage_out, prefix)
    return payload


def run_two_stage(config, fixed_duration_s=None, target_angle_rad=None,
                  verbose=True):
    """Automatically run fixed-duration then fixed-angle M22 scans."""
    validate_config(config)
    root = str(config["output_dir"])
    os.makedirs(root, exist_ok=True)
    duration = float(config.get("m22_fixed_duration_s", config["sim_duration"])
                     if fixed_duration_s is None else fixed_duration_s)
    angle = float(config.get("m22_target_angle_rad", 0.5 * math.pi)
                  if target_angle_rad is None else target_angle_rad)

    if verbose:
        print("[M22] Stage 1/2: fixed-duration scan", flush=True)
    fixed_duration = run_stage(
        config, "fixed_duration", fixed_duration_s=duration,
        output_dir=os.path.join(root, "fixed_duration"),
        output_prefix="applicability_fixed_duration", verbose=verbose)

    if verbose:
        print("[M22] Stage 2/2: fixed-angle scan", flush=True)
    fixed_angle = run_stage(
        config, "fixed_angle", target_angle_rad=angle,
        output_dir=os.path.join(root, "fixed_angle"),
        output_prefix="applicability_fixed_angle", verbose=verbose)

    payload = {
        "fixed_duration_s": duration,
        "target_angle_rad": angle,
        "target_angle_deg": math.degrees(angle),
        "fixed_duration": fixed_duration,
        "fixed_angle": fixed_angle,
    }
    save_json(payload, os.path.join(root, "applicability_two_stage.json"))
    save_config_snapshot(config, root, "applicability_two_stage_config_snapshot.json")
    return payload


def run(config, verbose=True):
    """Backward-compatible one-stage entry used by ``main.py failure``."""
    mode = str(config.get("m22_horizon_mode", "fixed_duration"))
    return run_stage(config, mode, verbose=verbose,
                     output_prefix="applicability_domain")


if __name__ == "__main__":
    from config import default_config

    cfg = default_config()
    apply_entity_count(cfg, 3, (1, 1, 1))
    cfg["mc_runs"] = 1
    cfg["sim_duration"] = 0.1
    cfg["m22_fixed_duration_s"] = 0.1
    cfg["m22_target_angle_rad"] = 0.01
    cfg["sweep_v_values"] = [2.0]
    cfg["sweep_T_values"] = [0.05]
    cfg["save_plots"] = False
    cfg["run_preflight"] = False
    cfg["separation_check"] = False
    result = run_stage(cfg, "fixed_duration", verbose=False)
    assert len(result["rows"]) == 1
    assert result["rows"][0]["zone"] in {
        "recommended", "transition", "not_recommended"}
    result2 = run_stage(cfg, "fixed_angle", verbose=False)
    assert result2["rows"][0]["actual_angle_rad"] >= 0.01
    print("[M22 ApplicabilityDomainExperiment] OK (two horizon modes)")
