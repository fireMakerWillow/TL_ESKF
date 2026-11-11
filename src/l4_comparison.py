"""M21 performance-resource Monte-Carlo comparison (RQ1).

The experiment has two deliberately separate method groups:

* covariance architectures: Centralized, Dual-layer, DCL (registry key
  ``local_only``), CI-trace, CI-det;
* sensor ablations: IMU-only and UWB-only.

Every Monte-Carlo trial runs all requested methods on one shared truth,
measurement and initial-error realisation supplied by M20.  Accuracy is pooled
from the post-burn-in error vectors.  Consistency is summarized by first taking
a post-burn-in time mean inside each seed and normalizing by the relevant
degrees of freedom.  The resulting seed-level TMNNEES values are then summarized
across independent seeds.  Final-epoch ANEES is retained only as an audit field.

The default suite runs the K=8 main scenario and the K=10 extension.  A single
scenario can be executed with :func:`run_scenario` for tests and exploratory
work.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np

from config import (
    ENTITY_CLASSES,
    IDX_BA,
    IDX_BG,
    IDX_P,
    IDX_TH,
    IDX_V,
    MAIN_LINE_ESTIMATORS,
    SENSOR_ABLATION_ESTIMATORS,
    apply_entity_count,
    validate_config,
)
from l1_metrics import summarize_nees
from l1_resource import storage_floats
from l2_estimators import evaluated_channels
from l3_runner import run_trial
from l4_results_io import (
    plot_comparison,
    save_config_snapshot,
    save_csv_rows,
    save_json,
)

_CHANNEL_SLICES = {
    "pos": IDX_P,
    "vel": IDX_V,
    "att": IDX_TH,
    "acc_bias": IDX_BA,
    "gyro_bias": IDX_BG,
}
_CHANNEL_RESULT_KEYS = {
    "pos": "pos_rmse",
    "vel": "vel_rmse",
    "att": "att_rmse_deg",
    "acc_bias": "acc_bias_rmse",
    "gyro_bias": "gyro_bias_rmse",
}


def _finite(values):
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _mean_std(values):
    arr = _finite(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))



def _sample_std(values):
    arr = _finite(values)
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1))


def _bootstrap_mean_ci(values, confidence=0.95, resamples=5000, seed=24680):
    """Percentile-bootstrap interval for a mean of independent seed values."""
    arr = _finite(values)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1 or int(resamples) <= 0:
        value = float(np.mean(arr))
        return (value, value)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, arr.size, size=(int(resamples), arr.size))
    means = np.mean(arr[indices], axis=1)
    alpha = 1.0 - float(confidence)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def _scalar_summary(values, config, bootstrap_offset=0):
    arr = _finite(values)
    confidence = float(config.get("nees_bootstrap_confidence", 0.95))
    resamples = int(config.get("nees_bootstrap_resamples", 5000))
    seed = int(config.get("nees_bootstrap_seed", 24680)) + int(
        bootstrap_offset)
    return {
        "values": arr.tolist(),
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": _sample_std(arr),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
        "bootstrap_ci": _bootstrap_mean_ci(
            arr, confidence=confidence, resamples=resamples, seed=seed),
        "bootstrap_confidence": confidence,
    }


def _time_mean_nees_summary(values, dof, config, bootstrap_offset=0):
    """Summarize one post-burn-in time-mean NEES value per independent seed."""
    raw = _finite(values)
    normalized = raw / float(dof) if raw.size else raw
    stats = _scalar_summary(normalized, config, bootstrap_offset)
    lo, hi = stats["bootstrap_ci"]
    if stats["count"] < 2:
        verdict = "insufficient_seeds"
    elif np.isfinite(lo) and lo > 1.0:
        verdict = "overconfident"
    elif np.isfinite(hi) and hi < 1.0:
        verdict = "conservative"
    else:
        verdict = "compatible_with_1"

    return {
        "seed_time_mean_nees": raw.tolist(),
        "seed_time_mean_nnees": normalized.tolist(),
        "time_mean_nees": (
            float(stats["mean"] * dof) if np.isfinite(stats["mean"])
            else float("nan")
        ),
        "time_mean_nees_std": (
            float(stats["std"] * dof) if np.isfinite(stats["std"])
            else float("nan")
        ),
        "time_mean_nees_bootstrap_ci": (
            float(lo * dof) if np.isfinite(lo) else float("nan"),
            float(hi * dof) if np.isfinite(hi) else float("nan"),
        ),
        "time_mean_nnees": stats["mean"],
        "time_mean_nnees_std": stats["std"],
        "time_mean_nnees_median": stats["median"],
        "time_mean_nnees_min": stats["min"],
        "time_mean_nnees_max": stats["max"],
        "time_mean_nnees_bootstrap_ci": stats["bootstrap_ci"],
        "time_mean_nnees_num_seeds": stats["count"],
        "time_mean_nnees_verdict": verdict,
        "time_mean_nnees_compatible_with_1": verdict == "compatible_with_1",
        "dof": int(dof),
    }


def _post_burn_time_mean(series, burn):
    values = _finite(np.asarray(series, dtype=float).ravel()[int(burn):])
    return float(np.mean(values)) if values.size else float("nan")


def _motion_diagnostic(logs, seed):
    truth = logs.get("_meta", {}).get("truth_history", {})
    p = np.asarray(truth.get("p", []), dtype=float)
    v = np.asarray(truth.get("v", []), dtype=float)
    out = {"seed": int(seed)}
    if p.ndim == 3 and p.shape[-1] >= 3:
        span = np.ptp(p[:, :, 2], axis=1)
        out.update({
            "z_span_min_m": float(np.min(span)),
            "z_span_mean_m": float(np.mean(span)),
            "z_span_final_m": float(span[-1]),
            "z_std_final_m": float(np.std(p[-1, :, 2])),
        })
    if v.ndim == 3 and v.shape[-1] >= 3:
        speed = np.linalg.norm(v[:, :, :3], axis=-1)
        out.update({
            "speed_3d_mean_mps": float(np.mean(speed)),
            "speed_3d_std_mps": float(np.std(speed)),
            "speed_3d_final_mean_mps": float(np.mean(speed[-1])),
        })
    topology = logs.get("_meta", {}).get("topology_history", [])
    if topology:
        for key in ("inter_edges", "anchor_edges", "total_edges",
                    "min_degree", "mean_degree", "max_degree"):
            values = np.asarray([row[key] for row in topology], dtype=float)
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_min"] = float(np.min(values))
            out[f"{key}_max"] = float(np.max(values))
    return out


def _pooled_rmse(run_rmses):
    """Pool equal-size trial RMS values without the mean-of-roots bias."""
    arr = _finite(run_rmses)
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else float("nan")


def _rms_error(E, sl, indices=None, degrees=False):
    E = np.asarray(E, dtype=float)
    if E.ndim != 3:
        raise ValueError(f"error history must be (epochs,K,15), got {E.shape}")
    if indices is not None:
        E = E[:, list(indices), :]
    if E.shape[0] == 0 or E.shape[1] == 0:
        return float("nan")
    block = E[:, :, sl]
    value = float(np.sqrt(np.mean(np.sum(block ** 2, axis=-1))))
    return float(np.degrees(value)) if degrees else value


def _method_groups(config):
    architecture = list(config.get("estimators", MAIN_LINE_ESTIMATORS))
    ablations = list(config.get("ablation_estimators", SENSOR_ABLATION_ESTIMATORS))
    names = architecture + ablations
    if not architecture or len(names) != len(set(names)):
        raise ValueError("M21 method lists must be non-empty and disjoint")
    if config.get("comparison_require_full_set", True):
        if tuple(architecture) != tuple(MAIN_LINE_ESTIMATORS):
            raise ValueError(
                "M21 main table must contain exactly "
                f"{list(MAIN_LINE_ESTIMATORS)}, got {architecture}")
        if tuple(ablations) != tuple(SENSOR_ABLATION_ESTIMATORS):
            raise ValueError(
                "M21 sensor-ablation table must contain exactly "
                f"{list(SENSOR_ABLATION_ESTIMATORS)}, got {ablations}")
    return architecture, ablations, names


def _scenario_specs(config):
    counts = list(config.get("comparison_entity_counts", [config["num_entities"]]))
    if not config.get("comparison_run_extension", True):
        counts = [int(config["num_entities"])]
    if not counts:
        raise ValueError("comparison_entity_counts must be non-empty")
    class_table = config.get("comparison_class_sizes", {})
    specs = []
    for raw_k in counts:
        K = int(raw_k)
        sizes = class_table.get(K, class_table.get(str(K)))
        if sizes is None:
            sizes = (2, K - 4, 2) if K >= 4 else None
        if sizes is None or len(sizes) != 3 or sum(map(int, sizes)) != K:
            raise ValueError(f"invalid M21 class split for K={K}: {sizes}")
        specs.append((K, tuple(map(int, sizes))))
    if len({k for k, _ in specs}) != len(specs):
        raise ValueError("comparison_entity_counts contains duplicates")
    return specs


def _burn_index(config, burn_in_frac):
    steps = int(round(config["sim_duration"] / config["dt_uwb"]))
    if steps <= 0:
        raise ValueError("M21 requires at least one UWB epoch")
    return min(int(steps * burn_in_frac), steps - 1)


def _blank_samples(names):
    out = {}
    for name in names:
        out[name] = {
            "channels": {key: [] for key in _CHANNEL_SLICES},
            "class_rmse": {label: [] for label in ENTITY_CLASSES},
            "rmte": [],
            "class_rmte": {label: [] for label in ENTITY_CLASSES},

            # One post-burn-in time-mean value per independent seed.
            "nees": [],
            "class_nees": {label: [] for label in ENTITY_CLASSES},
            "full_nees": [],
            "att_nees": [],

            # Final-epoch values are retained only for audit/backward checks.
            "nees_final": [],
            "class_nees_final": {label: [] for label in ENTITY_CLASSES},
            "full_nees_final": [],
            "att_nees_final": [],

            "comm_measurement_bytes": [],
            "comm_consumed_measurement_bytes": [],
            "comm_estimation_bytes": [],
            "comm_total_bytes": [],
            "flops": [],
            "runtime_total_s": [],
            "runtime_predict_imu_s": [],
            "runtime_update_period_s": [],
            "memory_array_bytes": [],
        }
    return out

def _append_trial(samples, per_run_rows, logs, names, config, burn, seed):
    class_map = list(config["entity_class_map"])
    class_indices = {
        label: [i for i, cls in enumerate(class_map) if cls == label]
        for label in ENTITY_CLASSES
    }
    dof_pos = int(config["pos_dim"] * config["num_entities"])
    dof_full = int(config["state_dim"] * config["num_entities"])

    # C_measurement is the raw sensor-network stream, not the subset consumed
    # by a particular ablation.
    network_measurement_bytes = max(
        float(logs[n]["resource"]["C_measurement_bytes"]) for n in names)

    for name in names:
        log = logs[name]
        E = np.asarray(log["error_history"][burn:], dtype=float)
        active = set(evaluated_channels(name))
        row = {"seed": int(seed), "method": name, "K": int(config["num_entities"])}

        for channel, sl in _CHANNEL_SLICES.items():
            value = (_rms_error(E, sl, degrees=(channel == "att"))
                     if channel in active else float("nan"))
            samples[name]["channels"][channel].append(value)
            row[_CHANNEL_RESULT_KEYS[channel]] = value

        for label in ENTITY_CLASSES:
            value = _rms_error(E, IDX_P, class_indices[label])
            samples[name]["class_rmse"][label].append(value)
            row[f"pos_rmse_{label}"] = value

        rmte_value = float(np.nanmean(log["rmte"][burn:]))
        samples[name]["rmte"].append(rmte_value)
        row["rmte"] = rmte_value
        for label in ENTITY_CLASSES:
            value = float(np.nanmean(log[f"rmte_{label}"][burn:]))
            samples[name]["class_rmte"][label].append(value)
            row[f"rmte_{label}"] = value

        # Main consistency metric: one post-burn time mean per seed.
        nees_tm = _post_burn_time_mean(log["nees"], burn)
        full_tm = _post_burn_time_mean(log["nees_full"], burn)
        att_tm = _post_burn_time_mean(log["nees_att"], burn)
        samples[name]["nees"].append(nees_tm)
        samples[name]["full_nees"].append(full_tm)
        samples[name]["att_nees"].append(att_tm)

        row["nees_pos_time_mean"] = nees_tm
        row["nnees_pos_time_mean"] = nees_tm / float(dof_pos)
        row["nees_full_time_mean"] = full_tm
        row["nnees_full_time_mean"] = full_tm / float(dof_full)
        row["nees_att_time_mean"] = att_tm
        row["nnees_att_time_mean"] = att_tm / float(dof_pos)

        # Final values are audit fields, not the primary ranking metric.
        pos_final = float(log["nees"][-1])
        full_final = float(log["nees_full"][-1])
        att_final = float(log["nees_att"][-1])
        samples[name]["nees_final"].append(pos_final)
        samples[name]["full_nees_final"].append(full_final)
        samples[name]["att_nees_final"].append(att_final)
        row["nees_pos_final"] = pos_final
        row["nees_full_final"] = full_final
        row["nees_att_final"] = att_final

        for label in ENTITY_CLASSES:
            dof_class = int(config["pos_dim"] * len(class_indices[label]))
            value_tm = _post_burn_time_mean(log[f"nees_{label}"], burn)
            samples[name]["class_nees"][label].append(value_tm)
            row[f"nees_pos_{label}_time_mean"] = value_tm
            row[f"nnees_pos_{label}_time_mean"] = (
                value_tm / float(dof_class)
            )

            value_final = float(log[f"nees_{label}"][-1])
            samples[name]["class_nees_final"][label].append(value_final)
            row[f"nees_pos_{label}_final"] = value_final

        resource = log["resource"]
        runtime = log["runtime"]
        resource_map = {
            "comm_measurement_bytes": network_measurement_bytes,
            "comm_consumed_measurement_bytes": resource["C_measurement_bytes"],
            "comm_estimation_bytes": resource["C_estimation_bytes"],
            "comm_total_bytes": (
                network_measurement_bytes + resource["C_estimation_bytes"]
            ),
            "flops": resource["flops"],
            "memory_array_bytes": resource["covariance_array_bytes"],
            "runtime_total_s": runtime.get("estimator_total_s", np.nan),
            "runtime_predict_imu_s": runtime.get(
                "predict_imu_step_mean_s", np.nan),
            "runtime_update_period_s": runtime.get(
                "update_period_mean_s", np.nan),
        }
        for key, value in resource_map.items():
            value = float(value)
            samples[name][key].append(value)
            row[key] = value
        per_run_rows.append(row)

def _aggregate_method(name, sample, config):
    K = int(config["num_entities"])
    class_map = list(config["entity_class_map"])
    dof_pos = int(config["pos_dim"] * K)
    dof_full = int(config["state_dim"] * K)
    confidence = float(config["nees_confidence"])
    offset = sum(ord(ch) for ch in name)

    pos_tm = _time_mean_nees_summary(
        sample["nees"], dof_pos, config, offset)
    full_tm = _time_mean_nees_summary(
        sample["full_nees"], dof_full, config, offset + 1000)
    att_tm = _time_mean_nees_summary(
        sample["att_nees"], dof_pos, config, offset + 2000)

    # Traditional fixed-time ANEES is retained only for final-epoch auditing.
    final_pos = summarize_nees(sample["nees_final"], dof_pos, confidence)
    final_full = summarize_nees(
        sample["full_nees_final"], dof_full, confidence)
    final_att = summarize_nees(
        sample["att_nees_final"], dof_pos, confidence)

    result = {
        "role": ("architecture" if name in MAIN_LINE_ESTIMATORS
                 else "sensor_ablation"),
        "pos_rmse": _pooled_rmse(sample["channels"]["pos"]),
        "vel_rmse": _pooled_rmse(sample["channels"]["vel"]),
        "att_rmse_deg": _pooled_rmse(sample["channels"]["att"]),
        "acc_bias_rmse": _pooled_rmse(sample["channels"]["acc_bias"]),
        "gyro_bias_rmse": _pooled_rmse(sample["channels"]["gyro_bias"]),
        "rmte": _mean_std(sample["rmte"])[0],
        "rmte_std": _sample_std(sample["rmte"]),

        # Primary consistency output.
        **pos_tm,
        "full_time_mean_nees": full_tm["time_mean_nees"],
        "full_time_mean_nnees": full_tm["time_mean_nnees"],
        "full_time_mean_nnees_std": full_tm["time_mean_nnees_std"],
        "full_time_mean_nnees_bootstrap_ci": (
            full_tm["time_mean_nnees_bootstrap_ci"]
        ),
        "att_time_mean_nees": att_tm["time_mean_nees"],
        "att_time_mean_nnees": att_tm["time_mean_nnees"],
        "att_time_mean_nnees_std": att_tm["time_mean_nnees_std"],
        "att_time_mean_nnees_bootstrap_ci": (
            att_tm["time_mean_nnees_bootstrap_ci"]
        ),

        # Backward-compatible names now point to time-mean NEES, not final NEES.
        "anees": pos_tm["time_mean_nees"],
        "anees_bounds": pos_tm["time_mean_nees_bootstrap_ci"],
        "anees_consistent": pos_tm["time_mean_nnees_compatible_with_1"],
        "anees_verdict": pos_tm["time_mean_nnees_verdict"],
        "anees_num_trials": pos_tm["time_mean_nnees_num_seeds"],
        "in_bound_fraction": float("nan"),

        # Audit-only final-epoch ANEES.
        "final_anees": final_pos["anees"],
        "final_anees_bounds": final_pos["anees_bounds"],
        "final_anees_consistent": final_pos["anees_consistent"],
        "final_anees_verdict": final_pos["verdict"],
        "final_in_bound_fraction": final_pos["in_bound_fraction"],
        "final_full_anees": final_full["anees"],
        "final_full_anees_bounds": final_full["anees_bounds"],
        "final_att_anees": final_att["anees"],
        "final_att_anees_bounds": final_att["anees_bounds"],

        "dof": dof_pos,
        "full_dof": dof_full,
        "class_metrics": {},
        "storage_floats": int(storage_floats(name, K)),
        "storage_analytic_bytes": int(
            storage_floats(name, K) * int(config["float_bytes"])),
    }
    result["rmse"] = result["pos_rmse"]
    result["rmse_rmte_ratio"] = (
        result["pos_rmse"] / result["rmte"]
        if np.isfinite(result["rmte"]) and result["rmte"] > 0
        else float("nan")
    )

    for class_index, label in enumerate(ENTITY_CLASSES):
        n = class_map.count(label)
        dof_class = int(config["pos_dim"] * n)
        class_tm = _time_mean_nees_summary(
            sample["class_nees"][label], dof_class, config,
            offset + 3000 + class_index)
        class_final = summarize_nees(
            sample["class_nees_final"][label], dof_class, confidence)
        class_result = {
            "entities": n,
            "pos_rmse": _pooled_rmse(sample["class_rmse"][label]),
            "rmte": _mean_std(sample["class_rmte"][label])[0],
            **class_tm,
            "anees": class_tm["time_mean_nees"],
            "anees_bounds": class_tm["time_mean_nees_bootstrap_ci"],
            "anees_consistent": (
                class_tm["time_mean_nnees_compatible_with_1"]
            ),
            "anees_verdict": class_tm["time_mean_nnees_verdict"],
            "in_bound_fraction": float("nan"),
            "final_anees": class_final["anees"],
            "final_anees_bounds": class_final["anees_bounds"],
            "final_anees_consistent": class_final["anees_consistent"],
            "dof": dof_class,
        }
        result["class_metrics"][label] = class_result
        result[f"pos_rmse_{label}"] = class_result["pos_rmse"]
        result[f"rmte_{label}"] = class_result["rmte"]
        result[f"time_mean_nnees_{label}"] = class_result["time_mean_nnees"]
        result[f"time_mean_nnees_{label}_std"] = (
            class_result["time_mean_nnees_std"]
        )
        result[f"time_mean_nnees_{label}_bootstrap_ci"] = (
            class_result["time_mean_nnees_bootstrap_ci"]
        )
        result[f"anees_{label}"] = class_result["anees"]
        result[f"anees_{label}_bounds"] = class_result["anees_bounds"]
        result[f"anees_{label}_consistent"] = (
            class_result["anees_consistent"]
        )

    resource_keys = (
        "comm_measurement_bytes",
        "comm_consumed_measurement_bytes",
        "comm_estimation_bytes",
        "comm_total_bytes",
        "flops",
        "runtime_total_s",
        "runtime_predict_imu_s",
        "runtime_update_period_s",
        "memory_array_bytes",
    )
    for key in resource_keys:
        result[key] = _mean_std(sample[key])[0]
        result[f"{key}_std"] = _sample_std(sample[key])

    result["comm_measurement_kb"] = result["comm_measurement_bytes"] / 1024.0
    result["comm_estimation_kb"] = result["comm_estimation_bytes"] / 1024.0
    result["comm_total_kb"] = result["comm_total_bytes"] / 1024.0
    result["comm_kb"] = result["comm_total_kb"]
    result["flops_M"] = result["flops"] / 1e6
    result["memory_array_kb"] = result["memory_array_bytes"] / 1024.0
    result["runtime_samples"] = len(_finite(sample["runtime_total_s"]))
    return result

def _flat_summary_rows(results, K):
    rows = []
    for name, result in results.items():
        row = {key: value for key, value in result.items()
               if key != "class_metrics"}
        row.update({"method": name, "K": int(K)})
        for label, values in result["class_metrics"].items():
            for key, value in values.items():
                row[f"class_{label}_{key}"] = value
        rows.append(row)
    return rows


def _run_warmup(config, names):
    if not config.get("comparison_warmup", True):
        return
    warm = copy.deepcopy(config)
    warm["sim_duration"] = max(float(config["dt_uwb"]),
                               min(float(config["sim_duration"]),
                                   2.0 * float(config["dt_uwb"])))
    warm["run_preflight"] = False
    warm["save_plots"] = False
    run_trial(warm, int(config["seed_base"]) - 1, names, run_preflight=False)


def run_scenario(config, burn_in_frac=None, verbose=True):
    """Run one fixed-K M21 Monte-Carlo scenario."""
    validate_config(config)
    architecture, ablations, names = _method_groups(config)
    burn_in_frac = (float(config.get("comparison_burn_in_frac", 0.5))
                    if burn_in_frac is None else float(burn_in_frac))
    if not 0.0 <= burn_in_frac < 1.0:
        raise ValueError("burn_in_frac must lie in [0,1)")
    burn = _burn_index(config, burn_in_frac)
    samples = _blank_samples(names)
    per_run_rows = []
    motion_diagnostics = []
    seeds = [int(config["seed_base"]) + i for i in range(int(config["mc_runs"]))]

    _run_warmup(config, names)
    preflight_report = None
    for run_index, seed in enumerate(seeds):
        logs, _ = run_trial(config, seed, names)
        if preflight_report is None:
            preflight_report = logs["_meta"].get("preflight")
        motion_diagnostics.append(_motion_diagnostic(logs, seed))
        _append_trial(samples, per_run_rows, logs, names, config, burn, seed)
        if verbose and (run_index + 1) % max(1, len(seeds) // 5) == 0:
            print(f"  ...K={config['num_entities']} {run_index + 1}/{len(seeds)} runs")

    results = {name: _aggregate_method(name, samples[name], config)
               for name in names}

    # Paired same-seed TMNNEES differences relative to centralized.
    paired = {}
    if "centralized" in samples:
        central = np.asarray(samples["centralized"]["nees"], dtype=float)
        dof_pos = int(config["pos_dim"] * config["num_entities"])
        for method_index, name in enumerate(names):
            if name == "centralized":
                continue
            current = np.asarray(samples[name]["nees"], dtype=float)
            paired[name] = _scalar_summary(
                current / float(dof_pos) - central / float(dof_pos),
                config,
                bootstrap_offset=9000 + method_index,
            )

    output_dir = str(config["output_dir"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metadata = {
        "K": int(config["num_entities"]),
        "class_sizes": list(config["class_sizes"]),
        "entity_class_map": list(config["entity_class_map"]),
        "architecture_methods": architecture,
        "sensor_ablation_methods": ablations,
        "mc_runs": int(config["mc_runs"]),
        "seeds": seeds,
        "burn_in_fraction": burn_in_frac,
        "burn_in_epoch": burn,
        "nees_sampling": (
            "post-burn-in time mean within each seed, normalized by DoF, "
            "then summarized across independent seeds"
        ),
        "final_nees_usage": "audit only",
        "bootstrap_note": (
            "With only three seeds, bootstrap intervals are descriptive and "
            "not stable formal uncertainty intervals."
        ),
        "motion_diagnostics": motion_diagnostics,
        "rmse_aggregation": "pooled root mean squared entity-vector error",
        "runtime_repeats_requested": int(config["runtime_repeats"]),
        "runtime_samples_observed": int(config["mc_runs"]),
        "preflight": preflight_report,
    }
    payload = {
        "metadata": metadata,
        "methods": results,
        "paired_time_mean_nnees_vs_centralized": paired,
    }
    save_json(payload, os.path.join(output_dir, "comparison_results.json"))
    save_csv_rows(_flat_summary_rows(results, config["num_entities"]),
                  os.path.join(output_dir, "comparison_results.csv"))
    save_csv_rows(per_run_rows,
                  os.path.join(output_dir, "comparison_per_run.csv"))
    if motion_diagnostics:
        save_csv_rows(
            motion_diagnostics,
            os.path.join(output_dir, "comparison_motion_diagnostics.csv"),
        )
    save_csv_rows([row for row in _flat_summary_rows(results, config["num_entities"])
                   if row["method"] in architecture],
                  os.path.join(output_dir, "comparison_architectures.csv"))
    save_csv_rows([row for row in _flat_summary_rows(results, config["num_entities"])
                   if row["method"] in ablations],
                  os.path.join(output_dir, "comparison_sensor_ablations.csv"))
    save_config_snapshot(config, output_dir, "comparison_config_snapshot.json")
    plot_paths = plot_comparison(results, config, output_dir,
                                 architecture_methods=architecture,
                                 ablation_methods=ablations)
    metadata["plot_paths"] = plot_paths
    save_json(payload, os.path.join(output_dir, "comparison_results.json"))

    if verbose:
        _print_tables(results, architecture, ablations, config)
        for path in plot_paths or []:
            print("  plot ->", path)
    return payload


def run(config, burn_in_frac=None, verbose=True):
    """Run the K=8 main scenario and K=10 extension (or configured suite)."""
    validate_config(config)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    suite = {}
    for K, sizes in _scenario_specs(config):
        cfg = copy.deepcopy(config)
        apply_entity_count(cfg, K, sizes)
        cfg["output_dir"] = str(root / f"comparison_K{K}")
        validate_config(cfg)
        suite[f"K{K}"] = run_scenario(cfg, burn_in_frac, verbose)

    suite_metadata = {
        "scenarios": [key for key in suite],
        "entity_counts": [int(key[1:]) for key in suite],
        "main_scenario": "K8" if "K8" in suite else next(iter(suite)),
        "extension_scenario": "K10" if "K10" in suite else None,
    }
    payload = {"metadata": suite_metadata, "scenarios": suite}
    save_json(payload, root / "comparison_suite_results.json")
    save_config_snapshot(config, str(root), "comparison_suite_config_snapshot.json")
    return payload


def _print_tables(results, architecture, ablations, config):
    def show(title, names):
        print(f"\n=== {title}: K={config['num_entities']}, "
              f"MC={config['mc_runs']} ===")
        print("  %-12s %8s %8s %8s %9s %8s %9s %9s" %
              ("method", "RMSE", "RMTE", "RMSE/T", "TMNNEES", "std",
               "estComm", "runtime"))
        for name in names:
            result = results[name]
            print("  %-12s %8.3f %8.3f %8.3f %9.3f %8.3f %9.1f %9.3f" % (
                name, result["pos_rmse"], result["rmte"],
                result["rmse_rmte_ratio"], result["time_mean_nnees"],
                result["time_mean_nnees_std"],
                result["comm_estimation_kb"], result["runtime_total_s"]))
        print("  C-class position RMSE:", "  ".join(
            f"{name}={results[name]['pos_rmse_C']:.3f}" for name in names))

    show("Covariance architectures", architecture)
    show("Sensor ablations (not architecture ranking)", ablations)


if __name__ == "__main__":
    import tempfile
    from config import default_config

    cfg = default_config()
    apply_entity_count(cfg, 4, (1, 1, 2))
    cfg["mc_runs"] = 2
    cfg["sim_duration"] = 0.2
    cfg["save_plots"] = False
    cfg["run_preflight"] = False
    cfg["comparison_require_full_set"] = True
    cfg["comparison_run_extension"] = False
    cfg["comparison_entity_counts"] = [4]
    with tempfile.TemporaryDirectory() as directory:
        cfg["output_dir"] = directory
        result = run_scenario(cfg, verbose=False)
        assert set(result["methods"]) == set(MAIN_LINE_ESTIMATORS + SENSOR_ABLATION_ESTIMATORS)
        assert len(result["metadata"]["seeds"]) == 2
        for name, row in result["methods"].items():
            assert np.isfinite(row["pos_rmse"])
            assert set(row["class_metrics"]) == set(ENTITY_CLASSES)
            assert row["time_mean_nnees_num_seeds"] == 2
            assert np.isfinite(row["time_mean_nnees"])
            assert row["comm_measurement_bytes"] >= 0
            assert row["memory_array_bytes"] >= 0
        assert Path(directory, "comparison_per_run.csv").exists()
    print("[M21 ComparisonExperiment] OK  (7 methods + classes + resources + MC)")
