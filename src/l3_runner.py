"""M20 ScenarioRunner (L3 system orchestration).

The runner is the single place where truth, measurements, estimators,
evaluation, structural diagnostics, resource accounting and runtime profiling
meet.  All estimators consume the same immutable truth/measurement stream and
are initialised from the same sampled error realisation.

The public :func:`run_trial` API remains backward compatible with M21/M22: the
legacy scalar arrays (``rmse``, ``nees``, ``comm_bytes``...) are still present.
The same object now also carries the complete M20 TrialLog contract:

* shared truth history and per-method nominal/error/covariance histories;
* all M10 metrics, including A/B/C RMSE/RMTE/marginal NEES;
* split measurement/estimation communication and analytic FLOPs;
* M27 predict/update/epoch timings and exact covariance-array footprint;
* M12 reports at ``pre_uwb``, ``at_innovation`` and ``post_uwb``.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from config import ENTITY_CLASSES, STATE_DIM, validate_config
from l1_diagnostic import StructuralResidualDiagnostic
from l1_mechanization import mechanize
from l1_metrics import error_vectors, evaluate
from l1_large_scale_metrics import StreamingMetricAccumulator
from l1_runtime import RuntimeProfiler, fairness_report
from l1_trajectory import min_pairwise_distance_spread
from l2_estimators import build
from l2_sensor import SensorSimulator
from l2_truth import TruthSimulator


# A process-local cache prevents an expensive pre-flight from being repeated for
# every Monte-Carlo seed.  The cache key includes the complete public config and
# the estimator set, so every (v,T) grid point is checked independently.
_PREFLIGHT_CACHE: dict[Any, dict] = {}


def _freeze(value):
    """Hashable, deterministic representation of nested config values."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()
                            if not str(k).startswith("_")))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _pack_states(states):
    """Numerical snapshot of a list of NominalState objects."""
    return {
        "p": np.stack([s.p for s in states]),
        "v": np.stack([s.v for s in states]),
        "R": np.stack([s.R for s in states]),
        "b_a": np.stack([s.b_a for s in states]),
        "b_g": np.stack([s.b_g for s in states]),
    }


def _stack_state_history(rows):
    if not rows:
        return {k: np.empty((0,)) for k in ("p", "v", "R", "b_a", "b_g")}
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _covariance_arrays(estimator):
    """Covariance objects physically held by an estimator (no copies)."""
    arrays = []
    for attr in ("P", "P_G"):
        value = getattr(estimator, attr, None)
        if isinstance(value, np.ndarray):
            arrays.append(value)
    local = getattr(estimator, "P_local", None)
    if isinstance(local, (list, tuple)):
        arrays.extend(a for a in local if isinstance(a, np.ndarray))
    cross = getattr(estimator, "cross", None)
    if isinstance(cross, dict):
        arrays.extend(a for a in cross.values() if isinstance(a, np.ndarray))
    return arrays


def separation_check(config, seed=0, n_epochs=5, tol=1e-9):
    """T1: reconstruct truth from noiseless IMU after removing the TRUE bias.

    Bias is intentionally left non-zero.  Zeroing it would only test the easy
    special case and could not detect a sensor path that attaches the wrong
    bias sample to an IMU packet.
    """
    c = copy.deepcopy(config)
    for key in ("sigma_acc", "sigma_gyro", "sigma_range"):
        c[key] = 0.0
    validate_config(c)

    truth = TruthSimulator(c)
    init = truth.reset(seed)
    sensor = SensorSimulator(c)
    sensor.reset(seed + 1)
    rebuilt = [s.copy() for s in init.states]
    max_err = 0.0
    for _ in range(int(n_epochs)):
        for _k in range(c["imu_per_uwb"]):
            rec = truth.step()
            meas = sensor.generate_imu(rec)
            for i in range(c["num_entities"]):
                a_m, w_m = meas[i]
                rebuilt[i] = mechanize(
                    rebuilt[i], a_m - rec.b_a[i], w_m - rec.b_g[i],
                    c["dt_imu"], c["gravity"], c["nominal_int_order"])
                max_err = max(
                    max_err,
                    float(np.linalg.norm(rebuilt[i].p - rec.states[i].p)),
                    float(np.linalg.norm(rebuilt[i].v - rec.states[i].v)),
                    float(np.linalg.norm(rebuilt[i].R - rec.states[i].R)),
                )
    if max_err > tol:
        raise AssertionError(
            "separation_check FAILED: truth/measurement paths coupled "
            f"(max state error={max_err:.3e})")
    return max_err


def _initial_consistency(config, seed, estimator_names, truth_states):
    """All ESKF methods share the full error draw; M26 shares its position draw."""
    errors = {}
    for name in estimator_names:
        est = build(name, config)
        est.reset(truth_states, seed=seed + 500)
        errors[name] = error_vectors(truth_states, est.get_nominal_state())

    reference_name = next((n for n in estimator_names if n != "uwb_only"),
                          estimator_names[0])
    reference = errors[reference_name]
    full_gap = 0.0
    pos_gap = 0.0
    for name, err in errors.items():
        pos_gap = max(pos_gap, float(np.max(np.abs(err[:, :3] - reference[:, :3]))))
        if name != "uwb_only":
            full_gap = max(full_gap, float(np.max(np.abs(err - reference))))
    if pos_gap != 0.0 or full_gap != 0.0:
        raise AssertionError(
            "initial-state fairness failed: methods did not receive the same "
            f"sampled error (position gap={pos_gap:.3e}, full gap={full_gap:.3e})")
    return {"reference": reference_name,
            "position_error_max_gap": pos_gap,
            "eskf_full_error_max_gap": full_gap}


def preflight_check(config, estimator_names=None, seed=None, force=False):
    """M20 pre-flight: T1, topology, formation geometry and initial fairness.

    Results are returned as a serialisable report and cached for repeated
    Monte-Carlo trials using the same configuration.
    """
    validate_config(config)
    names = tuple(config["estimators"] if estimator_names is None
                  else estimator_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("estimator_names must be non-empty and unique")
    seed = int(config.get("seed_base", 0) if seed is None else seed)
    key = (_freeze(config), names)
    if not force and key in _PREFLIGHT_CACHE:
        return copy.deepcopy(_PREFLIGHT_CACHE[key])

    required_points = {"post_uwb", "pre_uwb", "at_innovation"}
    actual_points = set(config.get("diag_sample_points", []))
    if actual_points != required_points:
        raise ValueError(
            "diag_sample_points must contain exactly post_uwb, pre_uwb and "
            f"at_innovation; got {sorted(actual_points)}")

    separation = (separation_check(config, seed=seed)
                  if config.get("separation_check", True) else float("nan"))

    spread = float(min_pairwise_distance_spread(config))
    threshold = 0.2 * float(config["circle_radius"])
    if config["num_entities"] > 1 and spread < threshold:
        raise AssertionError(
            "circular-formation pre-flight failed: minimum pairwise distance "
            f"spread {spread:.6g} m < 0.2R = {threshold:.6g} m")

    truth = TruthSimulator(config)
    truth.reset(seed)
    sensor = SensorSimulator(config)
    sensor.reset(seed + 10_000)
    edges = sensor.generate_ranges(truth.snapshot(), check_topology=True)
    topology = sensor.topology_report()
    topology["realised_edges"] = len(edges)

    initial = _initial_consistency(
        config, seed, names, truth.get_truth_states())

    report = {
        "passed": True,
        "seed": seed,
        "estimators": list(names),
        "separation_max_error": separation,
        "distance_spread_m": spread,
        "distance_spread_threshold_m": threshold,
        "topology": topology,
        "initial_consistency": initial,
        "runtime_environment": fairness_report(),
    }
    _PREFLIGHT_CACHE[key] = copy.deepcopy(report)
    return report


def _new_method_log(config):
    points = list(config["diag_sample_points"])
    return {
        "metrics_rows": [],
        "nominal_history": [],
        "error_history": [],
        "eval_cov_history": [],
        "diagnostics": {point: [] for point in points},
        "runtime": {},
        "resource": {},
        # Backward-compatible keys consumed by the current M21/M22.
        "rmse": [], "vel_rmse": [], "att_rmse_deg": [],
        "acc_bias_rmse": [], "gyro_bias_rmse": [], "rmte": [],
        "nees": [], "nees_full": [], "nees_att": [],
        "comm_bytes": [], "flops": [],
        # New resource/timing series.
        "comm_measurement_bytes": [], "comm_estimation_bytes": [],
        "predict_period_s": [], "predict_imu_step_s": [],
        "update_period_s": [], "epoch_total_s": [],
        "covariance_array_bytes": [],
    }


def _append_metrics(log, row):
    log["metrics_rows"].append(row)
    # Preserve the historical names.
    aliases = {
        "rmse": "pos_rmse", "vel_rmse": "vel_rmse",
        "att_rmse_deg": "att_rmse_deg", "acc_bias_rmse": "acc_bias_rmse",
        "gyro_bias_rmse": "gyro_bias_rmse", "rmte": "rmte",
        "nees": "nees_pos", "nees_full": "nees_full",
        "nees_att": "nees_ch_att",
    }
    for alias, source in aliases.items():
        log[alias].append(float(row[source]))

    # Every numeric M10 field is also exposed under its canonical name.
    for key, value in row.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            if key in log and key in aliases:
                continue
            log.setdefault(key, []).append(float(value))

    # Compact A/B/C aliases are convenient in M21 tables.
    for label in ENTITY_CLASSES:
        log.setdefault(f"rmse_{label}", []).append(float(row[f"pos_rmse_{label}"]))
        # rmte_<class> is already the canonical M10 key and was appended above.
        log.setdefault(f"nees_{label}", []).append(float(row[f"nees_pos_{label}"]))


def _append_diag(log, point, report):
    log["diagnostics"][point].append(report)
    prefix = {"pre_uwb": "diag_pre", "at_innovation": "diag_innovation",
              "post_uwb": "diag_post"}[point]
    for key, value in report.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            log.setdefault(f"{prefix}_{key}", []).append(float(value))


def _finalise_method_log(log, profiler):
    structured = {"metrics_rows", "nominal_history", "error_history",
                  "eval_cov_history", "diagnostics", "runtime", "resource"}
    for key, value in list(log.items()):
        if key not in structured and isinstance(value, list):
            log[key] = np.asarray(value, dtype=float)
    log["nominal_history"] = _stack_state_history(log["nominal_history"])
    log["error_history"] = np.asarray(log["error_history"], dtype=float)
    log["eval_cov_history"] = np.asarray(log["eval_cov_history"], dtype=float)
    log["runtime"] = profiler.summary()


def _is_connected(K, pairs):
    if K <= 1:
        return True
    adjacency = [set() for _ in range(K)]
    for i, j in pairs:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in adjacency[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == K


_PAIR_SCOPES = ("AA", "AB", "AC", "BB", "BC", "CC")
_DIAG_ETA_KEYS = ("eta_struct", "eta_abs", "eta_dyn", "eta_S")
_DIAG_IDENTITY_KEYS = ("E_pp_max", "HEH_max")


def _pair_scope(config, i, j):
    labels = list(config["entity_class_map"])
    return "".join(sorted((str(labels[int(i)]), str(labels[int(j)]))))


def _select_diagnostic_pairs(config):
    """Select deterministic pairs with up to two representatives per class."""
    K = int(config["num_entities"])
    all_pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    limit = int(config.get("diag_max_pairs", 12))
    if limit == 0 or limit >= len(all_pairs):
        return all_pairs

    by_scope = {scope: [] for scope in _PAIR_SCOPES}
    for pair in all_pairs:
        by_scope[_pair_scope(config, *pair)].append(pair)

    chosen = []
    for scope in _PAIR_SCOPES:
        chosen.extend(by_scope[scope][:2])
    if len(chosen) < limit:
        used = set(chosen)
        chosen.extend(pair for pair in all_pairs if pair not in used)
    return chosen[:limit]


class _DiagnosticSummaryAccumulator:
    """Compact pair-class summary for sampled large-scale diagnostics."""

    def __init__(self, config):
        self.config = config
        self.values = {
            point: {scope: {} for scope in ("all",) + _PAIR_SCOPES}
            for point in config["diag_sample_points"]
        }
        self.sample_counts = {point: 0 for point in config["diag_sample_points"]}

    def update(self, sample_point, report):
        self.sample_counts[sample_point] += 1
        for pair, row in (report.get("pairs", {}) or {}).items():
            i, j = pair
            pair_scope = _pair_scope(self.config, i, j)
            for scope in ("all", pair_scope):
                target = self.values[sample_point][scope]
                for key in _DIAG_ETA_KEYS + _DIAG_IDENTITY_KEYS:
                    value = row.get(key)
                    if isinstance(value, (int, float, np.integer, np.floating)) \
                            and np.isfinite(float(value)):
                        target.setdefault(key, []).append(float(value))

    def finalize(self):
        result = {}
        for point, scopes in self.values.items():
            point_out = {}
            for scope, metrics in scopes.items():
                row = {"sample_count": int(self.sample_counts[point])}
                pair_value_count = 0
                for key, values in metrics.items():
                    arr = np.asarray(values, dtype=float)
                    if not arr.size:
                        continue
                    pair_value_count = max(pair_value_count, int(arr.size))
                    if key in _DIAG_IDENTITY_KEYS:
                        row[key] = float(np.max(arr))
                    else:
                        row[f"{key}_mean"] = float(np.mean(arr))
                        row[f"{key}_p95"] = float(np.quantile(arr, 0.95))
                        row[f"{key}_max"] = float(np.max(arr))
                row["pair_value_count"] = pair_value_count
                if pair_value_count or scope == "all":
                    point_out[scope] = row
            result[point] = point_out
        return result


def run_trial(config, seed, estimator_names, run_preflight=None):
    """Run one shared-data trial and return ``(logs, estimators)``.

    Setting ``large_scale_summary_only=True`` activates a memory-bounded server
    path: only streaming post-burn-in metrics, theoretical resources, motion
    diagnostics and topology summaries are retained.  The default path remains
    backward compatible with M21/M22 and keeps complete histories.
    """
    validate_config(config)
    names = list(estimator_names)
    if not names:
        raise ValueError("estimator_names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("estimator_names contains duplicates")

    summary_only = bool(config.get("large_scale_summary_only", False))
    burn_in_s = float(config.get("large_scale_burn_in_s", 20.0))
    if run_preflight is None:
        run_preflight = bool(config.get("run_preflight", True))
    preflight = (preflight_check(config, names, seed=seed)
                 if run_preflight else None)

    truth = TruthSimulator(config)
    truth.reset(seed)
    sensor = SensorSimulator(config)
    sensor.reset(seed + 10_000)
    initial_truth = truth.get_truth_states()

    enabled = bool(config.get("enable_runtime_profile", True)) and not summary_only
    estimators = {}
    profilers = {}
    accumulators = {}
    logs = {"_meta": {
        "seed": int(seed),
        "preflight": preflight,
        "truth_history": [],
        "topology": sensor.topology_report(),
        "topology_history": [],
        "diag_sample_points": list(config["diag_sample_points"]),
        "summary_only": summary_only,
    }}
    for name in names:
        est = build(name, config)
        est.reset(initial_truth, seed=seed + 500)
        estimators[name] = est
        profilers[name] = RuntimeProfiler(
            config, enabled=enabled, track_memory=False,
            keep_samples=not summary_only)
        logs[name] = _new_method_log(config)
        if summary_only:
            accumulators[name] = StreamingMetricAccumulator(
                config, est.evaluated_channels, burn_in_s=burn_in_s)

    diagnostic = (
        StructuralResidualDiagnostic(config, mode="structural")
        if ("centralized" in estimators and "dual_layer" in estimators
            and (not summary_only or config.get("diag_in_summary", False)))
        else None
    )
    diagnostic_pairs = (_select_diagnostic_pairs(config)
                        if summary_only and diagnostic is not None else None)
    diagnostic_acc = (_DiagnosticSummaryAccumulator(config)
                      if summary_only and diagnostic is not None else None)
    diag_stride_s = float(config.get("diag_sample_stride_s", 0.5))
    next_diag_time_s = diag_stride_s

    topology_acc = {
        "epochs": 0, "inter_sum": 0.0, "inter_min": float("inf"),
        "inter_max": 0, "degree_sum": 0.0, "degree_min": float("inf"),
        "degree_max": 0, "disconnected_epochs": 0,
    }
    speed_samples = []
    z_span_samples = []
    specific_force_samples = []
    xy_out_of_bounds = 0
    xy_position_count = 0

    steps = int(round(config["sim_duration"] / config["dt_uwb"]))
    for _epoch in range(steps):
        imu_records = []
        imu_seq = []
        for _k in range(config["imu_per_uwb"]):
            record = truth.step()
            imu_records.append(record)
            imu_seq.append(sensor.generate_imu(record))
        current_truth = truth.snapshot()
        edges = sensor.generate_ranges(current_truth, check_topology=True)
        truth_states = current_truth.states
        current_time_s = (_epoch + 1) * float(config["dt_uwb"])
        if not summary_only:
            logs["_meta"]["truth_history"].append(_pack_states(truth_states))

        inter_pairs = [(e.i, e.j) for e in edges if not e.is_anchor]
        degrees = np.zeros(config["num_entities"], dtype=int)
        for i, j in inter_pairs:
            degrees[i] += 1
            degrees[j] += 1
        n_anchor_epoch = sum(1 for e in edges if e.is_anchor)
        topology_row = {
            "epoch": int(_epoch),
            "inter_edges": int(len(inter_pairs)),
            "anchor_edges": int(n_anchor_epoch),
            "total_edges": int(len(edges)),
            "min_degree": int(np.min(degrees)) if len(degrees) else 0,
            "mean_degree": float(np.mean(degrees)) if len(degrees) else 0.0,
            "max_degree": int(np.max(degrees)) if len(degrees) else 0,
        }
        if summary_only:
            topology_acc["epochs"] += 1
            topology_acc["inter_sum"] += topology_row["inter_edges"]
            topology_acc["inter_min"] = min(
                topology_acc["inter_min"], topology_row["inter_edges"])
            topology_acc["inter_max"] = max(
                topology_acc["inter_max"], topology_row["inter_edges"])
            topology_acc["degree_sum"] += topology_row["mean_degree"]
            topology_acc["degree_min"] = min(
                topology_acc["degree_min"], topology_row["min_degree"])
            topology_acc["degree_max"] = max(
                topology_acc["degree_max"], topology_row["max_degree"])
            if not _is_connected(int(config["num_entities"]), inter_pairs):
                topology_acc["disconnected_epochs"] += 1
            if current_time_s > burn_in_s:
                speed_samples.append(np.asarray(
                    [np.linalg.norm(st.v) for st in truth_states], dtype=float))
                z = np.asarray([st.p[2] for st in truth_states], dtype=float)
                z_span_samples.append(float(np.max(z) - np.min(z)))
                specific_force_samples.append(np.concatenate([
                    np.linalg.norm(np.asarray(record.f_b, dtype=float), axis=1)
                    for record in imu_records
                ]))
                xy = np.asarray([st.p[:2] for st in truth_states], dtype=float)
                field = float(config["field_size"])
                xy_out_of_bounds += int(np.count_nonzero(
                    (xy[:, 0] < 0.0) | (xy[:, 0] > field)
                    | (xy[:, 1] < 0.0) | (xy[:, 1] > field)))
                xy_position_count += int(xy.shape[0])
        else:
            logs["_meta"]["topology_history"].append(topology_row)

        predict_elapsed = {}
        update_elapsed = {}
        for name in names:
            est = estimators[name]
            prof = profilers[name]
            if summary_only:
                est.predict(imu_seq, config["dt_uwb"])
            else:
                t0 = time.perf_counter()
                with prof.profile("predict_period"):
                    est.predict(imu_seq, config["dt_uwb"])
                elapsed = time.perf_counter() - t0
                predict_elapsed[name] = elapsed
                logs[name]["predict_period_s"].append(elapsed)
                logs[name]["predict_imu_step_s"].append(
                    elapsed / int(config["imu_per_uwb"]))

        sample_diagnostic = (
            diagnostic is not None
            and (not summary_only
                 or current_time_s + 1e-12 >= next_diag_time_s)
        )
        pre_inputs = None
        if sample_diagnostic:
            central = estimators["centralized"]
            dual = estimators["dual_layer"]
            pre_inputs = (central.get_full_cov(), dual.get_global_cov(),
                          dual.get_phi_bar())
            report = diagnostic.evaluate(
                pre_inputs[0], P_G=pre_inputs[1],
                Phi_bar_list=pre_inputs[2], sample_point="pre_uwb",
                pairs=diagnostic_pairs)
            if summary_only:
                diagnostic_acc.update("pre_uwb", report)
            else:
                _append_diag(logs["dual_layer"], "pre_uwb", report)

        for name in names:
            est = estimators[name]
            prof = profilers[name]
            if summary_only:
                est.update(edges)
            else:
                t0 = time.perf_counter()
                with prof.profile("update_period"):
                    est.update(edges)
                elapsed = time.perf_counter() - t0
                update_elapsed[name] = elapsed
                logs[name]["update_period_s"].append(elapsed)

        if sample_diagnostic:
            central = estimators["centralized"]
            dual = estimators["dual_layer"]
            report = diagnostic.evaluate(
                pre_inputs[0], P_G=pre_inputs[1],
                Phi_bar_list=pre_inputs[2],
                innovations=dual.get_innovations(),
                sample_point="at_innovation", pairs=diagnostic_pairs)
            if summary_only:
                diagnostic_acc.update("at_innovation", report)
            else:
                _append_diag(logs["dual_layer"], "at_innovation", report)
            report = diagnostic.evaluate(
                central.get_full_cov(), P_G=dual.get_global_cov(),
                sample_point="post_uwb", pairs=diagnostic_pairs)
            if summary_only:
                diagnostic_acc.update("post_uwb", report)
                while next_diag_time_s <= current_time_s + 1e-12:
                    next_diag_time_s += diag_stride_s
            else:
                _append_diag(logs["dual_layer"], "post_uwb", report)

        for name in names:
            est = estimators[name]
            prof = profilers[name]
            full_cov = (est.get_eval_cov_full()
                        if hasattr(est, "get_eval_cov_full") else None)
            nominal = est.get_nominal_state()
            eval_cov = est.get_eval_cov()
            E = error_vectors(truth_states, nominal)
            if summary_only:
                accumulators[name].update(
                    current_time_s, E, eval_cov, full_cov)
            else:
                t0 = time.perf_counter()
                with prof.profile("evaluation"):
                    row = evaluate(
                        truth_states, nominal, eval_cov, config,
                        full_cov=full_cov, channels=est.evaluated_channels)
                evaluation_elapsed = time.perf_counter() - t0
                _append_metrics(logs[name], row)
                logs[name]["nominal_history"].append(_pack_states(nominal))
                logs[name]["error_history"].append(E)
                logs[name]["eval_cov_history"].append(eval_cov.copy())

                comm = est.get_comm_summary()
                logs[name]["comm_measurement_bytes"].append(
                    comm["C_measurement_bytes"])
                logs[name]["comm_estimation_bytes"].append(
                    comm["C_estimation_bytes"])
                logs[name]["comm_bytes"].append(comm["C_total_bytes"])
                logs[name]["flops"].append(est.get_flops())
                cov_bytes = prof.record_arrays(
                    "covariance", _covariance_arrays(est))
                logs[name]["covariance_array_bytes"].append(cov_bytes)
                logs[name]["epoch_total_s"].append(
                    predict_elapsed[name] + update_elapsed[name]
                    + evaluation_elapsed)

    if summary_only:
        logs["_meta"]["truth_history"] = {}
        n_top = max(1, int(topology_acc["epochs"]))
        logs["_meta"]["topology_summary"] = {
            "epochs": int(topology_acc["epochs"]),
            "inter_edges_mean": topology_acc["inter_sum"] / n_top,
            "inter_edges_min": (int(topology_acc["inter_min"])
                                if np.isfinite(topology_acc["inter_min"]) else 0),
            "inter_edges_max": int(topology_acc["inter_max"]),
            "mean_degree": topology_acc["degree_sum"] / n_top,
            "min_degree": (int(topology_acc["degree_min"])
                           if np.isfinite(topology_acc["degree_min"]) else 0),
            "max_degree": int(topology_acc["degree_max"]),
            "disconnected_epoch_fraction": (
                topology_acc["disconnected_epochs"] / n_top),
        }
        speeds = (np.concatenate(speed_samples) if speed_samples
                  else np.empty(0, dtype=float))
        spans = np.asarray(z_span_samples, dtype=float)
        specific_force = (np.concatenate(specific_force_samples)
                          if specific_force_samples else np.empty(0, dtype=float))
        logs["_meta"]["motion_summary"] = {
            "actual_mean_3d_speed": (float(np.mean(speeds))
                                     if speeds.size else float("nan")),
            "actual_std_3d_speed": (float(np.std(speeds, ddof=1))
                                    if speeds.size > 1 else float("nan")),
            "actual_speed_p05": (float(np.quantile(speeds, 0.05))
                                 if speeds.size else float("nan")),
            "actual_speed_p95": (float(np.quantile(speeds, 0.95))
                                 if speeds.size else float("nan")),
            "specific_force_mean": (float(np.mean(specific_force))
                                    if specific_force.size else float("nan")),
            "specific_force_p95": (float(np.quantile(specific_force, 0.95))
                                   if specific_force.size else float("nan")),
            "specific_force_max": (float(np.max(specific_force))
                                   if specific_force.size else float("nan")),
            "xy_out_of_bounds_fraction": (
                float(xy_out_of_bounds) / xy_position_count
                if xy_position_count else float("nan")),
            "z_span_mean": (float(np.mean(spans))
                            if spans.size else float("nan")),
            "z_span_min": (float(np.min(spans))
                           if spans.size else float("nan")),
            "z_span_final": (float(spans[-1])
                             if spans.size else float("nan")),
        }
        if diagnostic_acc is not None:
            logs["_meta"]["diagnostics_summary"] = diagnostic_acc.finalize()
        else:
            logs["_meta"]["diagnostics_summary"] = {}
    else:
        logs["_meta"]["truth_history"] = _stack_state_history(
            logs["_meta"]["truth_history"])

    for name in names:
        if summary_only:
            logs[name]["stream_metrics"] = accumulators[name].finalize()
            logs[name]["nominal_history"] = {}
            logs[name]["error_history"] = np.empty((0,), dtype=float)
            logs[name]["eval_cov_history"] = np.empty((0,), dtype=float)
            logs[name]["runtime"] = {"enabled": False, "sections": {},
                                     "memory": {}}
        else:
            _finalise_method_log(logs[name], profilers[name])
        comm = estimators[name].get_comm_summary()
        memory_bytes = (0 if summary_only else int(
            profilers[name].summary()["memory"]["array_bytes_total"]))
        logs[name]["resource"] = {
            **comm,
            "flops": estimators[name].get_flops(),
            "covariance_array_bytes": memory_bytes,
        }
        if not summary_only:
            sections = logs[name]["runtime"].get("sections", {})
            pred = sections.get("predict_period", {})
            if pred:
                logs[name]["runtime"]["predict_imu_step_mean_s"] = (
                    pred["mean_s"] / int(config["imu_per_uwb"]))
            upd = sections.get("update_period", {})
            if upd:
                logs[name]["runtime"]["update_period_mean_s"] = upd["mean_s"]
            logs[name]["runtime"]["estimator_total_s"] = float(sum(
                sections.get(section, {}).get("total_s", 0.0)
                for section in ("predict_period", "update_period", "evaluation")
            ))
    return logs, estimators


if __name__ == "__main__":
    from config import apply_entity_count, default_config

    # Full pre-flight on a valid experiment horizon.
    pre = default_config()
    apply_entity_count(pre, 3, (1, 1, 1))
    validate_config(pre)
    report = preflight_check(pre, ["centralized", "dual_layer"], seed=0,
                             force=True)
    assert report["passed"] and report["separation_max_error"] < 1e-12

    # Tiny numerical smoke test deliberately bypasses the 0.2R horizon rule.
    cfg = default_config()
    apply_entity_count(cfg, 3, (1, 1, 1))
    cfg["sim_duration"] = 0.2
    cfg["separation_check"] = False
    validate_config(cfg)
    names = ["centralized", "dual_layer", "local_only", "ci_trace", "ci_det",
             "imu_only", "uwb_only"]
    logs, _ = run_trial(cfg, seed=0, estimator_names=names,
                        run_preflight=False)
    for name in names:
        assert logs[name]["rmse"].shape == (2,)
        assert np.all(np.isfinite(logs[name]["rmse"]))
        assert logs[name]["rmse_A"].shape == (2,)
        assert logs[name]["comm_measurement_bytes"].shape == (2,)
        assert logs[name]["eval_cov_history"].shape[0] == 2
        assert logs[name]["runtime"]["sections"]["predict_period"]["calls"] == 2
    for point in cfg["diag_sample_points"]:
        assert len(logs["dual_layer"]["diagnostics"][point]) == 2
    replay, _ = run_trial(cfg, seed=0, estimator_names=["dual_layer"],
                          run_preflight=False)
    assert np.allclose(logs["dual_layer"]["rmse"],
                       replay["dual_layer"]["rmse"])
    print("[M20 ScenarioRunner] OK  (preflight + 3 diagnostics + M27 + full TrialLog)")
