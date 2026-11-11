"""Focused acceptance gate for M20 ScenarioRunner.

Covers the responsibilities that are not already pinned by the L1/L2 module
self-tests: pre-flight, shared initial conditions, rich TrialLog, diagnostic
sampling order, communication split and profiler zero impact.
"""

import copy
import numpy as np

from config import apply_entity_count, default_config, validate_config
from l3_runner import preflight_check, run_trial


def _small_config(profile=True):
    cfg = default_config()
    apply_entity_count(cfg, 3, (1, 1, 1))
    cfg["sim_duration"] = 0.2
    cfg["separation_check"] = False
    cfg["run_preflight"] = False
    cfg["enable_runtime_profile"] = bool(profile)
    validate_config(cfg)
    return cfg


def main():
    # 1) Full pre-flight succeeds on a valid horizon.
    cfg = default_config()
    apply_entity_count(cfg, 3, (1, 1, 1))
    validate_config(cfg)
    report = preflight_check(
        cfg, ["centralized", "dual_layer", "local_only", "ci_trace", "ci_det",
              "imu_only", "uwb_only"], seed=123, force=True)
    assert report["passed"]
    assert report["separation_max_error"] < 1e-12
    assert report["distance_spread_m"] >= report["distance_spread_threshold_m"]
    assert report["topology"]["realised_edges"] == 9  # K=3: 3 mesh + 6 anchors
    assert report["initial_consistency"]["position_error_max_gap"] == 0.0
    assert report["initial_consistency"]["eskf_full_error_max_gap"] == 0.0

    # 2) Quantitatively degenerate horizons are rejected even when phases,
    # directions and centre offsets satisfy M24's structural enum check.
    short = default_config()
    short["sim_duration"] = short["sync_interval"]
    validate_config(short)
    try:
        preflight_check(short, ["centralized", "dual_layer"], force=True)
        raise AssertionError("short degenerate horizon passed M20 pre-flight")
    except AssertionError as exc:
        assert "distance spread" in str(exc)

    # 3) Complete TrialLog and the three causal diagnostic labels.
    smoke = _small_config(profile=True)
    names = ["centralized", "dual_layer", "local_only", "ci_trace", "ci_det",
             "imu_only", "uwb_only"]
    logs, _ = run_trial(smoke, seed=7, estimator_names=names)
    assert logs["_meta"]["truth_history"]["p"].shape == (2, 3, 3)
    for name in names:
        log = logs[name]
        for key in ("rmse_A", "rmte_B", "nees_C",
                    "comm_measurement_bytes", "comm_estimation_bytes",
                    "predict_imu_step_s", "update_period_s",
                    "covariance_array_bytes"):
            assert log[key].shape == (2,), (name, key, log[key].shape)
        assert log["nominal_history"]["R"].shape == (2, 3, 3, 3)
        assert log["error_history"].shape == (2, 3, 15)
        assert log["eval_cov_history"].shape == (2, 9, 9)
        assert log["resource"]["C_total_bytes"] == log["comm_bytes"][-1]
        assert log["runtime"]["predict_imu_step_mean_s"] >= 0.0
    dual = logs["dual_layer"]
    for point in ("pre_uwb", "at_innovation", "post_uwb"):
        assert len(dual["diagnostics"][point]) == 2
        assert all(r["sample_point"] == point for r in dual["diagnostics"][point])
    assert np.all(np.isfinite(dual["diag_pre_eta_dyn_mean"]))
    assert np.all(np.isfinite(dual["diag_innovation_eta_S_mean"]))
    assert np.all(np.isfinite(dual["diag_post_eta_struct_mean"]))

    # 4) T23 at runner level: profiler on/off changes no numerical TrialLog.
    no_profile = _small_config(profile=False)
    a, _ = run_trial(smoke, seed=11,
                     estimator_names=["centralized", "dual_layer"])
    b, _ = run_trial(no_profile, seed=11,
                     estimator_names=["centralized", "dual_layer"])
    numerical = ("rmse", "rmte", "nees", "nees_full", "flops",
                 "comm_measurement_bytes", "comm_estimation_bytes")
    for name in ("centralized", "dual_layer"):
        for key in numerical:
            assert np.array_equal(a[name][key], b[name][key], equal_nan=True), (name, key)
        assert np.array_equal(a[name]["error_history"], b[name]["error_history"],
                              equal_nan=True)
        assert np.array_equal(a[name]["eval_cov_history"],
                              b[name]["eval_cov_history"], equal_nan=True)

    print("[M20 acceptance] PASS: pre-flight, full TrialLog, 3 diagnostics, "
          "resource split and T23")


if __name__ == "__main__":
    main()
