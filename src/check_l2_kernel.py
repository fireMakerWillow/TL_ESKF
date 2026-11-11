"""Task-2 gate: run the evaluation- and simulation-layer self-tests, then check
that they COMPOSE with the Task-1 kernel.

Per-module ``__main__`` blocks test each module alone.  This script additionally
runs the chain the experiment layer will depend on:

    M13 truth -> M14 measurements -> M4/M7 propagation
              -> M10 RMSE / RMTE / NEES
              -> M12 eta_struct / eta_dyn on a REAL joint covariance
              -> M11 analytic resources, M27 measured runtime

and asserts the four facts the later tasks stand on:

    (a) truth and measurements are separated       (T1 / T2 / T3 precursor)
    (b) the graph is the full mesh plus the class-derived anchor set   (T19)
    (c) dead reckoning has EXACTLY zero position cross blocks and its joint
        position ANEES lands in band                (T13 / T14 precursor)
    (d) E_pp = 0 and H E H^T = 0 on a genuine propagated-and-updated joint
        covariance, with eta_struct > 0             (T16 on real data)

The joint covariance used in (d) is built from the L1 primitives alone -- a
covariance recursion with no nominal injection and no reset.  It is a PROBE, not
an estimator: M15 is Task 3's deliverable and nothing here should be mistaken
for it.
"""

import subprocess
import sys

import numpy as np

MODULES = [
    "config.py", "l1_linalg.py", "l1_so3.py", "l1_geometry.py",
    "l1_mechanization.py", "l1_trajectory.py", "l1_true_bias.py",
    "l1_process_model.py", "l1_range_model.py", "l1_regression.py",
    "l1_metrics.py", "l1_resource.py", "l1_diagnostic.py", "l1_runtime.py",
    "l2_truth.py", "l2_sensor.py",
]
TASK2_MODULES = {"l1_metrics.py", "l1_resource.py", "l1_diagnostic.py",
                 "l1_runtime.py", "l2_truth.py", "l2_sensor.py"}


def run_unit_tests():
    failures = []
    for name in MODULES:
        proc = subprocess.run([sys.executable, name], capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.strip().splitlines()
                 if not ln.startswith("[config][WARNING]")]
        tail = lines[-1] if lines else ""
        status = "PASS" if proc.returncode == 0 else "FAIL"
        mark = "*" if name in TASK2_MODULES else " "
        if proc.returncode != 0:
            failures.append((name, proc.stderr.strip().splitlines()[-3:]))
        print(f"  [{status}]{mark} {name:<22} {tail}")
    return failures


# ----------------------------------------------------------------------
# probes built from L1 primitives only
# ----------------------------------------------------------------------
def _initial_covariance(cfg):
    from config import IDX_P, IDX_V, IDX_TH, IDX_BA, IDX_BG, STATE_DIM
    d = np.zeros(STATE_DIM)
    d[IDX_P] = cfg["init_pos_std"] ** 2
    d[IDX_V] = cfg["init_vel_std"] ** 2
    d[IDX_TH] = cfg["init_att_std"] ** 2
    d[IDX_BA] = cfg["init_acc_bias_std"] ** 2
    d[IDX_BG] = cfg["init_gyro_bias_std"] ** 2
    return np.diag(d)


def _sample_initial_nominal(truth_states, cfg, seed):
    """nominal = truth boxplus(-e0), e0 ~ N(0, P0) -- the M10 sign convention."""
    from config import STATE_DIM, IDX_P, IDX_V, IDX_TH, IDX_BA, IDX_BG
    from l1_so3 import boxplus
    rng = np.random.default_rng(seed)
    std = np.zeros(STATE_DIM)
    std[IDX_P] = cfg["init_pos_std"]; std[IDX_V] = cfg["init_vel_std"]
    std[IDX_TH] = cfg["init_att_std"]; std[IDX_BA] = cfg["init_acc_bias_std"]
    std[IDX_BG] = cfg["init_gyro_bias_std"]
    return [boxplus(st, -(std * rng.standard_normal(STATE_DIM)))
            for st in truth_states]


def dead_reckoning_trial(cfg, seed, profiler=None):
    """Pure inertial propagation: nominal states plus a BLOCK-DIAGONAL P.

    This is the evaluation-chain probe of plan section 0.3: with no ranges the
    cross-entity position covariance is zero BY CONSTRUCTION, so its joint
    position NEES must land in band.  If it does not, the fault is in M10 or in
    M13/M14, not in any filter.
    """
    from config import POS_DIM, STATE_DIM
    from l1_linalg import symmetrize
    from l1_mechanization import correct_imu, mechanize
    from l1_metrics import evaluate
    from l1_process_model import ProcessModel
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    K, dt = int(cfg["num_entities"]), float(cfg["dt_imu"])
    truth = TruthSimulator(cfg); truth.reset(seed)
    sensor = SensorSimulator(cfg); sensor.reset(seed + 10_000)
    pm = ProcessModel(cfg)

    nominal = _sample_initial_nominal(truth.get_truth_states(), cfg, seed + 500)
    P = [_initial_covariance(cfg) for _ in range(K)]
    epochs = int(round(float(cfg["sim_duration"]) / float(cfg["sync_interval"])))
    rows = []
    for _ in range(epochs):
        with (profiler.profile("predict") if profiler else _null()):
            for _k in range(int(cfg["imu_per_uwb"])):
                record = truth.step()
                imu = sensor.generate_imu(record)
                for i in range(K):
                    a_m, w_m = imu[i]
                    f_hat, w_hat = correct_imu(a_m, w_m, nominal[i].b_a,
                                               nominal[i].b_g)
                    Phi, Q = pm.step(nominal[i].R, f_hat, w_hat, dt)
                    P[i] = symmetrize(Phi @ P[i] @ Phi.T + Q)
                    b_a, b_g = nominal[i].b_a, nominal[i].b_g
                    nominal[i] = mechanize(nominal[i], f_hat, w_hat, dt,
                                           cfg["gravity"], cfg["nominal_int_order"])
                    nominal[i].b_a, nominal[i].b_g = b_a, b_g   # bias held
        snapshot = truth.snapshot()
        sensor.generate_ranges(snapshot)          # generated and discarded
        P_pos = np.zeros((POS_DIM * K, POS_DIM * K))
        P_full = np.zeros((STATE_DIM * K, STATE_DIM * K))
        for i in range(K):
            P_pos[i * POS_DIM:(i + 1) * POS_DIM,
                  i * POS_DIM:(i + 1) * POS_DIM] = P[i][:POS_DIM, :POS_DIM]
            P_full[i * STATE_DIM:(i + 1) * STATE_DIM,
                   i * STATE_DIM:(i + 1) * STATE_DIM] = P[i]
        rows.append(evaluate(snapshot.states, nominal, P_pos, cfg,
                             full_cov=P_full))
    return rows, P_pos, P_full


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def joint_covariance_probe(cfg, seed, epochs=3):
    """A genuine 15K x 15K joint covariance with non-zero cross blocks.

    Covariance recursion only: block-diagonal Phi/Q propagation followed by
    sequential scalar range updates linearised at the TRUE positions.  Ranges
    couple the entities, so after one update the cross blocks are non-zero and
    M12 has something real to diagnose.
    """
    from config import POS_DIM, STATE_DIM
    from l1_linalg import symmetrize
    from l1_process_model import ProcessModel
    from l1_range_model import build_H_state
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    K, dt = int(cfg["num_entities"]), float(cfg["dt_imu"])
    truth = TruthSimulator(cfg); truth.reset(seed)
    sensor = SensorSimulator(cfg); sensor.reset(seed + 7)
    pm = ProcessModel(cfg)

    P_x = np.zeros((STATE_DIM * K, STATE_DIM * K))
    for i in range(K):
        P_x[i * STATE_DIM:(i + 1) * STATE_DIM,
            i * STATE_DIM:(i + 1) * STATE_DIM] = _initial_covariance(cfg)
    R_scalar = float(cfg["sigma_range"]) ** 2

    for _ in range(int(epochs)):
        Phi_bar = [np.eye(STATE_DIM) for _ in range(K)]
        for _k in range(int(cfg["imu_per_uwb"])):
            record = truth.step()
            imu = sensor.generate_imu(record)
            Phi_joint = np.zeros_like(P_x)
            Q_joint = np.zeros_like(P_x)
            for i in range(K):
                a_m, w_m = imu[i]
                Phi, Q = pm.step(record.states[i].R,
                                 a_m - record.b_a[i], w_m - record.b_g[i], dt)
                sl = slice(i * STATE_DIM, (i + 1) * STATE_DIM)
                Phi_joint[sl, sl] = Phi
                Q_joint[sl, sl] = Q
                Phi_bar[i] = Phi @ Phi_bar[i]
            P_x = symmetrize(Phi_joint @ P_x @ Phi_joint.T + Q_joint)

        snapshot = truth.snapshot()
        positions = snapshot.positions()
        edges = sensor.generate_ranges(snapshot)
        H = build_H_state([e.to_edge() for e in edges], positions,
                          cfg["anchor_positions"], K)
        for row in H:                                   # sequential scalar updates
            S = float(row @ P_x @ row) + R_scalar
            k = (P_x @ row) / S
            P_x = symmetrize(P_x - np.outer(k, row @ P_x))
    P_G = np.zeros((POS_DIM * K, POS_DIM * K))
    for i in range(K):
        for j in range(K):
            P_G[i * POS_DIM:(i + 1) * POS_DIM, j * POS_DIM:(j + 1) * POS_DIM] = \
                P_x[i * STATE_DIM:i * STATE_DIM + POS_DIM,
                    j * STATE_DIM:j * STATE_DIM + POS_DIM]
    return P_x, P_G, Phi_bar, edges, positions


# ----------------------------------------------------------------------
def composition_check():
    from config import (default_config, validate_config, POS_DIM, STATE_DIM,
                        ENTITY_CLASSES, class_indices)
    from l1_diagnostic import StructuralResidualDiagnostic
    from l1_metrics import summarize_nees
    from l1_resource import CommunicationLedger, comparison_table
    from l1_runtime import RuntimeProfiler, fairness_report, repeat_timing
    from l1_range_model import predict_range
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    cfg = default_config()
    cfg["sim_duration"] = 1.0
    validate_config(cfg)
    K = int(cfg["num_entities"])
    info = {}

    # ---- (a) truth / measurement separation, end to end ----
    clean = default_config(); clean["sim_duration"] = 1.0
    for key in ("sigma_acc", "sigma_gyro", "sigma_range",
                "true_bias_init_std_acc", "true_bias_init_std_gyro",
                "true_bias_rw_acc", "true_bias_rw_gyro"):
        clean[key] = 0.0
    validate_config(clean)
    from l1_mechanization import mechanize
    truth = TruthSimulator(clean); init = truth.reset(0)
    sensor = SensorSimulator(clean); sensor.reset(1)
    rebuilt = [s.copy() for s in init.states]
    worst = 0.0
    for _ in range(int(round(clean["sim_duration"] / clean["dt_imu"]))):
        rec = truth.step()
        imu = sensor.generate_imu(rec)
        for i in range(clean["num_entities"]):
            a_m, w_m = imu[i]
            rebuilt[i] = mechanize(rebuilt[i], a_m - rebuilt[i].b_a,
                                   w_m - rebuilt[i].b_g, clean["dt_imu"],
                                   clean["gravity"], clean["nominal_int_order"])
            worst = max(worst, float(np.linalg.norm(rebuilt[i].p - rec.states[i].p)))
    assert worst < 1e-9, worst
    info["separation_error"] = worst

    # ---- (b) the graph is what the class map says it is ----
    t2 = TruthSimulator(cfg); r2 = t2.reset(0)
    s2 = SensorSimulator(cfg); s2.reset(1)
    edges = s2.generate_ranges(r2)
    report = s2.topology_report()
    expected_anchor = (4 * len(class_indices(cfg)["A"])
                       + 2 * len(class_indices(cfg)["B"]))
    expected_total = K * (K - 1) // 2 + expected_anchor
    assert len(edges) == report["total_edges_per_epoch"] == expected_total
    assert report["anchor_edges_per_epoch"] == expected_anchor
    info["edges"] = len(edges)

    # ---- (c) dead reckoning: exact zero cross blocks, ANEES in band ----
    profiler = RuntimeProfiler(cfg, enabled=True, track_memory=True)
    profiler.start_memory()
    trials, P_pos, P_full = [], None, None
    n_trials = 60
    for seed in range(n_trials):
        rows, P_pos, P_full = dead_reckoning_trial(cfg, 3000 + seed, profiler)
        trials.append(rows[-1])
    profiler.stop_memory()

    off = np.array(P_pos, copy=True)
    for i in range(K):
        off[i * POS_DIM:(i + 1) * POS_DIM, i * POS_DIM:(i + 1) * POS_DIM] = 0.0
    assert float(np.max(np.abs(off))) == 0.0, "dead reckoning grew a cross block"

    # The gate checks five bands at once (joint, A, B, C, full state).  At the
    # paper's 95% level that is a ~23% chance of a red gate on a CORRECT build,
    # so the assertions use a 99.9% band (family-wise ~0.5%) while the reported
    # verdict stays at ``nees_confidence``.  Widening the gate is the honest
    # move here; loosening the paper's level would not be.
    GATE_CONFIDENCE = 0.999
    nees_samples = [row["nees_pos"] for row in trials]
    summary = summarize_nees(nees_samples, POS_DIM * K, cfg["nees_confidence"])
    gate = summarize_nees(nees_samples, POS_DIM * K, GATE_CONFIDENCE)
    assert gate["anees_consistent"], gate
    class_summaries = {}
    for label in ENTITY_CLASSES:
        dof = trials[0][f"nees_pos_{label}_dof"]
        values = [row[f"nees_pos_{label}"] for row in trials]
        class_summaries[label] = summarize_nees(values, dof,
                                                cfg["nees_confidence"])
        assert summarize_nees(values, dof, GATE_CONFIDENCE)["anees_consistent"], \
            (label, class_summaries[label])
    full = summarize_nees([row["nees_full"] for row in trials],
                          STATE_DIM * K, cfg["nees_confidence"])
    assert summarize_nees([row["nees_full"] for row in trials], STATE_DIM * K,
                          GATE_CONFIDENCE)["anees_consistent"], full
    info["class_anees"] = {c: class_summaries[c]["anees"] for c in ENTITY_CLASSES}
    info["class_dof"] = {c: trials[0][f"nees_pos_{c}_dof"] for c in ENTITY_CLASSES}
    info["anees"] = summary["anees"]
    info["anees_bounds"] = summary["anees_bounds"]
    info["anees_full"] = full["anees"]
    info["rmse"] = float(np.mean([row["pos_rmse"] for row in trials]))
    info["rmte"] = float(np.mean([row["rmte"] for row in trials]))
    info["ratio"] = info["rmse"] / info["rmte"]
    # a consistent filter sits at 1: the RMSE convention of M10 is the
    # per-entity magnitude precisely so that this holds
    assert 0.75 < info["ratio"] < 1.35, info["ratio"]
    # the per-class split recombines into the global RMSE
    weights = [cfg["entity_class_map"].count(c) for c in ENTITY_CLASSES]
    recomb = np.sqrt(sum(w * np.mean([r[f"pos_rmse_{c}"] for r in trials]) ** 2
                         for w, c in zip(weights, ENTITY_CLASSES)) / K)
    assert abs(recomb - np.sqrt(np.mean([r["pos_rmse"] ** 2 for r in trials]))) \
        < 0.05 * info["rmse"]

    # ---- (d) M12 on a real joint covariance ----
    P_x, P_G, Phi_bar, probe_edges, positions = joint_covariance_probe(cfg, 42)
    profiler.record_arrays("P_x", P_x)
    profiler.record_arrays("P_G", P_G)
    cross = P_x[0:STATE_DIM, STATE_DIM:2 * STATE_DIM]
    assert float(np.max(np.abs(cross))) > 0.0, "the probe grew no cross block"
    innovations = []
    for e in probe_edges:
        if e.is_anchor:
            continue
        _, u = predict_range(e.to_edge(), positions, cfg["anchor_positions"])
        idx = np.concatenate([np.arange(e.i * POS_DIM, (e.i + 1) * POS_DIM),
                              np.arange(e.j * POS_DIM, (e.j + 1) * POS_DIM)])
        h = np.concatenate([u, -u])
        S = float(h @ P_G[np.ix_(idx, idx)] @ h) + cfg["sigma_range"] ** 2
        innovations.append({"i": e.i, "j": e.j, "u": u, "S": S})

    diagnostic = StructuralResidualDiagnostic(cfg)
    diag = diagnostic.evaluate(P_x, P_G, Phi_bar, innovations=innovations,
                               sample_point="post_uwb")
    assert diag["E_pp_max"] < 1e-9 and diag["HEH_max"] < 1e-9
    assert diag["eta_struct_max"] > 0.0
    assert np.isfinite(diag["eta_dyn_mean"]) and np.isfinite(diag["eta_S_mean"])
    assert abs(sum(diag[f"share_{c}"] for c in
                   ("pos", "vel", "att", "acc_bias", "gyro_bias")) - 1.0) < 1e-9
    # the dual layer's P_G here IS the exact position marginal, so the position
    # layer has not drifted at all -- only the STRUCTURE is lossy
    assert diag["pos_layer_dev_max"] < 1e-12
    info["eta_struct"] = diag["eta_struct_mean"]
    info["eta_dyn"] = diag["eta_dyn_mean"]
    info["eta_S"] = diag["eta_S_mean"]
    info["shares"] = {c: diag[f"share_{c}"]
                      for c in ("vel", "att", "acc_bias", "gyro_bias")}
    info["kappa"] = diag["kappa_mean"]
    info["kappa_curve"] = diag["kappa_full_curve"]

    # ---- resources and runtime ----
    ledger = CommunicationLedger(cfg, "dual_layer")
    for _ in range(10):
        ledger.add_epoch(measurement=len(edges))
    comm = ledger.summary()
    table = comparison_table(cfg, ("centralized", "dual_layer", "local_only"))
    assert (table["centralized"]["C_estimation_bytes"]
            > table["dual_layer"]["C_estimation_bytes"])
    info["comm"] = comm
    info["timing"] = profiler.summary()["sections"].get("predict", {})
    info["memory"] = profiler.summary()["memory"]
    info["repeat"] = repeat_timing(
        lambda: joint_covariance_probe(cfg, 1, epochs=1),
        repeats=int(cfg["runtime_repeats"]))
    info["fairness"] = fairness_report()
    return info


if __name__ == "__main__":
    print("== Task-2 unit tests (* = new or rewritten in Task 2) ==")
    failures = run_unit_tests()
    if failures:
        print("\nFAILURES:", failures)
        sys.exit(1)

    print("\n== cross-module composition ==")
    info = composition_check()
    print("  (a) truth/measurement separation error = %.1e m" % info["separation_error"])
    print("  (b) %d ranges per epoch (28 mesh + 12 class-derived anchors)"
          % info["edges"])
    print("  (c) dead reckoning: RMSE=%.3f m  RMTE=%.3f m  RMSE/RMTE=%.2f"
          % (info["rmse"], info["rmte"], info["ratio"]))
    print("      joint position ANEES=%.2f in [%.2f, %.2f] (dof 24); "
          "full-state ANEES=%.1f (dof 120)"
          % (info["anees"], info["anees_bounds"][0], info["anees_bounds"][1],
             info["anees_full"]))
    print("      per class: " + ", ".join(
        "%s=%.2f/%d" % (c, info["class_anees"][c], info["class_dof"][c])
        for c in ("A", "B", "C")))
    print("  (d) structural diagnostic on a real joint covariance:")
    print("      E_pp = H E H^T = 0 to machine precision, "
          "eta_struct=%.3f eta_dyn=%.3e eta_S=%.3e"
          % (info["eta_struct"], info["eta_dyn"], info["eta_S"]))
    print("      channel shares: " + ", ".join(
        "%s=%.2f" % (k, v) for k, v in info["shares"].items()))
    print("      kappa=%.3e vs analytic sqrt(2)/2 |f| T^2=%.3e"
          % (info["kappa"], info["kappa_curve"]))
    print("  resources: C_meas=%d B  C_est=%d B over %d epochs (dual layer)"
          % (info["comm"]["C_measurement_bytes"],
             info["comm"]["C_estimation_bytes"], info["comm"]["epochs"]))
    print("  runtime: %d predict sections, mean %.2f ms; array bytes %.1f kB, "
          "peak %.1f kB"
          % (info["timing"].get("calls", 0),
             1e3 * info["timing"].get("mean_s", float("nan")),
             info["memory"]["array_bytes_total"] / 1024.0,
             info["memory"]["tracemalloc_peak_bytes"] / 1024.0))
    print("  repeat timing of the covariance probe: %.1f +/- %.1f ms (%d runs)"
          % (1e3 * info["repeat"]["mean_s"], 1e3 * info["repeat"]["std_s"],
             info["repeat"]["repeats"]))
    print("\n[Task 2] L1 evaluation kernel + L2 simulation OK -- "
          "16/16 modules, composition verified.")
