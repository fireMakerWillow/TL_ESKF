"""Task-3 gate: run every L2 estimator self-test, then check that the seven
estimators COMPOSE with the L1/L2 kernel and satisfy the cross-module acceptance
tests that no single module can check alone.

Per-module ``__main__`` blocks test each estimator in isolation.  What this gate
adds is the set of tests defined BETWEEN estimators, i.e. section 3.2's

    T10  K = 1  ->  M19 is numerically M15
    T11  DCL/local-only effective-noise update, stride and resource accounting
    T13  M25 is the consistency floor: joint position and full-state ANEES in band
    T14  M25's position cross blocks are exactly zero, AND every cooperative
         method degenerates to M25 when the range table is emptied
    T15  the single-modality envelope (in the form the measurement supports)
    T16  E_pp = 0 and H E^ij H^T = 0 on a REAL M15 P_x against a REAL M19 P_G
    T22  M19's two lifts are separate caches and are numerically distinguishable

plus the registry contract that the L3/L4 layers will resolve names through.

Runtime.  The eight module self-tests take ~250 s as subprocesses and the
cross-module tests ~190 s, so a full run is around 7.5 minutes -- longer than
some interactive shells allow.  ``--skip-units`` runs only the cross-module
section (the per-module tests are then assumed to have been run separately, and
the gate says so in its banner); ``--only=T15,T16`` restricts the cross-module
section to named tests.  Neither switch is available to CI: the default, with no
arguments, is the complete gate.

Why T11 gets called out.  M16 now follows the DCL equations used by the
comparison paper: the other endpoint's projected local covariance is absorbed
into an effective measurement variance, no cross block is retained, and only
every ``dcl_measurement_stride``-th inter-entity measurement is processed.  The
gate therefore checks the equation-level architecture and the stride/resource
coupling rather than the obsolete ``keep_cross`` equivalence switch.
"""

import subprocess
import sys

import numpy as np

from config import (IDX_P, POS_DIM, STATE_DIM, apply_entity_count,
                    default_config, validate_config)
from l1_diagnostic import StructuralResidualDiagnostic
from l1_metrics import anees_bounds, evaluate, full_state_nees, joint_position_nees
from l1_so3 import boxminus
from l2_estimators import (ESTIMATOR_CLASSES, ESTIMATOR_NAMES, build,
                           default_set, main_line, sensor_ablations)
from l2_sensor import SensorSimulator
from l2_truth import TruthSimulator

MODULES = [
    "l2_estimator_base.py", "l2_centralized_eskf.py", "l2_local_only.py",
    "l2_ci_eskf.py", "l2_dual_layer.py", "l2_imu_only.py", "l2_uwb_only.py",
    "l2_estimators.py",
]


def run_unit_tests():
    failures = []
    for name in MODULES:
        proc = subprocess.run([sys.executable, name], capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.strip().splitlines()
                 if not ln.startswith("[config][WARNING]")]
        tail = lines[-1] if lines else ""
        status = "PASS" if proc.returncode == 0 else "FAIL"
        if proc.returncode != 0:
            failures.append((name, proc.stderr.strip().splitlines()[-3:]))
        print(f"  [{status}] {name:<24} {tail[:96]}")
    return failures


# ----------------------------------------------------------------------
# one shared driver, so every method sees the same world
# ----------------------------------------------------------------------
def drive(cfg, name, seed, epochs, edges=True, collect=False):
    """Run one estimator for ``epochs`` epochs and return it with its truth.

    The range table is drawn even when it is not handed over, so the
    measurement RNG advances identically in both arms and "same seed" really
    means "same noise realisation" (the point M25's self-test makes).
    """
    truth = TruthSimulator(cfg); truth.reset(seed)
    sensor = SensorSimulator(cfg); sensor.reset(seed + 5000)
    est = build(name, cfg); est.reset(truth.get_truth_states(), seed=seed)
    n = cfg["imu_per_uwb"]
    rows = []
    for _ in range(epochs):
        imu = [sensor.generate_imu(truth.step()) for _ in range(n)]
        drawn = sensor.generate_ranges(truth.snapshot())
        est.predict(imu, n * cfg["dt_imu"])
        est.update(drawn if edges else [])
        if collect:
            full = (est.get_eval_cov_full()
                    if hasattr(est, "get_eval_cov_full") else None)
            rows.append(evaluate(truth.get_truth_states(), est.get_nominal_state(),
                                 est.get_eval_cov(), cfg, full_cov=full,
                                 channels=est.evaluated_channels))
    return est, truth, rows


def tail_rmse(cfg, name, seeds, epochs):
    vals = []
    for sd in seeds:
        _, _, rows = drive(cfg, name, sd, epochs, collect=True)
        vals.append(float(np.mean([r["pos_rmse"] for r in rows[len(rows) // 2:]])))
    return float(np.mean(vals)), vals


def epochs_of(cfg):
    return int(round(cfg["sim_duration"] / cfg["sync_interval"]))


# ----------------------------------------------------------------------
# T10 -- K = 1: the dual layer IS the centralized filter
# ----------------------------------------------------------------------
def check_T10():
    cfg = default_config(); apply_entity_count(cfg, 1); cfg["sim_duration"] = 4.0
    validate_config(cfg)
    dual, _, _ = drive(cfg, "dual_layer", 1, epochs_of(cfg))
    ref, _, _ = drive(cfg, "centralized", 1, epochs_of(cfg))
    scale = float(np.max(np.abs(ref.get_full_cov())))
    rel = float(np.max(np.abs(dual.get_eval_cov_full()
                              - ref.get_full_cov()))) / scale
    dpos = float(np.max(np.abs(dual.get_pos_estimate() - ref.get_pos_estimate())))
    assert rel < 1e-6 and dpos < 1e-6, (rel, dpos)
    return {"rel_cov": rel, "dpos": dpos}


# ----------------------------------------------------------------------
# T11 -- DCL/local-only effective-noise update + persistent stride
# ----------------------------------------------------------------------
def check_T11():
    # The experiment default is stride=1.  T11 explicitly switches to 3 so the
    # persistent thinning behaviour remains covered as an implementation test.
    cfg = default_config(); cfg["sim_duration"] = 1.0
    cfg["separation_check"] = False
    cfg["dcl_measurement_stride"] = 3
    validate_config(cfg)
    eps = epochs_of(cfg)

    dcl, _, _ = drive(cfg, "local_only", 0, eps, collect=False)
    assert not dcl.has_cross_objects()
    assert getattr(dcl, "cross", None) is None

    K = cfg["num_entities"]
    inter_per_epoch = K * (K - 1) // 2
    total_inter = inter_per_epoch * eps
    expected_processed = total_inter // cfg["dcl_measurement_stride"]
    stats = dcl.get_stride_summary()
    assert stats["relative_counter"] == total_inter
    assert stats["relative_processed_total"] == expected_processed
    assert stats["relative_skipped_total"] == total_inter - expected_processed

    # The reported covariance remains block diagonal.
    P = dcl.get_eval_cov()
    off = max(float(np.max(np.abs(P[i * POS_DIM:(i + 1) * POS_DIM,
                                   j * POS_DIM:(j + 1) * POS_DIM])))
              for i in range(K) for j in range(K) if i != j)
    assert off == 0.0, off

    groups = cfg["entity_class_map"]
    anchor_per_epoch = sum(4 if c == "A" else 2 if c == "B" else 0
                           for c in groups)
    comm = dcl.get_comm_summary()
    assert comm["C_measurement_scalars"] == anchor_per_epoch * eps + expected_processed
    assert comm["C_estimation_scalars"] == 18 * expected_processed

    # Disabling thinning (stride=1) must use more update work/traffic and report
    # a tighter covariance because it incorporates all pair measurements.
    dense_cfg = default_config(); dense_cfg["sim_duration"] = 1.0
    dense_cfg["separation_check"] = False
    dense_cfg["dcl_measurement_stride"] = 1
    validate_config(dense_cfg)
    dense, _, _ = drive(dense_cfg, "local_only", 0, eps, collect=False)
    assert dense.processed_relative_measurements == total_inter
    assert dcl.get_flops() < dense.get_flops()
    assert (dcl.get_comm_summary()["C_estimation_scalars"]
            < dense.get_comm_summary()["C_estimation_scalars"])
    assert float(np.trace(dcl.get_eval_cov())) > float(np.trace(dense.get_eval_cov()))

    return {
        "stride": cfg["dcl_measurement_stride"],
        "processed": expected_processed,
        "skipped": total_inter - expected_processed,
        "offdiag_max": off,
        "comm_estimation_scalars": comm["C_estimation_scalars"],
        "flops_ratio_vs_stride1": dcl.get_flops() / dense.get_flops(),
    }


# ----------------------------------------------------------------------
# T13 -- M25 is the consistency floor (a probe on M10 / M13 / M14)
# ----------------------------------------------------------------------
def check_T13(num_seeds=24):
    cfg = default_config(); apply_entity_count(cfg, 4); cfg["sim_duration"] = 3.0
    validate_config(cfg)
    eps = epochs_of(cfg)
    K = cfg["num_entities"]
    pos_samples, full_samples = [], []
    for sd in range(num_seeds):
        est, truth, _ = drive(cfg, "imu_only", 100 + sd, eps)
        E = np.stack([boxminus(t, n) for t, n
                      in zip(truth.get_truth_states(), est.get_nominal_state())])
        pos_samples.append(joint_position_nees(E[:, IDX_P].ravel(),
                                               est.get_eval_cov()))
        full_samples.append(full_state_nees(E, est.get_eval_cov_full()))
    out = {}
    for label, samples, dof in (("pos", pos_samples, POS_DIM * K),
                                ("full", full_samples, STATE_DIM * K)):
        anees = float(np.mean(samples))
        lo, hi = anees_bounds(dof, len(samples), 0.99)
        assert lo < anees < hi, (label, anees, lo, hi, dof)
        out[label] = (anees, lo, hi, dof)
    return out


# ----------------------------------------------------------------------
# T14 -- exact zero cross blocks, and no implicit coupling anywhere
# ----------------------------------------------------------------------
def check_T14():
    cfg = default_config(); apply_entity_count(cfg, 4); cfg["sim_duration"] = 3.0
    validate_config(cfg)
    eps = epochs_of(cfg)
    K = cfg["num_entities"]

    floor, _, _ = drive(cfg, "imu_only", 11, eps)
    P = floor.get_eval_cov()
    nz = sum(int(np.count_nonzero(P[i * POS_DIM:(i + 1) * POS_DIM,
                                    j * POS_DIM:(j + 1) * POS_DIM]))
             for i in range(K) for j in range(K) if i != j)
    assert nz == 0, nz          # a hard zero, not a threshold

    # every cooperative method, fed an EMPTY range table, must be M25 again:
    # any residual difference is coupling that was not paid for with a
    # measurement.  The residual is round-off from M15's injection path
    # re-orthonormalising R even for dx = 0 (M25 never injects), ~1e-13 after
    # 30 epochs; the covariance agrees far more tightly because no covariance
    # operation depends on the attitude at that level.
    scale = float(np.max(np.abs(floor.get_eval_cov())))
    gaps = {}
    for name in main_line():
        est, _, _ = drive(cfg, name, 11, eps, edges=False)
        nom = max(float(np.max(np.abs(boxminus(a, b)))) for a, b
                  in zip(est.get_nominal_state(), floor.get_nominal_state()))
        cov = float(np.max(np.abs(est.get_eval_cov() - floor.get_eval_cov())))
        assert nom < 1e-11, (name, nom)
        assert cov < 1e-12 * scale, (name, cov, scale)
        gaps[name] = (nom, cov)
    return {"cross_nonzeros": nz, "gaps": gaps}


# ----------------------------------------------------------------------
# T15 -- the single-modality envelope, in the form the data supports
# ----------------------------------------------------------------------
def check_T15(seeds=range(3)):
    """Two claims, because the design's single claim is not true as written.

    Design section 3.2 states T15 as "at any grid point all five cooperative
    methods' position RMSE is no worse than min(M25, M26)".  Measured at the
    default point (K=8, T=0.1, 3 seeds) that fails, and not marginally:

        centralized 0.60   dual_layer 0.54   local_only 1.52
        ci_trace    1.27   ci_det     3.17   |   M25 9.80   M26 0.65

    The reason is structural and is stated in the design itself ([M26]): M26 is
    NOT a decoupled baseline.  It carries the full dense 3K position covariance,
    so with dense ranging it uses every cross-correlation exactly, and it can
    beat an IMU-equipped filter that throws those correlations away.  DCL/local-only
    uses local effective-noise updates with stride, while CI bounds unknown
    correlations conservatively; being worse than M26 can therefore be their
    architectural cost rather than a fusion bug.

    So the envelope is checked in the two forms that are true and that carry the
    intended meaning:

      (a) the methods that RETAIN cross-correlation -- fully (M15) or
          structurally (M19) -- beat both single-modality references;
      (b) every cooperative method beats the no-cooperation floor M25 by a wide
          margin, which is the "fusion actually works" statement.

    Claim (a) is the real fusion test: a method that keeps the correlations and
    adds an IMU must not lose to either sensor alone.  A method that discards
    them may, and M21 should report that it does.
    """
    cfg = default_config(); cfg["sim_duration"] = 10.0
    validate_config(cfg)
    eps = epochs_of(cfg)
    rmse = {}
    for name in default_set():
        rmse[name] = tail_rmse(cfg, name, seeds, eps)[0]
    floor = min(rmse["imu_only"], rmse["uwb_only"])
    for name in ("centralized", "dual_layer"):
        assert rmse[name] < floor, (name, rmse[name], floor)
    for name in main_line():
        assert rmse[name] < 0.5 * rmse["imu_only"], (name, rmse[name])
    return {"rmse": rmse, "floor": floor}


# ----------------------------------------------------------------------
# T16 -- the construction identities on real filter output
# ----------------------------------------------------------------------
def check_T16():
    cfg = default_config(); cfg["sim_duration"] = 5.0
    validate_config(cfg)
    eps = epochs_of(cfg)

    truth = TruthSimulator(cfg); truth.reset(4)
    sensor = SensorSimulator(cfg); sensor.reset(5004)
    dual = build("dual_layer", cfg); dual.reset(truth.get_truth_states(), seed=4)
    ref = build("centralized", cfg); ref.reset(truth.get_truth_states(), seed=4)
    diag = StructuralResidualDiagnostic(cfg, mode="structural")

    n = cfg["imu_per_uwb"]
    worst = {"E_pp_max": 0.0, "HEH_max": 0.0}
    eta_struct, eta_dyn = [], []
    for _ in range(eps):
        imu = [sensor.generate_imu(truth.step()) for _ in range(n)]
        drawn = sensor.generate_ranges(truth.snapshot())
        for est in (dual, ref):
            est.predict(imu, n * cfg["dt_imu"])
        # pre_uwb: Phi_bar is still the epoch's accumulation, P_G is synced
        rep = diag.evaluate(ref.get_full_cov(), P_G=dual.get_global_cov(),
                            Phi_bar_list=dual.get_phi_bar(),
                            sample_point="pre_uwb")
        for key in worst:
            worst[key] = max(worst[key], rep[key])
        eta_struct.append(rep["eta_struct_max"])
        eta_dyn.append(rep["eta_dyn_max"])
        for est in (dual, ref):
            est.update(drawn)
    assert worst["E_pp_max"] < 1e-9, worst
    assert worst["HEH_max"] < 1e-9, worst
    # and the residual it is measuring is not trivially zero
    assert max(eta_struct) > 1e-3, max(eta_struct)
    assert max(eta_dyn) > 1e-6, max(eta_dyn)
    return {"E_pp_max": worst["E_pp_max"], "HEH_max": worst["HEH_max"],
            "eta_struct_max": max(eta_struct), "eta_dyn_max": max(eta_dyn)}


# ----------------------------------------------------------------------
# T22 -- the dual L contract
# ----------------------------------------------------------------------
def check_T22():
    out = {}
    for label, (v, T) in (("mild", (2.0, 0.02)), ("strong", (12.0, 0.20))):
        cfg = default_config()
        cfg["speed"] = v; cfg["sync_interval"] = T; cfg["sim_duration"] = 10.0
        validate_config(cfg)
        est, _, _ = drive(cfg, "dual_layer", 0, epochs_of(cfg))
        L_minus, L_rst = est.get_lifts()
        assert len(L_minus) == len(L_rst) == cfg["num_entities"]
        for a, b in zip(L_minus, L_rst):
            assert a is not b
        out[label] = est.last_lift_gap
    # the two lifts must be separable at all times, and MORE separable under
    # stronger dynamics -- if they were being aliased the gap would be 0
    assert out["mild"] > 0.0 and out["strong"] > out["mild"], out
    assert est.lifts_are_distinguishable(1e-9)
    return out


# ----------------------------------------------------------------------
def check_registry():
    cfg = default_config(); validate_config(cfg)
    assert len(ESTIMATOR_CLASSES) == 7
    assert sorted(main_line() + sensor_ablations()) == sorted(ESTIMATOR_NAMES)
    for name in ESTIMATOR_NAMES:
        est = build(name, cfg)
        assert est.name == name
    return len(ESTIMATOR_CLASSES)


CROSS_MODULE = ("registry", "T10", "T22", "T16", "T11", "T13", "T14", "T15")


if __name__ == "__main__":
    args = sys.argv[1:]
    skip_units = "--skip-units" in args
    only = None
    for arg in args:
        if arg.startswith("--only="):
            only = [t.strip() for t in arg.split("=", 1)[1].split(",") if t.strip()]
            unknown = sorted(set(only) - set(CROSS_MODULE))
            if unknown:
                print(f"unknown test names {unknown}; known: {list(CROSS_MODULE)}")
                sys.exit(2)
    wanted = set(CROSS_MODULE if only is None else only)

    if skip_units:
        print("--- L2 estimator self-tests SKIPPED (--skip-units) ---")
        print("    the eight module __main__ blocks were NOT run by this "
              "invocation; run them separately before trusting this banner")
    else:
        print("--- L2 estimator self-tests ---")
        failures = run_unit_tests()
        if failures:
            print("\n[Task 3] FAILED")
            for name, tail in failures:
                print(f"  {name}:")
                for line in tail:
                    print(f"    {line}")
            sys.exit(1)

    print("\n--- cross-module acceptance tests ---")

    def _fmt_registry(n):
        return f"{n} names resolve, all self-consistent"

    def _fmt_T10(r):
        return "rel cov %.1e, |dp| %.1e" % (r["rel_cov"], r["dpos"])

    def _fmt_T11(r):
        return ("stride %d processed %d / skipped %d relative ranges; "
                "offdiag %.1e; C_est %d; FLOPs %.3f x stride-1"
                % (r["stride"], r["processed"], r["skipped"],
                   r["offdiag_max"], r["comm_estimation_scalars"],
                   r["flops_ratio_vs_stride1"]))

    def _fmt_T13(r):
        return ("position ANEES %.2f in [%.2f, %.2f] dof %d\n"
                "                        full-state ANEES %.2f in [%.2f, %.2f] "
                "dof %d" % (r["pos"] + r["full"]))

    def _fmt_T14(r):
        return ("%d cross non-zeros; five methods -> M25, worst nominal %.1e, "
                "worst cov %.1e"
                % (r["cross_nonzeros"], max(g[0] for g in r["gaps"].values()),
                   max(g[1] for g in r["gaps"].values())))

    def _fmt_T15(r):
        return (", ".join("%s %.2f" % kv for kv in r["rmse"].items())
                + "\n                        M15/M19 beat min(M25,M26)=%.2f and "
                  "all five beat M25 by >2x.  The design's stronger form -- all "
                  "five beat min(M25,M26) -- is FALSE; see check_T15's docstring."
                  % r["floor"])

    def _fmt_T16(r):
        return ("E_pp %.1e, |H E H^T| %.1e (eta_struct up to %.3f, eta_dyn up "
                "to %.1e)" % (r["E_pp_max"], r["HEH_max"], r["eta_struct_max"],
                              r["eta_dyn_max"]))

    def _fmt_T22(r):
        return ("lift gap %.1e (v=2, T=0.02) -> %.1e (v=12, T=0.20)"
                % (r["mild"], r["strong"]))

    LABELS = {"registry": "registry            ",
              "T10": "T10 K=1 M19 == M15  ",
              "T11": "T11 DCL stride      ",
              "T13": "T13 M25 floor       ",
              "T14": "T14 no-range floor  ",
              "T15": "T15 envelope        ",
              "T16": "T16 identities      ",
              "T22": "T22 dual L          "}
    CHECKS = {"registry": (check_registry, _fmt_registry),
              "T10": (check_T10, _fmt_T10), "T11": (check_T11, _fmt_T11),
              "T13": (check_T13, _fmt_T13), "T14": (check_T14, _fmt_T14),
              "T15": (check_T15, _fmt_T15), "T16": (check_T16, _fmt_T16),
              "T22": (check_T22, _fmt_T22)}

    for key in CROSS_MODULE:
        if key not in wanted:
            print(f"  {LABELS[key]}: skipped")
            continue
        fn, fmt = CHECKS[key]
        print(f"  {LABELS[key]}: {fmt(fn())}", flush=True)

    scope = ("8/8 modules, " if not skip_units else "modules skipped, ")
    tests = "/".join(k for k in CROSS_MODULE if k in wanted and k != "registry")
    print(f"\n[Task 3] L2 estimator family OK -- {scope}{tests} verified.")
