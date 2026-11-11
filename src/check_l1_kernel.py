"""Task-1 gate: run every L1 kernel self-test, then check they COMPOSE.

The per-module ``__main__`` blocks test each module in isolation.  This script
additionally runs the chain that the rest of the project depends on:

    M5 truth  ->  M6 IMU corruption  ->  M4 mechanization  ->  M7 Phi/Q
              ->  M3 edges  ->  M8 ranges  ->  M9 lift

and asserts the three structural facts the dual layer is built on:

    (a) H L_i = I_3                       (M9)
    (b) the range Jacobian has no non-position column   (M8)
    (c) the accumulated coupling ||Phi_bar[P, TH]|| tracks 0.5 a_c T^2  (M7/M5)
"""

import subprocess
import sys

import numpy as np

MODULES = [
    "config.py", "l1_linalg.py", "l1_so3.py", "l1_geometry.py",
    "l1_mechanization.py", "l1_trajectory.py", "l1_true_bias.py",
    "l1_process_model.py", "l1_range_model.py", "l1_regression.py",
]


def run_unit_tests():
    failures = []
    for name in MODULES:
        proc = subprocess.run([sys.executable, name], capture_output=True, text=True)
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        status = "PASS" if proc.returncode == 0 else "FAIL"
        if proc.returncode != 0:
            failures.append((name, proc.stderr.strip().splitlines()[-1:]))
        print(f"  [{status}] {name:<24} {tail}")
    return failures


def composition_check():
    from config import (default_config, validate_config, centripetal_accel,
                        coupling_prediction, coupling_prediction_full,
                        POS_DIM, STATE_DIM, IDX_P, IDX_TH)
    from l1_geometry import build_edges
    from l1_mechanization import correct_imu, mechanize
    from l1_process_model import ProcessModel, coupling_norm, split_position_row
    from l1_range_model import build_H_state, simulate_ranges, predict_all
    from l1_regression import lift_matrix
    from l1_trajectory import CircleFormationTrajectory
    from l1_true_bias import TrueBiasProcess, corrupt_imu

    cfg = default_config(); validate_config(cfg)
    K, dt, T = cfg["num_entities"], cfg["dt_imu"], cfg["sync_interval"]
    n = cfg["imu_per_uwb"]
    rng = np.random.default_rng(cfg["seed_base"])

    traj = CircleFormationTrajectory(cfg)
    traj.reset(rng)
    bias = TrueBiasProcess(cfg); bias.reset(rng)
    pm = ProcessModel(cfg)

    states = [traj.initial_state(i) for i in range(K)]
    for i in range(K):
        states[i].b_a = bias.b_a[i].copy()
        states[i].b_g = bias.b_g[i].copy()

    # one full sync interval of IMU integration for entity 0
    samples = []
    t = 0.0
    st = states[0]
    for _ in range(n):
        _, w_b, f_b = traj.command(0, st.R, st.v, st.p, t)
        f_m, w_m = corrupt_imu(f_b, w_b, bias.b_a[0], bias.b_g[0],
                               cfg["sigma_acc"], cfg["sigma_gyro"], dt, rng)
        f_hat, w_hat = correct_imu(f_m, w_m, st.b_a, st.b_g)
        samples.append((st.R.copy(), f_hat, w_hat))
        st = mechanize(st, f_hat, w_hat, dt, cfg["gravity"],
                       cfg["nominal_int_order"])
        bias.step(dt)
        t += dt

    Phi_bar, Q_bar = pm.accumulate(samples)
    assert Phi_bar.shape == (STATE_DIM, STATE_DIM)
    assert np.min(np.linalg.eigvalsh(Q_bar)) > -1e-18

    # (c) the coupling scalar matches the FULL analytic curve
    #     sqrt(2) * 0.5 |a^n - g| T^2 to a few percent.  It does NOT match the
    #     plan's 0.5 a_c T^2, which drops gravity and is ~37x smaller here --
    #     see config.coupling_prediction_full.
    kappa = coupling_norm(Phi_bar)
    pred = coupling_prediction_full(cfg)
    pred_v_only = coupling_prediction(cfg)
    assert abs(kappa - pred) < 0.05 * pred, (kappa, pred)
    assert kappa > 5.0 * pred_v_only
    parts = split_position_row(Phi_bar)
    assert np.allclose(parts["att"], Phi_bar[IDX_P, IDX_TH])

    # drift after one interval must be small but non-zero (the IMU is noisy)
    truth_p = traj.position(0, t)
    drift = float(np.linalg.norm(st.p - truth_p))
    assert 0.0 < drift < 0.5, drift

    # (b) the range Jacobian is supported on positions only
    positions = [traj.position(i, t) for i in range(K)]
    edges = build_edges(cfg, positions)
    Hs = build_H_state(edges, positions, cfg["anchor_positions"], K)
    mask = np.ones(STATE_DIM * K, dtype=bool)
    for i in range(K):
        mask[i * STATE_DIM:i * STATE_DIM + POS_DIM] = False
    assert np.all(Hs[:, mask] == 0.0)

    z, _ = simulate_ranges(edges, positions, cfg["anchor_positions"],
                           cfg["sigma_range"], rng)
    resid = z - predict_all(edges, positions, cfg["anchor_positions"])
    assert float(np.std(resid)) < 4.0 * cfg["sigma_range"]

    # (a) H L = I for a realistic propagated covariance
    P0 = np.diag(np.concatenate([
        np.full(3, cfg["init_pos_std"] ** 2), np.full(3, cfg["init_vel_std"] ** 2),
        np.full(3, cfg["init_att_std"] ** 2),
        np.full(3, cfg["init_acc_bias_std"] ** 2),
        np.full(3, cfg["init_gyro_bias_std"] ** 2)]))
    P_minus = Phi_bar @ P0 @ Phi_bar.T + Q_bar
    L = lift_matrix(P_minus, cfg["reg_epsilon"])
    assert np.allclose(L[IDX_P, :], np.eye(POS_DIM))

    return {"kappa": kappa, "kappa_pred": pred, "kappa_pred_v": pred_v_only,
            "drift": drift,
            "a_c": centripetal_accel(cfg), "edges": len(edges),
            "sigma_hat": float(np.std(resid))}


if __name__ == "__main__":
    print("== L1 kernel unit tests ==")
    failures = run_unit_tests()
    if failures:
        print("\nFAILURES:", failures)
        sys.exit(1)
    print("\n== cross-module composition ==")
    info = composition_check()
    print("  a_c = %.3f m/s^2, kappa = %.4e" % (info["a_c"], info["kappa"]))
    print("    full analytic sqrt(2)*0.5|a-g|T^2 = %.4e  (ratio %.3f)"
          % (info["kappa_pred"], info["kappa"] / info["kappa_pred"]))
    print("    plan curve 0.5 a_c T^2            = %.4e  (ratio %.1f)  <-- "
          "omits gravity" % (info["kappa_pred_v"],
                             info["kappa"] / info["kappa_pred_v"]))
    print("  one-interval INS drift = %.4f m, %d range edges, sigma_hat = %.3f m"
          % (info["drift"], info["edges"], info["sigma_hat"]))
    print("\n[Task 1] L1 kernel OK -- 10/10 modules, composition verified.")
