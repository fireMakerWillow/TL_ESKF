"""M4 Mechanization (L1 basic capability) -- nominal strapdown integration.

Propagates the NOMINAL state with the bias-corrected IMU sample:

    a^n   = R f^b + g                      (specific force -> navigation accel)
    p <- p + v dt + 0.5 a^n dt^2
    v <- v + a^n dt
    R <- R Exp(omega^b dt)                 (right / body convention of M2)

Biases are constant across one nominal step; their random walk lives in Q (M7)
and in the TRUTH generator (M6) separately.

``order``:
    '1'        -- evaluate a^n with the attitude at the START of the step;
    'midpoint' -- average the accelerations at the start and end attitudes.
The midpoint form removes the O(omega dt) attitude lag and is what makes the M5
closure test tight at large omega; '1' is retained because it is the form the
error-state Jacobian of M7 is derived from, so the pair (nominal_int_order,
phi_order) must be reported together.

No noise is ever injected here: M4 is deterministic given its inputs.
"""

import numpy as np

from l1_so3 import Exp, NominalState, project_so3


def specific_force_to_accel(R, f_b, gravity):
    """a^n = R f^b + g -- the single definition used everywhere."""
    return np.asarray(R, dtype=float) @ np.asarray(f_b, dtype=float).ravel() \
        + np.asarray(gravity, dtype=float).ravel()


def correct_imu(f_b_meas, w_b_meas, b_a, b_g):
    """Subtract the current bias estimates from the raw IMU sample."""
    return (np.asarray(f_b_meas, dtype=float).ravel() - np.asarray(b_a, dtype=float).ravel(),
            np.asarray(w_b_meas, dtype=float).ravel() - np.asarray(b_g, dtype=float).ravel())


def mechanize(state, f_b, w_b, dt, gravity, order="1"):
    """Advance the nominal state by ``dt`` with a bias-CORRECTED IMU sample.

    Parameters
    ----------
    state : NominalState        (p, v, R) at the start of the step
    f_b   : (3,) specific force in the body frame, bias already removed
    w_b   : (3,) angular rate in the body frame, bias already removed
    dt    : step length (s)
    gravity : (3,) navigation-frame gravity vector
    order : '1' or 'midpoint'
    """
    dt = float(dt)
    f_b = np.asarray(f_b, dtype=float).ravel()
    w_b = np.asarray(w_b, dtype=float).ravel()
    g = np.asarray(gravity, dtype=float).ravel()

    R0 = state.R
    dR = Exp(w_b * dt)
    R1 = project_so3(R0 @ dR)

    if order == "1":
        a_n = specific_force_to_accel(R0, f_b, g)
    elif order == "midpoint":
        a_n = 0.5 * (specific_force_to_accel(R0, f_b, g)
                     + specific_force_to_accel(R1, f_b, g))
    else:
        raise ValueError(f"unknown nominal_int_order {order!r}")

    out = state.copy()
    out.p = state.p + state.v * dt + 0.5 * a_n * dt ** 2
    out.v = state.v + a_n * dt
    out.R = R1
    return out


def mechanize_sequence(state, f_seq, w_seq, dt, gravity, order="1"):
    """Integrate a whole IMU burst; returns the final state and the a^n list."""
    accels = []
    cur = state
    for f_b, w_b in zip(f_seq, w_seq):
        accels.append(specific_force_to_accel(cur.R, f_b, gravity))
        cur = mechanize(cur, f_b, w_b, dt, gravity, order)
    return cur, accels


if __name__ == "__main__":
    from config import default_config
    from l1_so3 import Log

    cfg = default_config()
    g = cfg["gravity"]
    dt = cfg["dt_imu"]

    # ---- 1) free fall: f^b = 0 gives exactly the analytic parabola ----
    st = NominalState(p=np.zeros(3), v=np.zeros(3), R=np.eye(3))
    steps = 200
    cur = st
    for _ in range(steps):
        cur = mechanize(cur, np.zeros(3), np.zeros(3), dt, g)
    t = steps * dt
    assert np.allclose(cur.v, g * t, atol=1e-12)
    assert np.allclose(cur.p, 0.5 * g * t ** 2, atol=1e-9)

    # ---- 2) stationary: f^b = -R^T g holds the state exactly still ----
    R0 = Exp(np.array([0.05, -0.02, 0.3]))
    st = NominalState(p=np.array([1.0, 2.0, 3.0]), v=np.zeros(3), R=R0)
    cur = st
    for _ in range(500):
        f_b = cur.R.T @ (-g)
        cur = mechanize(cur, f_b, np.zeros(3), dt, g)
    stationary_drift = float(np.linalg.norm(cur.p - st.p))
    assert stationary_drift < 1e-9
    assert np.allclose(cur.v, 0.0, atol=1e-9)
    assert np.allclose(cur.R, R0, atol=1e-12)

    # ---- 3) pure rotation integrates to the exact total angle ----
    rate = np.array([0.0, 0.0, 1.7])
    cur = NominalState(R=np.eye(3))
    n = 300                      # total angle 2.55 rad < pi, so Log does not wrap
    for _ in range(n):
        cur = mechanize(cur, cur.R.T @ (-g), rate, dt, g)
    assert abs(np.linalg.norm(Log(cur.R)) - abs(rate[2]) * n * dt) < 1e-9
    assert np.allclose(cur.R.T @ cur.R, np.eye(3), atol=1e-12)

    # ---- 4) a^n = R f^b + g is exactly inverted by f^b = R^T (a^n - g) ----
    a_target = np.array([0.3, -1.2, 0.4])
    R = Exp(np.array([0.2, 0.1, -0.4]))
    f_b = R.T @ (a_target - g)
    assert np.allclose(specific_force_to_accel(R, f_b, g), a_target, atol=1e-12)

    # ---- 5) midpoint beats first order on a fast-rotating constant f^b ----
    rate = np.array([0.0, 0.0, 3.0])
    f_b = np.array([0.0, 2.0, 9.81])
    base = NominalState(R=np.eye(3))
    coarse, fine = 0.02, 0.0005
    n_c, n_f = 100, 4000
    ref = base
    for _ in range(n_f):
        ref = mechanize(ref, f_b, rate, fine, g, "midpoint")
    err = {}
    for mode in ("1", "midpoint"):
        cur = base
        for _ in range(n_c):
            cur = mechanize(cur, f_b, rate, coarse, g, mode)
        err[mode] = float(np.linalg.norm(cur.p - ref.p))
    assert err["midpoint"] < err["1"], err

    # ---- 6) bias correction ----
    fc, wc = correct_imu(np.ones(3), np.ones(3), 0.2 * np.ones(3), 0.1 * np.ones(3))
    assert np.allclose(fc, 0.8) and np.allclose(wc, 0.9)

    # ---- 7) sequence helper matches the step loop ----
    fs = [f_b] * 10; ws = [rate] * 10
    seq, accels = mechanize_sequence(base, fs, ws, dt, g)
    cur = base
    for a, b in zip(fs, ws):
        cur = mechanize(cur, a, b, dt, g)
    assert np.allclose(seq.p, cur.p) and len(accels) == 10

    print("[M4 Mechanization] OK  (free fall exact, stationary drift %.1e m, "
          "midpoint error %.2e < first order %.2e)"
          % (stationary_drift, err["midpoint"], err["1"]))
