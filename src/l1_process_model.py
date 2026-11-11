"""M7 ProcessModel (L1 basic capability) -- error-state dynamics.

Continuous error dynamics in the RIGHT (body) attitude convention of M2,
state order [dp, dv, dtheta, dba, dbg]:

    dp'      = dv
    dv'      = -R_hat [f_hat]_x dtheta - R_hat dba - R_hat n_a
    dtheta'  = -[w_hat]_x dtheta - dbg - n_g
    dba'     = n_ba
    dbg'     = n_bg

with f_hat = f_meas - b_hat_a and w_hat = w_meas - b_hat_g.  The only non-zero
blocks of F are therefore

    F[P, V] = I,  F[V, TH] = -R_hat [f_hat]_x,  F[V, BA] = -R_hat,
    F[TH, TH] = -[w_hat]_x,  F[TH, BG] = -I.

Discretisation (``config['phi_order']``):
    '1'        Phi = I + F dt                       (matches nominal order '1')
    '2'        Phi = I + F dt + 0.5 F^2 dt^2        (recovers the 0.5 a dt^2
                                                     position-attitude coupling
                                                     that the mechanization has)
    'van_loan' exact Phi and Q of the LTI system over the step.

Why the position row matters.  The whole v3 story is that the dual layer keeps
only the position block, so the error it makes is exactly the part of the
propagated cross-covariance that flows through the NON-position states.  Writing
the accumulated transition as

    H Phi_bar = [F_p | F_z] = [F_p | F_vel  F_att  F_ba  F_bg],

:func:`split_position_row` hands M12 those five 3x3 blocks.  Over one sync
interval the dominant off-diagonal entry is

    Phi_bar[P, TH] ~ -0.5 R [f]_x T^2,      ||.|| ~ 0.5 a_c T^2,

which is precisely the analytic coupling curve of the M22 sweep.

The filter's Q is built from ``filter_sigma_*`` and NEVER from the truth
parameters of M6 (separation contract).
"""

import numpy as np
from scipy.linalg import expm

from config import (STATE_DIM, POS_DIM, Z_CHANNELS, Z_CHANNEL_SLICES,
                    IDX_P, IDX_V, IDX_TH, IDX_BA, IDX_BG)
from l1_linalg import symmetrize
from l1_so3 import skew

_H = np.zeros((POS_DIM, STATE_DIM))
_H[:, IDX_P] = np.eye(POS_DIM)


def position_selector():
    """Return the 3x15 selector H with H x = dp."""
    return _H.copy()


def error_dynamics_F(R_hat, f_hat, w_hat):
    """Continuous error-state matrix F (15x15)."""
    R_hat = np.asarray(R_hat, dtype=float).reshape(3, 3)
    F = np.zeros((STATE_DIM, STATE_DIM))
    F[IDX_P, IDX_V] = np.eye(3)
    F[IDX_V, IDX_TH] = -R_hat @ skew(f_hat)
    F[IDX_V, IDX_BA] = -R_hat
    F[IDX_TH, IDX_TH] = -skew(w_hat)
    F[IDX_TH, IDX_BG] = -np.eye(3)
    return F


def noise_gain_G(R_hat):
    """Continuous noise gain G (15x12) for [n_a, n_g, n_ba, n_bg]."""
    R_hat = np.asarray(R_hat, dtype=float).reshape(3, 3)
    G = np.zeros((STATE_DIM, 12))
    G[IDX_V, 0:3] = -R_hat
    G[IDX_TH, 3:6] = -np.eye(3)
    G[IDX_BA, 6:9] = np.eye(3)
    G[IDX_BG, 9:12] = np.eye(3)
    return G


def continuous_Q(config):
    """Continuous-time PSD of the driving noise (12x12), FILTER parameters."""
    q = np.zeros(12)
    q[0:3] = float(config["sigma_acc"]) ** 2
    q[3:6] = float(config["sigma_gyro"]) ** 2
    q[6:9] = float(config["filter_sigma_acc_bias"]) ** 2
    q[9:12] = float(config["filter_sigma_gyro_bias"]) ** 2
    return np.diag(q)


def transition_matrix(R_hat, f_hat, w_hat, dt, order="1"):
    """Discrete error-state transition Phi over one IMU step."""
    F = error_dynamics_F(R_hat, f_hat, w_hat)
    dt = float(dt)
    if order == "1":
        return np.eye(STATE_DIM) + F * dt
    if order == "2":
        return np.eye(STATE_DIM) + F * dt + 0.5 * (F @ F) * dt ** 2
    if order == "van_loan":
        return expm(F * dt)
    raise ValueError(f"unknown phi_order {order!r}")


def process_noise(R_hat, dt, config, order=None):
    """Discrete process-noise covariance Q over one IMU step (15x15)."""
    dt = float(dt)
    order = config["phi_order"] if order is None else order
    G = noise_gain_G(R_hat)
    Qc = continuous_Q(config)
    GQG = G @ Qc @ G.T
    if order != "van_loan":
        return symmetrize(GQG * dt)
    F = np.zeros((STATE_DIM, STATE_DIM))          # placeholder, filled by caller
    raise ValueError("van_loan Q must be built by van_loan_step")


def van_loan_step(R_hat, f_hat, w_hat, dt, config):
    """Exact (Phi, Q) of the LTI system over one step (Van Loan's method)."""
    dt = float(dt)
    F = error_dynamics_F(R_hat, f_hat, w_hat)
    G = noise_gain_G(R_hat)
    GQG = G @ continuous_Q(config) @ G.T
    n = STATE_DIM
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -F
    M[:n, n:] = GQG
    M[n:, n:] = F.T
    E = expm(M * dt)
    Phi = E[n:, n:].T
    Q = symmetrize(Phi @ E[:n, n:])
    return Phi, Q


class ProcessModel:
    """Stateless helper bundling Phi, Q and the burst accumulation."""

    def __init__(self, config):
        self.cfg = config
        self.order = config["phi_order"]
        self.dt = float(config["dt_imu"])

    def step(self, R_hat, f_hat, w_hat, dt=None):
        """Return (Phi, Q) for one IMU step."""
        dt = self.dt if dt is None else float(dt)
        if self.order == "van_loan":
            return van_loan_step(R_hat, f_hat, w_hat, dt, self.cfg)
        Phi = transition_matrix(R_hat, f_hat, w_hat, dt, self.order)
        Q = process_noise(R_hat, dt, self.cfg, self.order)
        return Phi, Q

    def propagate(self, P, R_hat, f_hat, w_hat, dt=None):
        """One covariance propagation step; returns (P', Phi, Q)."""
        Phi, Q = self.step(R_hat, f_hat, w_hat, dt)
        return symmetrize(Phi @ np.asarray(P, dtype=float) @ Phi.T + Q), Phi, Q

    def accumulate(self, samples, dt=None):
        """Accumulate a burst of steps into (Phi_bar, Q_bar).

        ``samples`` is an iterable of ``(R_hat, f_hat, w_hat)`` triples ordered
        in time.  The recursion is the standard
        ``Phi_bar <- Phi_k Phi_bar`` and ``Q_bar <- Phi_k Q_bar Phi_k^T + Q_k``,
        so that ``P^- = Phi_bar P Phi_bar^T + Q_bar`` over the whole interval.
        """
        Phi_bar = np.eye(STATE_DIM)
        Q_bar = np.zeros((STATE_DIM, STATE_DIM))
        for (R_hat, f_hat, w_hat) in samples:
            Phi_k, Q_k = self.step(R_hat, f_hat, w_hat, dt)
            Phi_bar = Phi_k @ Phi_bar
            Q_bar = symmetrize(Phi_k @ Q_bar @ Phi_k.T + Q_k)
        return Phi_bar, Q_bar


def split_position_row(Phi):
    """Split the position row of a transition matrix into its five 3x3 blocks.

    Returns ``{'F_p', 'vel', 'att', 'acc_bias', 'gyro_bias', 'F_z'}`` where
    ``F_z`` is the concatenated (3, 12) non-position part, so that

        H Phi = [F_p | F_z],   F_z = [vel | att | acc_bias | gyro_bias].

    M12 uses the four named blocks to attribute the dual-layer structural error
    to a physical channel; ``F_p`` carries the position-to-position part that
    the dual layer represents EXACTLY and that therefore contributes no error.
    """
    Phi = np.asarray(Phi, dtype=float)
    if Phi.shape != (STATE_DIM, STATE_DIM):
        raise ValueError(f"expected a {STATE_DIM}x{STATE_DIM} matrix, got {Phi.shape}")
    row = Phi[IDX_P, :]
    out = {"F_p": row[:, IDX_P].copy(), "F_z": row[:, POS_DIM:].copy()}
    for name in Z_CHANNELS:
        sl = Z_CHANNEL_SLICES[name]
        out[name] = out["F_z"][:, sl].copy()
    return out


def coupling_norm(Phi_bar):
    """||Phi_bar[0:3, 6:9]||_F -- the measured attitude-coupling scalar kappa."""
    return float(np.linalg.norm(np.asarray(Phi_bar, dtype=float)[IDX_P, IDX_TH],
                                "fro"))


def propagate_state(state, f_meas, w_meas, dt, gravity, order="1"):
    """Nominal propagation with a RAW IMU sample (biases held constant).

    Kept here rather than in M4 so that the error-state Jacobian and the nominal
    step it linearises are defined side by side and can be cross-checked.
    """
    from l1_mechanization import correct_imu, mechanize
    f_hat, w_hat = correct_imu(f_meas, w_meas, state.b_a, state.b_g)
    out = mechanize(state, f_hat, w_hat, dt, gravity, order)
    out.b_a = state.b_a.copy()
    out.b_g = state.b_g.copy()
    return out


if __name__ == "__main__":
    from config import default_config, validate_config
    from l1_so3 import Exp, NominalState, boxplus, boxminus

    cfg = default_config(); validate_config(cfg)
    rng = np.random.default_rng(4)
    g = cfg["gravity"]
    dt = cfg["dt_imu"]

    R_hat = Exp(np.array([0.1, -0.2, 0.4]))
    f_hat = np.array([0.4, -0.9, 9.9])
    w_hat = np.array([0.02, -0.01, 0.35])

    # ---- 1) structure of F and of the first-order Phi ----
    F = error_dynamics_F(R_hat, f_hat, w_hat)
    assert np.allclose(F[IDX_P, IDX_V], np.eye(3))
    assert np.allclose(F[IDX_V, IDX_TH], -R_hat @ skew(f_hat))
    assert np.allclose(F[IDX_V, IDX_BA], -R_hat)
    assert np.allclose(F[IDX_TH, IDX_TH], -skew(w_hat))
    assert np.allclose(F[IDX_TH, IDX_BG], -np.eye(3))
    assert np.allclose(F[IDX_BA, :], 0.0) and np.allclose(F[IDX_BG, :], 0.0)
    Phi1 = transition_matrix(R_hat, f_hat, w_hat, dt, "1")
    assert np.allclose(Phi1[IDX_P, IDX_V], dt * np.eye(3))
    assert np.allclose(Phi1[IDX_P, IDX_TH], 0.0)          # only order 2 has it
    Phi2 = transition_matrix(R_hat, f_hat, w_hat, dt, "2")
    assert np.allclose(Phi2[IDX_P, IDX_TH], -0.5 * R_hat @ skew(f_hat) * dt ** 2)

    # ---- 2) Phi really is the Jacobian of the nominal step (T-jacobian) ----
    #        propagate a perturbed truth and the nominal with the SAME raw IMU
    #        The (nominal_int_order, phi_order) pair must be reported together:
    #        each Phi linearises ONE nominal integrator.  '1' pairs with '1';
    #        'midpoint' pairs with '2' / 'van_loan', which additionally carry the
    #        0.5 F^2 dt^2 term.  A mismatched pair is not wrong, just less
    #        accurate, and the check below pins that down quantitatively.
    base = NominalState(p=rng.standard_normal(3), v=rng.standard_normal(3),
                        R=R_hat, b_a=0.01 * rng.standard_normal(3),
                        b_g=1e-3 * rng.standard_normal(3))
    f_meas = f_hat + base.b_a
    w_meas = w_hat + base.b_g

    def jacobian_residual(nominal_order, phi_order, step, trials=20):
        probe = np.random.default_rng(9)
        worst = 0.0
        Phi = transition_matrix(base.R, f_hat, w_hat, step, phi_order)
        for _ in range(trials):
            dx = 1e-7 * probe.standard_normal(STATE_DIM)
            nxt_base = propagate_state(base, f_meas, w_meas, step, g, nominal_order)
            nxt_pert = propagate_state(boxplus(base, dx), f_meas, w_meas, step,
                                       g, nominal_order)
            actual = boxminus(nxt_pert, nxt_base)
            worst = max(worst, float(np.linalg.norm(actual - Phi @ dx)
                                     / np.linalg.norm(dx)))
        return worst

    residual = {(n, p): jacobian_residual(n, p, dt)
                for n in ("1", "midpoint") for p in ("1", "2", "van_loan")}
    # every pair is a valid first-order Jacobian
    assert max(residual.values()) < 1e-3, residual
    # matched pairs beat the mismatched ones
    assert residual[("1", "1")] < residual[("midpoint", "1")]
    assert residual[("midpoint", "2")] < residual[("1", "2")]
    # the tightest pair is (midpoint, order 2) and it beats (1, 1) outright
    assert residual[("midpoint", "2")] < 0.5 * residual[("1", "1")], residual
    assert abs(residual[("midpoint", "2")]
               - residual[("midpoint", "van_loan")]) < 1e-8
    # ... and the error is genuinely second order in dt
    for pair in (("1", "1"), ("midpoint", "2")):
        coarse = residual[pair]
        fine = jacobian_residual(pair[0], pair[1], 0.5 * dt)
        assert 3.5 < coarse / fine < 4.5, (pair, coarse, fine)

    # ---- 3) Q is symmetric PSD, linear in dt, and uses FILTER parameters ----
    Q = process_noise(R_hat, dt, cfg)
    assert np.allclose(Q, Q.T)
    assert np.min(np.linalg.eigvalsh(Q)) >= -1e-18
    assert np.allclose(process_noise(R_hat, 2 * dt, cfg), 2 * Q)
    assert abs(Q[IDX_V, IDX_V][0, 0] - cfg["sigma_acc"] ** 2 * dt) < 1e-18
    assert abs(Q[IDX_BA, IDX_BA][0, 0]
               - cfg["filter_sigma_acc_bias"] ** 2 * dt) < 1e-24
    assert np.allclose(Q[IDX_P, :], 0.0)          # position is noise-free
    tuned = default_config(); tuned["filter_sigma_acc_bias"] *= 3.0
    validate_config(tuned)
    assert process_noise(R_hat, dt, tuned)[IDX_BA, IDX_BA][0, 0] > \
        Q[IDX_BA, IDX_BA][0, 0]

    # ---- 4) van Loan agrees with the series to the expected order ----
    Phi_vl, Q_vl = van_loan_step(R_hat, f_hat, w_hat, dt, cfg)
    assert np.max(np.abs(Phi_vl - Phi2)) < 1e-6
    assert np.max(np.abs(Phi_vl - Phi2)) < np.max(np.abs(Phi_vl - Phi1))
    assert np.allclose(Q_vl, Q_vl.T) and np.min(np.linalg.eigvalsh(Q_vl)) >= -1e-18
    assert np.max(np.abs(Q_vl - Q)) < 0.05 * np.max(np.abs(Q))

    # ---- 5) burst accumulation reproduces a step-by-step propagation ----
    pm = ProcessModel(cfg)
    n = cfg["imu_per_uwb"]
    samples = [(Exp(w_hat * (k * dt)) @ R_hat, f_hat, w_hat) for k in range(n)]
    P0 = np.diag(np.concatenate([
        np.full(3, cfg["init_pos_std"] ** 2), np.full(3, cfg["init_vel_std"] ** 2),
        np.full(3, cfg["init_att_std"] ** 2),
        np.full(3, cfg["init_acc_bias_std"] ** 2),
        np.full(3, cfg["init_gyro_bias_std"] ** 2)]))
    P_seq = P0.copy()
    for s in samples:
        P_seq, _, _ = pm.propagate(P_seq, *s)
    Phi_bar, Q_bar = pm.accumulate(samples)
    P_acc = symmetrize(Phi_bar @ P0 @ Phi_bar.T + Q_bar)
    assert np.max(np.abs(P_acc - P_seq)) < 1e-12 * max(1.0, np.max(np.abs(P_seq)))

    # ---- 6) the accumulated position-attitude coupling follows 0.5 a T^2 ----
    T = cfg["sync_interval"]
    kappa = coupling_norm(Phi_bar)
    predicted = np.sqrt(2.0) * 0.5 * float(np.linalg.norm(f_hat)) * T ** 2
    assert 0.5 * predicted < kappa < 2.0 * predicted, (kappa, predicted)
    # and it scales as T^2 when the interval doubles
    Phi_bar2, _ = pm.accumulate(samples * 2)
    assert 3.0 < coupling_norm(Phi_bar2) / kappa < 5.0

    # ---- 7) split_position_row is an exact partition ----
    parts = split_position_row(Phi_bar)
    assert parts["F_p"].shape == (3, 3) and parts["F_z"].shape == (3, 12)
    assert np.allclose(np.hstack([parts["F_p"], parts["F_z"]]), Phi_bar[IDX_P, :])
    assert np.allclose(np.hstack([parts[c] for c in Z_CHANNELS]), parts["F_z"])
    assert np.allclose(parts["att"], Phi_bar[IDX_P, IDX_TH])
    assert np.allclose(parts["vel"], Phi_bar[IDX_P, IDX_V])
    try:
        split_position_row(np.eye(3))
        raise AssertionError("split_position_row accepted a wrong shape")
    except ValueError:
        pass
    assert np.allclose(position_selector() @ Phi_bar,
                       np.hstack([parts["F_p"], parts["F_z"]]))

    print("[M7 ProcessModel] OK  (Jacobian residual (1,1)=%.1e "
          "(midpoint,2)=%.1e, kappa=%.3e vs sqrt2*0.5|f|T^2=%.3e)"
          % (residual[("1", "1")], residual[("midpoint", "2")], kappa,
             predicted))
