"""Shared ESKF estimator base (L2 Estimator subsystem) -- v3.

Every cooperative filter of section 6 of the restructured plan runs the same
cadence, and the base class owns it so that the five architectures differ ONLY
in how they represent cross-entity correlation -- which is the single axis RQ1
compares.  The cadence is

    predict(imu_seq, T)     nominal mechanisation (M4) at dt_imu
                            + error-covariance recursion (M7) at dt_imu
    update(range_edges)     scalar sequential range updates (M8) at the UWB rate
                            -> ONE manifold injection (M2 boxplus) at the end
                            -> error-state reset  P <- G P G^T

Two cadence rules are enforced here rather than being left to each subclass,
because a difference in either would silently masquerade as an architectural
difference in the paper's tables:

1. **One linearisation point per epoch.**  Every edge of an epoch is predicted
   at the nominal state as it stood when the epoch began, and each sequential
   scalar update subtracts the running error mean (``y - h dx``) instead of
   re-predicting.  A batch update at a fixed nominal and this sequential form
   are then algebraically identical, which is what makes M19 == M15 for K = 1
   (T10) and what keeps M26's cadence comparable to M15's.
2. **One injection per epoch.**  The nominal state never moves inside
   ``update``; the accumulated ``dx`` is injected once, and the reset Jacobian
   is built from that single accumulated correction.

``predict``'s public ``dt`` argument is the COMPLETE UWB interval T (this
differs from v2, where the same argument meant the IMU sample period; the
validator now rejects the v2 reading instead of silently accepting it).

Interface every estimator honours (M20 depends on exactly this set)::

    reset(truth_states, seed)      predict(imu_seq, dt)     update(range_edges)
    get_nominal_state()            get_pos_estimate()
    get_eval_cov()      -> (3K, 3K) joint POSITION covariance, as reported
    get_eval_cov_full() -> (15K, 15K) full-state covariance   [OPTIONAL]
    get_full_cov()      -> alias of the above                 [OPTIONAL]
    evaluated_channels  -> class attribute; M26 narrows it to ('pos',)
    get_comm_summary()  -> M11 ledger split into C_measurement / C_estimation

The two optional methods are deliberately optional: M26 has no attitude and no
biases, and inventing a padded 15K covariance for it would produce a number
that reads like a consistency result while meaning nothing (see M26).
"""

import numpy as np

from config import (IDX_BA, IDX_BG, IDX_P, IDX_TH, IDX_V, POS_DIM, STATE_DIM,
                    validate_config)
from l1_linalg import symmetrize
from l1_mechanization import correct_imu, mechanize
from l1_process_model import ProcessModel, position_selector
from l1_range_model import predict_range
from l1_resource import CommunicationLedger
from l1_so3 import NominalState, boxplus, reset_jacobian

H_POS = position_selector()          # 3 x 15 selector, H x = dp

CHANNEL_STDS = (("init_pos_std", IDX_P), ("init_vel_std", IDX_V),
                ("init_att_std", IDX_TH), ("init_acc_bias_std", IDX_BA),
                ("init_gyro_bias_std", IDX_BG))


def assemble_reset(dx, mode="first_order"):
    """G = blkdiag(I, I, G_theta, I, I) of one entity's error-state reset."""
    dx = np.asarray(dx, dtype=float).ravel()
    G = np.eye(STATE_DIM)
    G[IDX_TH, IDX_TH] = reset_jacobian(dx[IDX_TH], mode)
    return G


class CooperativeESKF:
    """Common cadence of the cooperative ESKF family."""

    name = "base"
    evaluated_channels = ("pos", "vel", "att", "acc_bias", "gyro_bias")
    provides_full_cov = True

    def __init__(self, config):
        validate_config(config)
        self.cfg = config
        self.K = int(config["num_entities"])
        self.dt = float(config["dt_imu"])
        self.g = np.asarray(config["gravity"], dtype=float)
        self.anchors = np.asarray(config["anchor_positions"], dtype=float)
        self.sigma_r = float(config["sigma_range"])
        self.R_range = self.sigma_r ** 2
        self.joseph = bool(config["joseph_form"])
        self.reset_mode = config["reset_jacobian"]
        self.nominal_order = config["nominal_int_order"]
        self.pm = ProcessModel(config)
        self.nominal = None                     # list[NominalState]
        self.flops = 0
        self.epochs = 0
        self.last_innovations = []              # [{'i','j','u','S','y'}] of M12
        self.ledger = CommunicationLedger(config, self.name)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def initial_local_cov(self):
        """P_0 of one entity: diagonal, from the configured init std devs."""
        d = np.zeros(STATE_DIM)
        for key, sl in CHANNEL_STDS:
            d[sl] = float(self.cfg[key]) ** 2
        return np.diag(d)

    def init_nominal_from_truth(self, truth_states, seed=None):
        """Nominal states carrying a sampled error ``e0 = truth boxminus nom``.

        The sign convention matters and is fixed here once: M10 forms its error
        vector as ``boxminus(truth, nominal)``, so the nominal must be built as
        ``truth boxplus (-e0)``.  Under a shared seed every method receives the
        BIT-IDENTICAL initial error (M20's fairness self-check), which is what
        makes single-trial differences attributable to the architecture.
        """
        if len(truth_states) != self.K:
            raise ValueError(f"expected {self.K} truth states, got {len(truth_states)}")
        rng = np.random.default_rng(seed)
        stds = np.zeros(STATE_DIM)
        for key, sl in CHANNEL_STDS:
            stds[sl] = float(self.cfg[key])
        return [boxplus(st, -(stds * rng.standard_normal(STATE_DIM)))
                for st in truth_states]

    def validate_predict_input(self, imu_seq, dt=None):
        """Check the public predict contract; return the IMU sample period."""
        if self.nominal is None:
            raise RuntimeError("reset() must be called before predict()")
        if len(imu_seq) == 0:
            raise ValueError("imu_seq must contain at least one IMU sample")
        for t, sample in enumerate(imu_seq):
            if len(sample) != self.K:
                raise ValueError(
                    f"IMU sample {t} must hold {self.K} entities, got {len(sample)}")
            for pair in sample:
                if len(pair) != 2:
                    raise ValueError("each IMU measurement must be (a_m, w_m)")
                a_m = np.asarray(pair[0], dtype=float)
                w_m = np.asarray(pair[1], dtype=float)
                if a_m.shape != (3,) or w_m.shape != (3,):
                    raise ValueError("a_m and w_m must be 3-vectors")
                if not (np.all(np.isfinite(a_m)) and np.all(np.isfinite(w_m))):
                    raise ValueError("IMU measurements contain NaN or Inf")
        interval = len(imu_seq) * self.dt
        if dt is not None:
            total = float(dt)
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError("predict dt must be positive and finite")
            if not np.isclose(total, interval, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    "public predict dt is the WHOLE UWB interval and must equal "
                    f"len(imu_seq)*dt_imu ({total:.12g} vs {interval:.12g})")
        return self.dt

    @staticmethod
    def entity_imu_seq(imu_seq, i):
        """Entity i's ``(a_m, w_m)`` sequence over one UWB interval."""
        return [imu_seq[t][i] for t in range(len(imu_seq))]

    # ------------------------------------------------------------------
    # propagation / injection primitives
    # ------------------------------------------------------------------
    def propagate_entity(self, nom, P_i, seq_i, accumulate=False):
        """Propagate one entity over a whole IMU burst.

        Returns ``(nom+, P+, Phi_bar)``.  The Jacobian of step k is linearised
        at the nominal BEFORE that step, matching the nominal integration it
        describes.  Biases are removed by M4's ``correct_imu``; the raw sample
        is never fed to the mechanisation and the true bias is never available
        here (separation contract).
        """
        dt = self.dt
        Phi_bar = np.eye(STATE_DIM) if accumulate else None
        for (a_m, w_m) in seq_i:
            f_hat, w_hat = correct_imu(a_m, w_m, nom.b_a, nom.b_g)
            Phi, Q = self.pm.step(nom.R, f_hat, w_hat, dt)
            P_i = Phi @ P_i @ Phi.T + Q
            nom = mechanize(nom, f_hat, w_hat, dt, self.g, self.nominal_order)
            if accumulate:
                Phi_bar = Phi @ Phi_bar
            self.flops += 4 * STATE_DIM ** 3
        return nom, symmetrize(P_i), Phi_bar

    def inject_and_reset(self, nom, P_i, dx_i):
        """boxplus injection + covariance reset; returns ``(nom+, P+, G)``."""
        nom = boxplus(nom, dx_i)
        G = assemble_reset(dx_i, self.reset_mode)
        return nom, symmetrize(G @ P_i @ G.T), G

    def apply_injection_and_reset(self, dx, covs):
        """Inject a stacked ``dx`` into every nominal and reset every block.

        ``dx`` is the concatenated (15K,) correction, ``covs`` the list of local
        15x15 covariances.  Returns ``(covs+, [G_i])``.  Subclasses that keep
        cross blocks apply the same ``G_i`` on both sides of ``P^ij`` themselves;
        the per-entity part is shared here so that the reset lemma (T8) holds by
        construction for every member of the family rather than five times over.
        """
        Gs = []
        out = []
        for i in range(self.K):
            dxi = np.asarray(dx, dtype=float).ravel()[i * STATE_DIM:(i + 1) * STATE_DIM]
            nom, P_i, G = self.inject_and_reset(self.nominal[i], covs[i], dxi)
            self.nominal[i] = nom
            out.append(P_i)
            Gs.append(G)
        return out, Gs

    # ------------------------------------------------------------------
    # measurement primitives (M8 through a single door)
    # ------------------------------------------------------------------
    def positions(self):
        """Current nominal positions as a list of 3-vectors."""
        return [s.p for s in self.nominal]

    def edge_geometry(self, edge, positions=None):
        """Predicted range, LOS unit vector and innovation of one edge.

        The main line has no lever arm, so the endpoint is the body origin and
        the row is position-only: ``u^T`` at i and ``-u^T`` at j.
        """
        pos = self.positions() if positions is None else positions
        d_hat, u = predict_range(edge, pos, self.anchors)
        return d_hat, u, float(edge.meas) - d_hat

    # ------------------------------------------------------------------
    # small assembly helpers
    # ------------------------------------------------------------------
    @staticmethod
    def block_diag_position(blocks):
        """Assemble a block-diagonal 3K x 3K position covariance."""
        K = len(blocks)
        P = np.zeros((POS_DIM * K, POS_DIM * K))
        for i in range(K):
            P[i * POS_DIM:(i + 1) * POS_DIM,
              i * POS_DIM:(i + 1) * POS_DIM] = blocks[i]
        return P

    @staticmethod
    def block_diag_full(blocks):
        """Assemble a block-diagonal 15K x 15K full-state covariance."""
        K = len(blocks)
        P = np.zeros((STATE_DIM * K, STATE_DIM * K))
        for i in range(K):
            P[i * STATE_DIM:(i + 1) * STATE_DIM,
              i * STATE_DIM:(i + 1) * STATE_DIM] = blocks[i]
        return P

    def charge_epoch(self, measurement=None, estimation=None):
        """Charge one UWB epoch to the M11 communication ledger."""
        self.ledger.add_epoch(measurement=measurement, estimation=estimation)

    # ------------------------------------------------------------------
    # interface
    # ------------------------------------------------------------------
    def reset(self, truth_states, seed=None):
        raise NotImplementedError

    def predict(self, imu_seq, dt=None):
        raise NotImplementedError

    def update(self, range_edges):
        raise NotImplementedError

    def get_nominal_state(self):
        return [s.copy() for s in self.nominal]

    def get_pos_estimate(self):
        return np.concatenate([s.p for s in self.nominal])

    def get_eval_cov(self):
        raise NotImplementedError

    def get_reported_cov(self):
        return self.get_eval_cov()

    def get_comm_summary(self):
        return self.ledger.summary()

    def get_flops(self):
        return int(self.flops)

    def get_innovations(self):
        """Last epoch's ``{'i','j','u','S'}`` records (M12's at_innovation)."""
        return list(self.last_innovations)


def sample_initial_states(truth_states, config, seed):
    """Standalone helper: the initial nominal set shared by all methods."""
    probe = CooperativeESKF(config)
    return probe.init_nominal_from_truth(truth_states, seed)


if __name__ == "__main__":
    from config import apply_entity_count, default_config
    from l1_so3 import boxminus
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    cfg = default_config(); apply_entity_count(cfg, 2); cfg["sim_duration"] = 4.0
    validate_config(cfg)
    base = CooperativeESKF(cfg)

    truth = TruthSimulator(cfg); truth.reset(0)
    sensor = SensorSimulator(cfg); sensor.reset(1)
    tstates = truth.get_truth_states()

    # ---- 1) the sampled initial error follows boxminus(truth, nominal) ----
    noms = base.init_nominal_from_truth(tstates, seed=0)
    e0 = boxminus(tstates[0], noms[0])
    assert np.linalg.norm(e0[IDX_P]) < 5 * cfg["init_pos_std"]
    zero = default_config(); apply_entity_count(zero, 1)
    for key, _ in CHANNEL_STDS:
        zero[key] = 0.0
    validate_config(zero)
    zn = CooperativeESKF(zero).init_nominal_from_truth([tstates[0]], seed=4)[0]
    assert np.allclose(boxminus(tstates[0], zn), 0.0)

    # ---- 2) the same seed gives bit-identical initial errors ----
    a = base.init_nominal_from_truth(tstates, seed=7)
    b = CooperativeESKF(cfg).init_nominal_from_truth(tstates, seed=7)
    assert all(np.array_equal(x.p, y.p) and np.array_equal(x.R, y.R)
               for x, y in zip(a, b))

    # ---- 3) propagation keeps symmetry and returns an accumulated Phi ----
    base.nominal = noms
    imu_seq = []
    for _ in range(cfg["imu_per_uwb"]):
        imu_seq.append(sensor.generate_imu(truth.step()))
    seq0 = base.entity_imu_seq(imu_seq, 0)
    P0 = base.initial_local_cov()
    nom1, P1, Phi_bar = base.propagate_entity(noms[0], P0, seq0, accumulate=True)
    assert np.allclose(P1, P1.T) and Phi_bar.shape == (STATE_DIM, STATE_DIM)
    assert np.min(np.linalg.eigvalsh(P1)) > 0.0

    # Phi_bar really is the burst product: rebuild (Phi_bar, Q_bar) through the
    # M7 accumulator along the SAME nominal trajectory and check the interval
    # recursion P^- = Phi_bar P0 Phi_bar^T + Q_bar reproduces P1 exactly.
    samples, walk = [], noms[0]
    for (a_m, w_m) in seq0:
        f_hat, w_hat = correct_imu(a_m, w_m, walk.b_a, walk.b_g)
        samples.append((walk.R, f_hat, w_hat))
        walk = mechanize(walk, f_hat, w_hat, base.dt, base.g, base.nominal_order)
    Phi_ref, Q_ref = base.pm.accumulate(samples)
    assert np.allclose(Phi_bar, Phi_ref, atol=1e-12)
    assert np.allclose(P1, symmetrize(Phi_ref @ P0 @ Phi_ref.T + Q_ref), atol=1e-10)

    # ---- 4) reset lemma: H P H^T is invariant under the error reset (T8) ----
    HPH = H_POS @ P1 @ H_POS.T
    dxi = np.zeros(STATE_DIM); dxi[IDX_TH] = [0.01, -0.02, 0.005]
    _, P2, G = base.inject_and_reset(nom1, P1, dxi)
    assert np.allclose(H_POS @ P2 @ H_POS.T, HPH, atol=1e-12)
    assert np.allclose(H_POS @ G, H_POS)

    # ---- 5) the public predict dt is the WHOLE interval ----
    ok = base.validate_predict_input(imu_seq, cfg["imu_per_uwb"] * cfg["dt_imu"])
    assert ok == cfg["dt_imu"]
    try:
        base.validate_predict_input(imu_seq, cfg["dt_imu"])       # the v2 reading
        raise AssertionError("the v2 per-sample dt reading was accepted")
    except ValueError:
        pass

    # ---- 6) edge geometry: anchor rows only touch entity i ----
    edges = sensor.generate_ranges(truth.snapshot())
    anchor_edge = next(e for e in edges if e.is_anchor)
    d_hat, u, y = base.edge_geometry(anchor_edge)
    assert abs(np.linalg.norm(u) - 1.0) < 1e-12 and np.isfinite(y)

    print("[ESKF base] OK  (Phi_bar matches M7 accumulate to %.1e, reset lemma "
          "|dHPH|=%.1e, init error %.3f m, predict dt contract enforced)"
          % (float(np.max(np.abs(Phi_bar - Phi_ref))),
             float(np.max(np.abs(H_POS @ P2 @ H_POS.T - HPH))),
             float(np.linalg.norm(e0[IDX_P]))))
