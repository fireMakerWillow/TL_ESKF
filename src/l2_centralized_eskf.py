"""M15 CentralizedESKF (L2 Estimator subsystem) -- the reference filter.

One joint 15K x 15K error-state covariance ``P_x`` in which every cross-entity
block is carried exactly.  Cadence (plan section 6.1.1):

    nominal state per entity through M4  ->  block-diagonal (Phi, Q) from M7,
    with the cross blocks riding along Phi and NO cross-entity Q  ->  scalar
    sequential range updates in the joint state  ->  one boxplus injection
    ->  P_x <- G P_x G^T with the cross blocks transformed on both sides.

Role in v3 (plan section 6.1.1, restated in the module design section [M15]).
M15 is not invoked as "the optimal estimator" but as **the limit of the dual
layer when the truncation vanishes**: M19 with every cross block retained IS
M15.  Every RQ2 comparison is therefore internal to one family, and a
consistency defect of M15 is inherited unchanged by M19, leaving the difference
between them -- the quantity the paper reports -- unaffected.

Its ``P_x`` is also the gold standard M12 differences against.

Numerics.  The covariance update is written as the symmetric rank-1 downdate
``P <- P - a a^T / S`` with ``a = P h^T``.  For a scalar measurement with the
exact gain this is algebraically the Joseph form (see the M1 self-test), but it
costs O(n^2) instead of O(n^3) per edge -- with 40 edges per epoch and n = 120
the difference is the whole run time of the experiment.  ``joseph_form=False``
selects the asymmetric short form so the ablation stays available.
"""

import numpy as np

from config import IDX_P, STATE_DIM
from l1_linalg import get_block, symmetrize
from l1_mechanization import correct_imu, mechanize
from l1_resource import flops_matmul
from l2_estimator_base import CooperativeESKF


class CentralizedESKF(CooperativeESKF):
    """Full-joint ESKF; the reference every other architecture is read against."""

    name = "centralized"

    # ------------------------------------------------------------------
    def reset(self, truth_states, seed=None):
        self.nominal = self.init_nominal_from_truth(truth_states, seed)
        P0 = self.initial_local_cov()
        self.P = self.block_diag_full([P0.copy() for _ in range(self.K)])
        self.flops = 0
        self.epochs = 0
        self.last_innovations = []
        return self

    # ------------------------------------------------------------------
    def _state_slice(self, i):
        return slice(i * STATE_DIM, (i + 1) * STATE_DIM)

    def _pos_index(self, i):
        return slice(i * STATE_DIM + IDX_P.start, i * STATE_DIM + IDX_P.stop)

    def _local_blocks(self):
        return [get_block(self.P, i, i, STATE_DIM).copy() for i in range(self.K)]

    def _congruence(self, Ms):
        """P <- blkdiag(M_i) P blkdiag(M_j)^T, done block-wise (O(K^2 d^3))."""
        Ms = np.asarray(Ms)
        K, d = self.K, STATE_DIM
        blocks = self.P.reshape(K, d, K, d).transpose(0, 2, 1, 3)
        tmp = np.einsum("iab,ijbc->ijac", Ms, blocks, optimize=True)
        blocks = np.einsum("ijac,jdc->ijad", tmp, Ms, optimize=True)
        return blocks.transpose(0, 2, 1, 3).reshape(K * d, K * d)

    # ------------------------------------------------------------------
    def predict(self, imu_seq, dt=None):
        """Propagate the joint covariance over one UWB interval.

        The joint transition is block diagonal, so the propagation is done
        block-wise (``P^ij <- Phi_i P^ij Phi_j^T``) rather than by forming the
        dense 15K x 15K product: the two are identical, but the block form is
        O(K^2 d^3) instead of O(K^3 d^3).  Process noise is added on the
        diagonal only -- there is no cross-entity Q, because the entities'
        IMU noises are independent.
        """
        dt_step = self.validate_predict_input(imu_seq, dt)
        K, d = self.K, STATE_DIM

        for t in range(len(imu_seq)):
            Phis = np.empty((K, d, d))
            Qs = np.empty((K, d, d))
            for i in range(K):
                nom = self.nominal[i]
                a_m, w_m = imu_seq[t][i]
                f_hat, w_hat = correct_imu(a_m, w_m, nom.b_a, nom.b_g)
                Phis[i], Qs[i] = self.pm.step(nom.R, f_hat, w_hat, dt_step)
                self.nominal[i] = mechanize(nom, f_hat, w_hat, dt_step, self.g,
                                            self.nominal_order)
            self.P = self._congruence(Phis)
            for i in range(K):
                self.P[self._state_slice(i), self._state_slice(i)] += Qs[i]
            self.P = symmetrize(self.P)
            self.flops += K * K * 2 * flops_matmul(d, d, d)

    # ------------------------------------------------------------------
    def _row_apply(self, edge, u):
        """Return ``a = P h^T`` for one range edge, exploiting the sparsity.

        ``h`` has ``u^T`` on entity i's position rows and ``-u^T`` on j's; the
        product therefore touches at most six columns of P.
        """
        a = self.P[:, self._pos_index(edge.i)] @ u
        if not edge.is_anchor:
            a = a - self.P[:, self._pos_index(edge.j)] @ u
        return a

    def update(self, range_edges):
        """Sequential scalar range updates at the epoch's linearisation point."""
        if self.nominal is None:
            raise RuntimeError("reset() must be called before update()")
        n = self.K * STATE_DIM
        dx = np.zeros(n)
        positions = self.positions()          # frozen for the whole epoch
        self.last_innovations = []

        for e in range_edges:
            _, u, y = self.edge_geometry(e, positions)
            if not np.any(u):
                continue
            a = self._row_apply(e, u)
            S = float(a[self._pos_index(e.i)] @ u + self.R_range)
            if not e.is_anchor:
                S -= float(a[self._pos_index(e.j)] @ u)
            if not np.isfinite(S) or S <= 0.0:
                raise np.linalg.LinAlgError(
                    f"non-positive innovation variance S={S} on {e}")
            # running-mean correction: the sequential form then equals a batch
            # update taken at the epoch's fixed nominal
            hdx = float(u @ dx[self._pos_index(e.i)])
            if not e.is_anchor:
                hdx -= float(u @ dx[self._pos_index(e.j)])
            gain = a / S
            dx = dx + gain * (y - hdx)
            if self.joseph:
                self.P = symmetrize(self.P - np.outer(a, a) / S)
            else:
                self.P = self.P - np.outer(gain, a)
            self.last_innovations.append(
                {"i": e.i, "j": (-1 if e.is_anchor else e.j), "u": u.copy(),
                 "S": S, "y": y})
            self.flops += 2 * n * n

        # one injection, one reset, cross blocks transformed on both sides
        _, Gs = self.apply_injection_and_reset(dx, self._local_blocks())
        self.P = symmetrize(self._congruence(Gs))

        self.epochs += 1
        self.charge_epoch(measurement=len(range_edges))
        return dx

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------
    def _position_indices(self):
        return np.concatenate([np.arange(i * STATE_DIM + IDX_P.start,
                                         i * STATE_DIM + IDX_P.stop)
                               for i in range(self.K)])

    def get_eval_cov(self):
        """Joint 3K position marginal, cross blocks included (exact)."""
        idx = self._position_indices()
        return self.P[np.ix_(idx, idx)].copy()

    def get_eval_cov_full(self):
        return self.P.copy()

    def get_full_cov(self):
        return self.P.copy()

    def get_local_covariances(self):
        return self._local_blocks()


if __name__ == "__main__":
    from config import apply_entity_count, default_config, validate_config
    from l1_metrics import evaluate, single_run_bounds
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    def run(cfg, seed=0):
        validate_config(cfg)
        truth = TruthSimulator(cfg); truth.reset(seed)
        sensor = SensorSimulator(cfg); sensor.reset(seed + 5000)
        est = CentralizedESKF(cfg); est.reset(truth.get_truth_states(), seed=seed)
        n = cfg["imu_per_uwb"]
        steps = int(round(cfg["sim_duration"] / cfg["sync_interval"]))
        rows = []
        for _ in range(steps):
            imu = [sensor.generate_imu(truth.step()) for _ in range(n)]
            est.predict(imu, n * cfg["dt_imu"])
            est.update(sensor.generate_ranges(truth.snapshot()))
            rows.append(evaluate(truth.get_truth_states(), est.get_nominal_state(),
                                 est.get_eval_cov(), cfg,
                                 full_cov=est.get_eval_cov_full()))
        return est, rows

    cfg = default_config(); cfg["sim_duration"] = 20.0
    est, rows = run(cfg)

    # ---- 1) the joint covariance stays symmetric and positive definite ----
    P = est.get_full_cov()
    assert np.allclose(P, P.T, atol=1e-12)
    assert float(np.min(np.linalg.eigvalsh(P))) > 0.0

    # ---- 2) cross-entity blocks are genuinely populated (T14's counterpart) --
    cross = max(float(np.max(np.abs(get_block(P, i, j, STATE_DIM))))
                for i in range(cfg["num_entities"])
                for j in range(cfg["num_entities"]) if i != j)
    assert cross > 1e-6, cross

    # ---- 3) accuracy and consistency of the reference ----
    tail = rows[len(rows) // 2:]
    rmse = float(np.mean([r["pos_rmse"] for r in tail]))
    nees = float(np.mean([r["nees_pos"] for r in tail]))
    lo, hi = single_run_bounds(3 * cfg["num_entities"], 0.999)
    assert rmse < 1.0, rmse
    assert lo < nees < hi, (nees, lo, hi)

    # ---- 4) T6 bias observability: the accel-bias error shrinks ----
    ba0 = float(np.mean([r["acc_bias_rmse"] for r in rows[:5]]))
    ba1 = float(np.mean([r["acc_bias_rmse"] for r in rows[-5:]]))
    assert ba1 < ba0, (ba0, ba1)

    # ---- 5) T8 reset lemma directly on the joint covariance -------------
    idx = est._position_indices()
    before = est.P[np.ix_(idx, idx)].copy()
    dx = np.zeros(est.K * STATE_DIM)
    rng = np.random.default_rng(11)
    for i in range(est.K):
        dx[i * STATE_DIM + 6:i * STATE_DIM + 9] = 0.05 * rng.standard_normal(3)
    saved = [s.copy() for s in est.nominal]
    _, Gs = est.apply_injection_and_reset(dx, est._local_blocks())
    est.P = symmetrize(est._congruence(Gs))
    after = est.P[np.ix_(idx, idx)]
    assert np.allclose(after, before, atol=1e-12), float(np.max(np.abs(after - before)))
    est.nominal = saved

    # ---- 6) K = 1 sanity: a single entity with four anchors converges ----
    one = default_config(); apply_entity_count(one, 1); one["sim_duration"] = 10.0
    est1, rows1 = run(one, seed=1)
    assert rows1[-1]["pos_rmse"] < 0.5, rows1[-1]["pos_rmse"]

    print("[M15 CentralizedESKF] OK  (tail RMSE=%.3f m, NEES=%.1f/%d, "
          "acc-bias RMSE %.4f -> %.4f, max|P^ij|=%.2e, reset lemma %.1e)"
          % (rmse, nees, 3 * cfg["num_entities"], ba0, ba1, cross,
             float(np.max(np.abs(after - before)))))
