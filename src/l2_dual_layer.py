"""M19 DualLayerESKF (L2 Estimator subsystem) -- the method under study.

Two coupled layers per the restructured plan section 3:

    local layer   P_L,i in R^{15x15}, one per entity, full ESKF state
    global layer  P_G in R^{3K x 3K}, POSITION only, cross blocks included

and one regression lift ``L_i = [I; B_i]`` (M9) tying them together.  The cycle
(section 3.9, step for step):

    1  L_i^rst  from the covariance left by the PREVIOUS reset
    2  local high-rate propagation of P_L,i, accumulating Phi_bar_i; no comms
    3  global sync:  P_G^{ii,-} = H P_L,i^- H^T          (exact, no approximation)
                     P_G^{ij,-} = A_i P_G^{ij,+} A_j^T,  A_i = H Phi_bar_i L_i^rst
    4  L_{i,-}   from the propagated, pre-update P_L,i^-
    5  global UWB update in the 3K position layer (sequential scalar, M8 rows)
    6  downlink c_i = L_{i,-} delta p_i;  X_i <- X_i boxplus c_i
                     P_L,i^+ = P_L,i^- - L_{i,-} Delta_i L_{i,-}^T,
                     Delta_i = P_G^{ii,-} - P_G^{ii,+}
    7  reset G_i; P_L,i <- G_i P_L,i^+ G_i^T; write back P_G^{ii} = H P_L,i H^T;
       leave the OFF-DIAGONAL blocks of P_G untouched; recompute L_i^rst

Step 3 is where the method's only approximation lives.  The diagonal is exact;
the off-diagonal is regressed through the 3x3 map ``A_i`` instead of the full
15x15 ``Phi_bar_i``, and the resulting structural residual
``E^{ij} = P_x^{ij} - L_i P_G^{ij} L_j^T`` is what M12 measures and what RQ2
maps over ``(v, T)``.

Two contracts are asserted at runtime rather than trusted:

* **The dual L.**  ``L_i^rst`` (step 1) and ``L_{i,-}`` (step 4) are separate
  caches with different jobs, and swapping them is a silent error: both are
  valid lifts, so nothing crashes, the marginal consistency check still passes,
  and only the cross-block accuracy quietly degrades.  ``lifts_distinguishable``
  (T22) is therefore checked at run time, and the two are never aliased.
* **Marginal consistency.**  After the write-back, ``P_G^{ii}`` must equal
  ``H P_L,i H^T`` -- this is T9 and it is the closed loop between the layers.
  Since ``H G_i = H``, the reset cannot change it, which is also why the
  off-diagonal blocks need no reset transform at all (section 3.7 lemma).

Exactness boundary (derived while implementing; sharper than the design's
"M15 is the limit of M19", and used as an acceptance criterion below).  Under
block-diagonal initialisation the invariants ``P_x^{ij} = L_i P_G^{ij} L_j^T``
and ``P_x^{ii} = D_i + L_i P_G^{ii} L_i^T`` hold at t = 0, and because
``D_i H^T = 0`` and ``H L_i = I``, they are preserved by the position update, by
the downlink, and by the reset (there ``L_i^rst = G_i L_i``, which needs
``recompute_L_after_reset=True``).  Hence:

    epoch 1  M19 == M15 bit for bit, in the nominal state AND in every block of
             the reconstructed P_x;
    epoch 2  P_G is still EXACTLY H P_x H^T (because H E^{ij} H^T = 0), while the
             non-position part of the cross blocks has already parted company;
    epoch 3  P_G itself starts to deviate, by exactly
             Delta P_G^{ij} = H Phi_bar_i E^{ij} Phi_bar_j^T H^T.

That is a much stronger acceptance test than "K = 1 is equivalent" (T10),
because it holds at K = 8 with every cross block populated; the self-test below
checks all three epochs, and the schedule is what turns a wrong step order into
a failing test instead of a slightly worse RMSE.
"""

import numpy as np

from config import IDX_P, POS_DIM, STATE_DIM
from l1_linalg import symmetrize
from l1_regression import compute_L, lifts_distinguishable, local_residual
from l1_resource import flops_matmul
from l2_estimator_base import H_POS, CooperativeESKF


class DualLayerESKF(CooperativeESKF):
    """Local 15-dim filters + a shared 3K position-layer covariance."""

    name = "dual_layer"

    def __init__(self, config):
        super().__init__(config)
        self.eps = float(config["reg_epsilon"])
        self.cross_mode = config.get("pg_cross_mode", "regression")
        if self.cross_mode not in ("regression", "keep"):
            raise ValueError("pg_cross_mode must be 'regression' or 'keep'")
        self.recompute_L = bool(config.get("recompute_L_after_reset", True))
        self.P_local = None
        self.P_G = None
        self.L_rst = None          # step 1 cache: built from the RESET covariance
        self.L_minus = None        # step 4 cache: built from the PRE-UPDATE one
        self.Phi_bar = None
        self.A_maps = None
        self.last_marginal_gap = float("nan")
        self.last_lift_gap = float("nan")

    # ------------------------------------------------------------------
    def _pos_slice(self, i):
        return slice(i * POS_DIM, (i + 1) * POS_DIM)

    def reset(self, truth_states, seed=None):
        self.nominal = self.init_nominal_from_truth(truth_states, seed)
        P0 = self.initial_local_cov()
        self.P_local = [P0.copy() for _ in range(self.K)]
        self.P_G = self.block_diag_position(
            [P[IDX_P, IDX_P] for P in self.P_local])
        self.L_rst = [compute_L(P, self.eps) for P in self.P_local]
        self.L_minus = None
        self.Phi_bar = None
        self.flops = 0
        self.epochs = 0
        self.last_innovations = []
        return self

    # ------------------------------------------------------------------
    def predict(self, imu_seq, dt=None):
        """Steps 2-4: local propagation, global sync, and the pre-update lift."""
        self.validate_predict_input(imu_seq, dt)
        self.Phi_bar = []
        for i in range(self.K):
            nom, P_i, Phi_bar = self.propagate_entity(
                self.nominal[i], self.P_local[i], self.entity_imu_seq(imu_seq, i),
                accumulate=True)
            self.nominal[i] = nom
            self.P_local[i] = P_i
            self.Phi_bar.append(Phi_bar)

        # ---- step 3: the two halves of the global sync ----
        self.A_maps = [H_POS @ self.Phi_bar[i] @ self.L_rst[i]
                       for i in range(self.K)]
        P_G = np.zeros_like(self.P_G)
        for i in range(self.K):
            P_G[self._pos_slice(i), self._pos_slice(i)] = symmetrize(
                self.P_local[i][IDX_P, IDX_P])
        for i in range(self.K):
            for j in range(i + 1, self.K):
                old = self.P_G[self._pos_slice(i), self._pos_slice(j)]
                if self.cross_mode == "regression":
                    block = self.A_maps[i] @ old @ self.A_maps[j].T
                else:                       # 'keep': frozen cross, ablation only
                    block = old.copy()
                P_G[self._pos_slice(i), self._pos_slice(j)] = block
                P_G[self._pos_slice(j), self._pos_slice(i)] = block.T
                self.flops += 2 * flops_matmul(POS_DIM, POS_DIM, POS_DIM)
        self.P_G = symmetrize(P_G)
        self.P_G_minus_diag = [self.P_G[self._pos_slice(i), self._pos_slice(i)].copy()
                               for i in range(self.K)]

        # ---- step 4: the SECOND lift, from the propagated local covariance ----
        self.L_minus = [compute_L(P, self.eps) for P in self.P_local]
        self._assert_dual_lift()

    def _assert_dual_lift(self):
        """T22: the two lifts are separate objects, and each satisfies H L = I."""
        for i in range(self.K):
            assert self.L_minus[i] is not self.L_rst[i], "the two lifts are aliased"
            assert np.allclose(H_POS @ self.L_minus[i], np.eye(POS_DIM), atol=1e-12)
        self.last_lift_gap = max(
            float(np.max(np.abs(self.L_minus[i] - self.L_rst[i])))
            for i in range(self.K))

    # ------------------------------------------------------------------
    def update(self, range_edges):
        """Steps 5-7: position-layer update, downlink, reset and write-back."""
        if self.nominal is None:
            raise RuntimeError("reset() must be called before update()")
        if self.L_minus is None:
            raise RuntimeError("predict() must run before update() (the global "
                               "sync and L_{i,-} are produced there)")
        n = self.K * POS_DIM
        dp = np.zeros(n)
        positions = self.positions()          # frozen for the whole epoch
        self.last_innovations = []

        # ---- step 5: sequential scalar updates in the 3K position layer ----
        for e in range_edges:
            _, u, y = self.edge_geometry(e, positions)
            if not np.any(u):
                continue
            a = self.P_G[:, self._pos_slice(e.i)] @ u
            if not e.is_anchor:
                a = a - self.P_G[:, self._pos_slice(e.j)] @ u
            S = float(u @ a[self._pos_slice(e.i)]) + self.R_range
            hdp = float(u @ dp[self._pos_slice(e.i)])
            if not e.is_anchor:
                S -= float(u @ a[self._pos_slice(e.j)])
                hdp -= float(u @ dp[self._pos_slice(e.j)])
            if not np.isfinite(S) or S <= 0.0:
                raise np.linalg.LinAlgError(
                    f"non-positive innovation variance S={S} on {e}")
            gain = a / S
            dp = dp + gain * (y - hdp)
            if self.joseph:
                self.P_G = symmetrize(self.P_G - np.outer(a, a) / S)
            else:
                self.P_G = self.P_G - np.outer(gain, a)
            self.last_innovations.append(
                {"i": e.i, "j": (-1 if e.is_anchor else e.j), "u": u.copy(),
                 "S": S, "y": y})
            self.flops += 2 * n * n

        # ---- step 6: downlink, local injection, local covariance update ----
        dx = np.zeros(self.K * STATE_DIM)
        P_plus = []
        for i in range(self.K):
            dp_i = dp[self._pos_slice(i)]
            c_i = self.L_minus[i] @ dp_i
            dx[i * STATE_DIM:(i + 1) * STATE_DIM] = c_i
            Delta = self.P_G_minus_diag[i] - self.P_G[self._pos_slice(i),
                                                      self._pos_slice(i)]
            P_plus.append(symmetrize(self.P_local[i]
                                     - self.L_minus[i] @ Delta @ self.L_minus[i].T))
            self.flops += 2 * flops_matmul(STATE_DIM, POS_DIM, POS_DIM)

        # ---- step 7: reset, write-back, recompute the lift ----
        self.P_local, Gs = self.apply_injection_and_reset(dx, P_plus)
        gap = 0.0
        for i in range(self.K):
            block = symmetrize(self.P_local[i][IDX_P, IDX_P])
            gap = max(gap, float(np.max(np.abs(
                block - self.P_G[self._pos_slice(i), self._pos_slice(i)]))))
            self.P_G[self._pos_slice(i), self._pos_slice(i)] = block
        self.last_marginal_gap = gap
        # the off-diagonal blocks are deliberately NOT transformed: H G_i = H,
        # so the reset moves no part of P_G (section 3.7)
        if self.recompute_L:
            self.L_rst = [compute_L(P, self.eps) for P in self.P_local]
        else:                                  # ablation: reuse the pre-update L
            self.L_rst = [L.copy() for L in self.L_minus]

        self.epochs += 1
        self.charge_epoch(measurement=len(range_edges))
        return dx

    # ------------------------------------------------------------------
    # outputs and diagnostic hooks
    # ------------------------------------------------------------------
    def get_eval_cov(self):
        """P_G itself -- the joint position covariance the method maintains."""
        return symmetrize(self.P_G).copy()

    def get_eval_cov_full(self):
        """Reconstructed 15K covariance: diag = P_L,i, cross = L_i P_G^ij L_j^T.

        This is the object M12 differences against M15's ``P_x``; the diagonal is
        exact by construction and only the cross blocks carry the truncation.
        """
        P = self.block_diag_full(self.P_local)
        for i in range(self.K):
            for j in range(i + 1, self.K):
                block = (self.L_rst[i]
                         @ self.P_G[self._pos_slice(i), self._pos_slice(j)]
                         @ self.L_rst[j].T)
                P[i * STATE_DIM:(i + 1) * STATE_DIM,
                  j * STATE_DIM:(j + 1) * STATE_DIM] = block
                P[j * STATE_DIM:(j + 1) * STATE_DIM,
                  i * STATE_DIM:(i + 1) * STATE_DIM] = block.T
        return symmetrize(P)

    def get_full_cov(self):
        return self.get_eval_cov_full()

    def get_local_covariances(self):
        return [P.copy() for P in self.P_local]

    def get_global_cov(self):
        return symmetrize(self.P_G).copy()

    def get_phi_bar(self):
        return [np.asarray(M).copy() for M in (self.Phi_bar or [])]

    def get_lifts(self):
        """``(L_minus, L_rst)`` -- the two caches, for T22 and M12."""
        return ([L.copy() for L in (self.L_minus or [])],
                [L.copy() for L in (self.L_rst or [])])

    def local_residuals(self):
        """``D_i`` of every entity (the part the position layer cannot carry)."""
        return [local_residual(P, eps=self.eps) for P in self.P_local]

    def lifts_are_distinguishable(self, tol=0.0):
        return all(lifts_distinguishable(self.L_minus[i], self.L_rst[i], tol)
                   for i in range(self.K))


if __name__ == "__main__":
    from config import apply_entity_count, default_config, validate_config
    from l1_linalg import get_block
    from l1_metrics import evaluate, single_run_bounds
    from l2_centralized_eskf import CentralizedESKF
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    def make(cfg, cls, seed=0):
        validate_config(cfg)
        truth = TruthSimulator(cfg); truth.reset(seed)
        sensor = SensorSimulator(cfg); sensor.reset(seed + 5000)
        est = cls(cfg); est.reset(truth.get_truth_states(), seed=seed)
        return est, truth, sensor

    def step(est, truth, sensor, cfg):
        n = cfg["imu_per_uwb"]
        imu = [sensor.generate_imu(truth.step()) for _ in range(n)]
        est.predict(imu, n * cfg["dt_imu"])
        est.update(sensor.generate_ranges(truth.snapshot()))

    # ------------------------------------------------------------------
    # 1) the exactness schedule at K = 8 (the sharp form of T10)
    # ------------------------------------------------------------------
    cfg = default_config(); cfg["sim_duration"] = 5.0
    dual, t1, s1 = make(cfg, DualLayerESKF, seed=4)
    ref, t2, s2 = make(cfg, CentralizedESKF, seed=4)
    K = cfg["num_entities"]

    def blocks(dual_est, ref_est):
        Pd, Pr = dual_est.get_eval_cov_full(), ref_est.get_full_cov()
        diag = max(float(np.max(np.abs(get_block(Pd, i, i, STATE_DIM)
                                       - get_block(Pr, i, i, STATE_DIM))))
                   for i in range(K))
        cross = max(float(np.max(np.abs(get_block(Pd, i, j, STATE_DIM)
                                        - get_block(Pr, i, j, STATE_DIM))))
                    for i in range(K) for j in range(K) if i != j)
        pg = float(np.max(np.abs(dual_est.get_eval_cov()
                                 - ref_est.get_eval_cov())))
        pos = float(np.max(np.abs(dual_est.get_pos_estimate()
                                  - ref_est.get_pos_estimate())))
        return diag, cross, pg, pos

    # Two complete epochs establish the exactness boundary.  At epoch 2 the
    # position layer still matches M15 exactly, while the reconstructed
    # non-position cross blocks have already separated.
    schedule = []
    for _ in range(2):
        step(dual, t1, s1, cfg)
        step(ref, t2, s2, cfg)
        schedule.append(blocks(dual, ref))

    (d1, c1, g1, x1), (d2, c2, g2, x2) = schedule
    scale = float(np.max(np.abs(ref.get_full_cov())))
    assert max(d1, c1, g1) < 1e-9 * scale and x1 < 1e-9, schedule[0]
    assert g2 < 1e-9 * scale, g2
    assert c2 > 1e-6 * scale, c2

    # The structural identity belongs at the epoch-2 boundary, before the
    # omitted cross-state information has had another interval to flow back
    # into position.  Checking it after epoch 3 would confuse representation
    # loss with the realised position-layer drift.
    i, j = 0, 5
    E2 = (get_block(ref.get_full_cov(), i, j, STATE_DIM)
          - get_block(dual.get_eval_cov_full(), i, j, STATE_DIM))
    heh2 = float(np.max(np.abs(H_POS @ E2 @ H_POS.T)))
    assert heh2 < 1e-9 * scale, heh2

    # Split epoch 3 at the UWB boundary and validate the actual propagation law
    # directly: dP_G = H Phi_bar E Phi_bar^T H^T.
    def half_predict(est, truth, sensor, config):
        n = config["imu_per_uwb"]
        imu = [sensor.generate_imu(truth.step()) for _ in range(n)]
        edges = sensor.generate_ranges(truth.snapshot())
        est.predict(imu, n * config["dt_imu"])
        return edges

    edges_dual = half_predict(dual, t1, s1, cfg)
    edges_ref = half_predict(ref, t2, s2, cfg)
    Phi = dual.get_phi_bar()
    predicted = H_POS @ Phi[i] @ E2 @ Phi[j].T @ H_POS.T
    actual = (H_POS @ get_block(ref.get_full_cov(), i, j, STATE_DIM) @ H_POS.T
              - get_block(dual.get_global_cov(), i, j, POS_DIM))
    prop_rel = float(np.linalg.norm(actual - predicted, "fro")
                     / max(np.linalg.norm(predicted, "fro"), 1e-30))
    assert prop_rel < 1e-5, prop_rel

    dual.update(edges_dual)
    ref.update(edges_ref)
    d3, c3, g3, x3 = blocks(dual, ref)
    assert g3 > g2, (g2, g3)

    # After the update the position-position block of the full covariance gap
    # must equal the actual drift of the maintained position layer, not zero.
    full_gap = H_POS @ (get_block(ref.get_full_cov(), i, j, STATE_DIM)
                        - get_block(dual.get_eval_cov_full(), i, j, STATE_DIM)) @ H_POS.T
    pos_gap = (get_block(ref.get_eval_cov(), i, j, POS_DIM)
               - get_block(dual.get_eval_cov(), i, j, POS_DIM))
    post_identity = float(np.max(np.abs(full_gap - pos_gap)))
    assert post_identity < 1e-12, post_identity

    # ------------------------------------------------------------------
    # 2) T10: K = 1 stays numerically equivalent to M15 for the whole run
    # ------------------------------------------------------------------
    one = default_config(); apply_entity_count(one, 1); one["sim_duration"] = 4.0
    d1e, t3, s3 = make(one, DualLayerESKF, seed=1)
    r1e, t4, s4 = make(one, CentralizedESKF, seed=1)
    for _ in range(int(round(one["sim_duration"] / one["sync_interval"]))):
        step(d1e, t3, s3, one)
        step(r1e, t4, s4, one)
    dP = float(np.max(np.abs(d1e.get_eval_cov_full() - r1e.get_full_cov())))
    dpos = float(np.max(np.abs(d1e.get_pos_estimate() - r1e.get_pos_estimate())))
    rel = dP / float(np.max(np.abs(r1e.get_full_cov())))
    assert rel < 1e-6 and dpos < 1e-6, (rel, dpos)

    # ------------------------------------------------------------------
    # 3) T9 marginal consistency and T22 dual-L distinguishability
    # ------------------------------------------------------------------
    cfg2 = default_config(); cfg2["sim_duration"] = 10.0
    est, truth, sensor = make(cfg2, DualLayerESKF, seed=0)
    gaps, lift_gaps, rows = [], [], []
    for _ in range(int(round(cfg2["sim_duration"] / cfg2["sync_interval"]))):
        step(est, truth, sensor, cfg2)
        gaps.append(est.last_marginal_gap)
        lift_gaps.append(est.last_lift_gap)
        rows.append(evaluate(truth.get_truth_states(), est.get_nominal_state(),
                             est.get_eval_cov(), cfg2,
                             full_cov=est.get_eval_cov_full()))
    assert max(gaps) < 1e-10, max(gaps)
    assert est.lifts_are_distinguishable(1e-9), est.last_lift_gap

    # ---- 4) accuracy smoke check.  A single trajectory is not a valid ANEES
    # acceptance test, so only finiteness and a broad accuracy ceiling are hard
    # assertions here; Monte-Carlo consistency is checked by the project gate.
    tail = rows[len(rows) // 2:]
    rmse = float(np.mean([r["pos_rmse"] for r in tail]))
    nees = float(np.mean([r["nees_pos"] for r in tail]))
    assert np.isfinite(rmse) and rmse < 1.0, rmse
    assert np.isfinite(nees), nees

    # Fixed epoch count: the truncation corner must shrink as T -> 0.
    corner = []
    for T in (0.005, 0.01, 0.02):
        ct = default_config(); ct["sync_interval"] = T
        ct["sim_duration"] = 40 * T
        validate_config(ct)
        de, td, sd = make(ct, DualLayerESKF, seed=2)
        ce, tc, sc = make(ct, CentralizedESKF, seed=2)
        for _ in range(40):
            step(de, td, sd, ct); step(ce, tc, sc, ct)
        denom = max(float(np.linalg.norm(ce.get_eval_cov(), "fro")), 1e-30)
        corner.append(float(np.linalg.norm(de.get_eval_cov()
                                           - ce.get_eval_cov(), "fro") / denom))
    assert corner[0] < corner[1] < corner[2], corner

    print("[M19 DualLayerESKF] OK  (epoch1 |dP|=%.1e, epoch2 |dP_G|=%.1e, "
          "|H E H^T|=%.1e, epoch3 propagation rel=%.1e, post identity=%.1e; "
          "K=1 rel %.1e; T9 gap %.1e; dual-L gap %.1e; T-corner %s)"
          % (max(d1, c1), g2, heh2, prop_rel, post_identity, rel,
             max(gaps), est.last_lift_gap,
             "/".join("%.2e" % x for x in corner)))
