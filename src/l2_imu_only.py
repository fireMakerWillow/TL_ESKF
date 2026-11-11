"""M25 ImuOnlyESKF (L2 Estimator subsystem) -- the no-cooperation floor.

Pure inertial dead reckoning.  Each entity runs an independent 15-dimensional
ESKF: nominal propagation through M4, error covariance through M7, and NOTHING
else.  ``update(range_edges)`` is a deliberate no-op -- the edge list is counted
and discarded -- so no range information ever enters the state, no error is ever
injected, and no covariance reset is ever performed (``dx == 0``, hence
``G == I`` exactly, not approximately).

Why the comparison needs it
---------------------------
1. **It isolates the contribution of cooperation.**  M15-M19 share this exact
   propagation and this exact Q; the only thing separating them from M25 is the
   UWB update.  Any RMSE gap is therefore attributable to ranging alone, with no
   confound from the trajectory, the IMU noise realisation or the attitude
   parameterisation.  The v3 self-test pins that claim rather than asserting it:
   M15 fed an empty edge list reproduces M25 in the nominal state to round-off
   and in the covariance to round-off.

2. **It is the consistency floor, and therefore a probe on the evaluation
   path.**  Under independent initial errors and independent IMU noise the true
   cross-entity correlation of a dead-reckoning ensemble is exactly zero, so the
   block-diagonal joint covariance reported here is not an approximation -- it
   is the exact structure.  This is the sharp contrast with M16, whose joint
   covariance has the SAME block-diagonal shape as a defining approximation.
   Consequently M25's joint position ANEES must land in band by construction:
   if it does not, the defect is in M10's error construction or in the M13/M14
   truth-measurement separation, not in any filter.  That sits one level below
   T12, which only checks that the gold standard is trustworthy.

3. **It supplies the divergence envelope of the (v, T) sweep.**  The dual-layer
   truncation can only matter where ranging actually constrains the solution;
   M25 is the curve everything else is measured against as T grows.

Communication is zero in BOTH ledger channels, and that is a modelling
statement, not an accounting shortcut: M25 exchanges no estimates (there is
nothing to exchange) and it also consumes no ranges, so the measurement traffic
that every cooperative method pays for is not incurred either.  M25 is the only
member of the family for which ``C_measurement`` is zero.

Plot note (design [M25]): position RMSE grows like O(t^{3/2}) to O(t^2), so the
vertical axis must be logarithmic or the five cooperative methods collapse onto
one line at the bottom of the figure.

Calls: M4 / M7 (through :class:`CooperativeESKF`), M1, M11.
Never calls: M3, M8, M9 -- there is no measurement model in this file at all.
"""

import numpy as np

from config import IDX_P, POS_DIM, STATE_DIM
from l1_linalg import symmetrize
from l2_estimator_base import CooperativeESKF


class ImuOnlyESKF(CooperativeESKF):
    """Independent per-entity dead reckoning; range edges are counted, not used."""

    name = "imu_only"
    evaluated_channels = ("pos", "vel", "att", "acc_bias", "gyro_bias")
    provides_full_cov = True

    # ------------------------------------------------------------------
    def reset(self, truth_states, seed=None):
        self.nominal = self.init_nominal_from_truth(truth_states, seed)
        P0 = self.initial_local_cov()
        self.P_local = [P0.copy() for _ in range(self.K)]
        self.ignored_edges = 0
        self.flops = 0
        self.epochs = 0
        self.last_innovations = []
        return self

    # ------------------------------------------------------------------
    def predict(self, imu_seq, dt=None):
        """Propagate every entity independently over one UWB interval."""
        self.validate_predict_input(imu_seq, dt)
        for i in range(self.K):
            nom, P_i, _ = self.propagate_entity(
                self.nominal[i], self.P_local[i], self.entity_imu_seq(imu_seq, i))
            self.nominal[i] = nom
            self.P_local[i] = symmetrize(P_i)

    # ------------------------------------------------------------------
    def update(self, range_edges):
        """No-op: count the offered edges and return the zero correction.

        The zero vector is returned rather than ``None`` so that M20 can treat
        every estimator's ``update`` return value uniformly, and so that "M25
        injected nothing" is a checkable fact instead of a docstring claim.

        No injection and no reset are performed, and that is not a shortcut:
        the accumulated error state is identically zero, so ``boxplus`` would
        be the identity and ``G`` would be ``I``.  Performing them anyway would
        leave round-off in the covariance and cost the cross blocks their exact
        zero.
        """
        if self.nominal is None:
            raise RuntimeError("reset() must be called before update()")
        self.ignored_edges += len(range_edges)
        self.last_innovations = []
        self.epochs += 1
        # both channels are zero by definition -- see the module docstring
        self.charge_epoch(measurement=0, estimation=0)
        return np.zeros(self.K * STATE_DIM)

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------
    def get_eval_cov(self):
        """Joint 3K position covariance; the zero cross blocks are EXACT."""
        return self.block_diag_position([P[IDX_P, IDX_P] for P in self.P_local])

    def get_eval_cov_full(self):
        return self.block_diag_full(self.P_local)

    def get_full_cov(self):
        return self.get_eval_cov_full()

    def get_local_covariances(self):
        return [P.copy() for P in self.P_local]


if __name__ == "__main__":
    from config import apply_entity_count, default_config, validate_config
    from l1_metrics import (anees_bounds, evaluate, joint_position_nees,
                            single_run_bounds)
    from l1_so3 import boxminus
    from l2_centralized_eskf import CentralizedESKF
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    def make(cfg, cls, seed=0):
        truth = TruthSimulator(cfg); truth.reset(seed)
        sensor = SensorSimulator(cfg); sensor.reset(seed + 5000)
        est = cls(cfg); est.reset(truth.get_truth_states(), seed=seed)
        return est, truth, sensor

    def step(est, truth, sensor, cfg, edges=True):
        """One epoch.  The ranges are ALWAYS generated even when they are not

        handed to the estimator, so that the measurement RNG is consumed
        identically in both arms of the comparison; otherwise "same seed" would
        not mean "same noise realisation" and tests 1 and 3 would compare two
        different worlds.
        """
        n = cfg["imu_per_uwb"]
        imu = [sensor.generate_imu(truth.step()) for _ in range(n)]
        drawn = sensor.generate_ranges(truth.snapshot())
        est.predict(imu, n * cfg["dt_imu"])
        est.update(drawn if edges else [])

    # ------------------------------------------------------------------
    # 1) the update really is a no-op: edges change nothing at all
    # ------------------------------------------------------------------
    cfg = default_config(); apply_entity_count(cfg, 4); cfg["sim_duration"] = 3.0
    validate_config(cfg)
    steps = int(round(cfg["sim_duration"] / cfg["sync_interval"]))

    fed, t1, s1 = make(cfg, ImuOnlyESKF, seed=3)
    starved, t2, s2 = make(cfg, ImuOnlyESKF, seed=3)
    for _ in range(steps):
        step(fed, t1, s1, cfg, edges=True)
        step(starved, t2, s2, cfg, edges=False)
    assert fed.ignored_edges > 0 and starved.ignored_edges == 0
    assert all(np.array_equal(a.p, b.p) and np.array_equal(a.R, b.R)
               for a, b in zip(fed.get_nominal_state(), starved.get_nominal_state()))
    assert np.array_equal(fed.get_full_cov(), starved.get_full_cov())
    assert np.array_equal(fed.update([]), np.zeros(cfg["num_entities"] * STATE_DIM))

    # ------------------------------------------------------------------
    # 2) the cross-entity position blocks are EXACTLY zero (not just small)
    # ------------------------------------------------------------------
    P_pos = fed.get_eval_cov()
    K = cfg["num_entities"]
    for i in range(K):
        for j in range(K):
            if i != j:
                blk = P_pos[i * POS_DIM:(i + 1) * POS_DIM,
                            j * POS_DIM:(j + 1) * POS_DIM]
                assert np.count_nonzero(blk) == 0, (i, j)

    # ------------------------------------------------------------------
    # 3) M15 with an empty edge list IS M25 (the "cooperation only" claim)
    # ------------------------------------------------------------------
    a, t3, s3 = make(cfg, ImuOnlyESKF, seed=11)
    b, t4, s4 = make(cfg, CentralizedESKF, seed=11)
    for _ in range(steps):
        step(a, t3, s3, cfg, edges=False)
        step(b, t4, s4, cfg, edges=False)
    nom_gap = max(float(np.max(np.abs(boxminus(x, y))))
                  for x, y in zip(a.get_nominal_state(), b.get_nominal_state()))
    scale = float(np.max(np.abs(b.get_full_cov())))
    cov_gap = float(np.max(np.abs(a.get_full_cov() - b.get_full_cov())))
    # NOT bit for bit, and the reason is worth recording: M15 still walks its
    # injection path every epoch, and ``boxplus`` re-orthonormalises R through
    # ``project_so3`` even for dx = 0, which perturbs the last bits of the
    # attitude.  M25 never injects, so it never re-orthonormalises.  The gap is
    # therefore pure round-off accumulated over the run (~1e-13 after 30
    # epochs), and the positions stay bit-identical for the first epochs.
    assert nom_gap < 1e-11, nom_gap
    assert cov_gap < 1e-12 * scale, (cov_gap, scale)

    # ------------------------------------------------------------------
    # 4) dead reckoning only ever loses information: tr P is non-decreasing
    # ------------------------------------------------------------------
    mono, t5, s5 = make(cfg, ImuOnlyESKF, seed=2)
    traces = [float(np.trace(mono.get_eval_cov()))]
    for _ in range(steps):
        step(mono, t5, s5, cfg)
        traces.append(float(np.trace(mono.get_eval_cov())))
    assert all(traces[k + 1] > traces[k] for k in range(len(traces) - 1))

    # ------------------------------------------------------------------
    # 5) the consistency floor: joint position ANEES in band over seeds.
    #    This is the probe on M10 / M13 / M14 described in the docstring --
    #    a failure here is NOT a filter bug.
    # ------------------------------------------------------------------
    dof = POS_DIM * K
    samples = []
    for seed in range(24):
        est, truth, sensor = make(cfg, ImuOnlyESKF, seed=100 + seed)
        for _ in range(steps):
            step(est, truth, sensor, cfg)
        err = np.concatenate([boxminus(t, n)[IDX_P] for t, n
                              in zip(truth.get_truth_states(),
                                     est.get_nominal_state())])
        samples.append(joint_position_nees(err, est.get_eval_cov()))
    anees = float(np.mean(samples))
    lo, hi = anees_bounds(dof, len(samples), 0.99)
    assert lo < anees < hi, (anees, lo, hi, dof)

    # ------------------------------------------------------------------
    # 6) zero communication in both ledger channels
    # ------------------------------------------------------------------
    # ``mono`` ran exactly ``steps`` epochs on a full edge list; ``fed`` took
    # one extra no-op update in test 1, so the epoch count is read off ``mono``.
    summary = mono.get_comm_summary()
    assert summary["C_measurement_bytes"] == 0
    assert summary["C_estimation_bytes"] == 0
    assert summary["epochs"] == steps

    row = evaluate(t5.get_truth_states(), mono.get_nominal_state(),
                   mono.get_eval_cov(), cfg, full_cov=mono.get_eval_cov_full(),
                   channels=ImuOnlyESKF.evaluated_channels)
    slo, shi = single_run_bounds(dof, cfg["nees_confidence"])

    print("[M25 ImuOnlyESKF] OK  (%d edges offered and ignored; M15-with-no-"
          "ranges gap: nominal %.1e, cov %.1e; tr P_pos %.3e -> %.3e; "
          "ANEES %.2f in [%.2f, %.2f] over %d seeds, dof %d; RMSE %.3f m, "
          "single-run NEES %.1f in [%.1f, %.1f]; C_total %d B)"
          % (fed.ignored_edges, nom_gap, cov_gap, traces[0], traces[-1],
             anees, lo, hi, len(samples), dof, row["pos_rmse"],
             row["nees_pos"], slo, shi, summary["C_total_bytes"]))
