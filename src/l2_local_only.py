"""M16 DCL-style local-only ESKF.

This module implements the Decentralized Cooperative Localization (DCL) update
used by Khosravi et al. while preserving the repository's public estimator key
``local_only`` for backward compatibility.

For a range measurement between entities i and j, the local filters do not
store a historical cross-covariance.  Instead, each receiver treats the other
entity's projected position uncertainty as additional measurement noise:

    R_eff,i = u^T P_j,pp u + R
    S_i     = u^T P_i,pp u + R_eff,i

and symmetrically for j.  The repository's UWB edge is undirected and available
to both endpoints, so one accepted edge is applied in both directions.  This is
the direct symmetric adaptation of the paper's directed ``observer i -> robot
j`` equations to the simulator's pairwise range model.

DCL's measurement stride is part of the estimator, not just a benchmark-side
subsampler.  ``dcl_measurement_stride = 3`` processes every third inter-entity
measurement using one persistent 1-based counter; anchor measurements are local
absolute references and are always processed.  Skipped relative measurements
are neither fused nor charged as communication/estimation traffic.

State and storage remain local: K independent 15x15 covariance matrices are
kept and the reported joint covariance is block diagonal.  No cross-entity
covariance object is allocated.
"""

from __future__ import annotations

import numpy as np

from config import IDX_P, POS_DIM, STATE_DIM
from l1_linalg import symmetrize
from l1_resource import flops_scalar_update, symmetric_scalars
from l2_estimator_base import CooperativeESKF


class LocalOnlyESKF(CooperativeESKF):
    """DCL-compatible local filters with effective-noise fusion and stride."""

    name = "local_only"
    display_name = "DCL"

    def __init__(self, config):
        super().__init__(config)
        self.measurement_stride = int(config.get("dcl_measurement_stride", 1))
        if self.measurement_stride < 1:
            raise ValueError("dcl_measurement_stride must be a positive integer")
        self.P_local = None
        self.cross = None  # compatibility probe: deliberately never populated
        self.relative_measurement_counter = 0
        self.processed_relative_measurements = 0
        self.skipped_relative_measurements = 0
        self.last_stride_stats = {
            "stride": self.measurement_stride,
            "relative_seen": 0,
            "relative_processed": 0,
            "relative_skipped": 0,
            "anchor_processed": 0,
        }

    # ------------------------------------------------------------------
    def reset(self, truth_states, seed=None):
        self.nominal = self.init_nominal_from_truth(truth_states, seed)
        P0 = self.initial_local_cov()
        self.P_local = [P0.copy() for _ in range(self.K)]
        self.cross = None
        self.relative_measurement_counter = 0
        self.processed_relative_measurements = 0
        self.skipped_relative_measurements = 0
        self.last_stride_stats = {
            "stride": self.measurement_stride,
            "relative_seen": 0,
            "relative_processed": 0,
            "relative_skipped": 0,
            "anchor_processed": 0,
        }
        self.flops = 0
        self.epochs = 0
        self.last_innovations = []
        return self

    # ------------------------------------------------------------------
    def predict(self, imu_seq, dt=None):
        """Independent local propagation; no cross-covariance communication."""
        self.validate_predict_input(imu_seq, dt)
        for i in range(self.K):
            nom, P_i, _ = self.propagate_entity(
                self.nominal[i], self.P_local[i],
                self.entity_imu_seq(imu_seq, i))
            self.nominal[i] = nom
            self.P_local[i] = P_i

    # ------------------------------------------------------------------
    @staticmethod
    def _projected_variance(P_i, u):
        """Return u^T P_pp u for one local covariance."""
        return float(u @ P_i[IDX_P, IDX_P] @ u)

    @staticmethod
    def _local_posterior(P_i, h, R_eff):
        """One scalar local Kalman covariance update.

        ``h`` is the receiver's position Jacobian (+u or -u).  The returned
        gain column and posterior implement equations (11)-(13) of the paper in
        the repository's 15-state error coordinates.
        """
        a = P_i[:, IDX_P] @ h
        S = float(h @ a[IDX_P]) + float(R_eff)
        if not np.isfinite(S) or S <= 0.0:
            raise np.linalg.LinAlgError(
                f"non-positive DCL innovation variance S={S}")
        P_post = symmetrize(P_i - np.outer(a, a) / S)
        return a, S, P_post

    def _accept_relative_measurement(self):
        """Persistent one-based stride scheduler: accept s, 2s, 3s, ..."""
        self.relative_measurement_counter += 1
        accepted = (self.relative_measurement_counter
                    % self.measurement_stride == 0)
        if accepted:
            self.processed_relative_measurements += 1
        else:
            self.skipped_relative_measurements += 1
        return accepted

    def update(self, range_edges):
        """Apply all anchor updates and every ``stride``-th relative update.

        The nominal positions are frozen for the complete UWB epoch, matching
        the common estimator cadence.  For an accepted pair edge, both local
        posteriors are computed from the same pre-edge covariances so the result
        is independent of which endpoint is evaluated first.
        """
        if self.nominal is None:
            raise RuntimeError("reset() must be called before update()")

        positions = self.positions()
        dx = [np.zeros(STATE_DIM) for _ in range(self.K)]
        self.last_innovations = []
        anchor_processed = 0
        relative_seen = 0
        relative_processed = 0

        for e in range_edges:
            if not e.is_anchor:
                relative_seen += 1
                if not self._accept_relative_measurement():
                    continue
                relative_processed += 1

            _, u, y = self.edge_geometry(e, positions)
            if not np.any(u):
                continue

            if e.is_anchor:
                i = e.i
                resid = y - float(u @ dx[i][IDX_P])
                a_i, S_i, P_i_post = self._local_posterior(
                    self.P_local[i], u, self.R_range)
                dx[i] = dx[i] + a_i * (resid / S_i)
                self.P_local[i] = P_i_post
                anchor_processed += 1
                self.flops += flops_scalar_update(STATE_DIM)
                self.last_innovations.append({
                    "i": i, "j": -1, "u": u.copy(), "S": S_i, "y": y,
                    "R_eff": self.R_range, "stride_accepted": True,
                })
                continue

            # Symmetric adaptation of the paper's directed DCL equations.
            i, j = e.i, e.j
            P_i_prior = self.P_local[i]
            P_j_prior = self.P_local[j]
            q_i = self._projected_variance(P_i_prior, u)
            q_j = self._projected_variance(P_j_prior, u)
            R_eff_i = self.R_range + q_j
            R_eff_j = self.R_range + q_i

            # h_i = +u, h_j = -u for h = ||p_i - p_j||.
            a_i, S_i, P_i_post = self._local_posterior(
                P_i_prior, u, R_eff_i)
            a_j, S_j, P_j_post = self._local_posterior(
                P_j_prior, -u, R_eff_j)
            if not np.isclose(S_i, S_j, rtol=1e-11, atol=1e-13):
                raise AssertionError(
                    f"symmetric DCL innovation variances disagree: {S_i} vs {S_j}")

            resid = (y - float(u @ dx[i][IDX_P])
                     + float(u @ dx[j][IDX_P]))
            dx[i] = dx[i] + a_i * (resid / S_i)
            dx[j] = dx[j] + a_j * (resid / S_j)
            self.P_local[i] = P_i_post
            self.P_local[j] = P_j_post
            self.flops += 2 * flops_scalar_update(STATE_DIM)
            self.last_innovations.append({
                "i": i, "j": j, "u": u.copy(), "S": S_i, "y": y,
                "R_eff_i": R_eff_i, "R_eff_j": R_eff_j,
                "stride_accepted": True,
            })

        self.P_local, _ = self.apply_injection_and_reset(
            np.concatenate(dx), self.P_local)
        self.epochs += 1

        # One scalar range is communicated for each fused measurement.  Each
        # accepted pair also exchanges (p, P_pp) in both directions: 2*(3+6)=18.
        measurement_scalars = anchor_processed + relative_processed
        estimation_per_direction = POS_DIM + symmetric_scalars(POS_DIM)
        estimation_scalars = 2 * estimation_per_direction * relative_processed
        self.charge_epoch(measurement=measurement_scalars,
                          estimation=estimation_scalars)

        self.last_stride_stats = {
            "stride": self.measurement_stride,
            "relative_seen": relative_seen,
            "relative_processed": relative_processed,
            "relative_skipped": relative_seen - relative_processed,
            "anchor_processed": anchor_processed,
        }
        return np.concatenate(dx)

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------
    def get_eval_cov(self):
        """Block-diagonal 3K position covariance reported by the local filters."""
        return self.block_diag_position([P[IDX_P, IDX_P] for P in self.P_local])

    def get_eval_cov_full(self):
        return self.block_diag_full(self.P_local)

    def get_full_cov(self):
        return self.get_eval_cov_full()

    def get_local_covariances(self):
        return [P.copy() for P in self.P_local]

    def get_stride_summary(self):
        """Return cumulative and last-epoch stride diagnostics."""
        return {
            **self.last_stride_stats,
            "relative_counter": self.relative_measurement_counter,
            "relative_processed_total": self.processed_relative_measurements,
            "relative_skipped_total": self.skipped_relative_measurements,
        }

    def has_cross_objects(self):
        return False


# Clear alias for external code that wants the algorithm name while the
# repository continues to use the stable estimator key/file name local_only.
DCLESKF = LocalOnlyESKF


if __name__ == "__main__":
    from config import apply_entity_count, default_config, validate_config
    from l1_geometry import Edge
    from l2_sensor import RangeEdge, SensorSimulator
    from l2_truth import TruthSimulator

    # ---- 1) equation-level check against the paper's effective-noise form ----
    rng = np.random.default_rng(4)
    M = rng.standard_normal((STATE_DIM, STATE_DIM))
    Pi = symmetrize(M @ M.T) + np.eye(STATE_DIM)
    M = rng.standard_normal((STATE_DIM, STATE_DIM))
    Pj = symmetrize(M @ M.T) + np.eye(STATE_DIM)
    u = rng.standard_normal(POS_DIM)
    u /= np.linalg.norm(u)
    R = 0.04
    qj = float(u @ Pj[IDX_P, IDX_P] @ u)
    a, S, Pp = LocalOnlyESKF._local_posterior(Pi, u, qj + R)
    a_ref = Pi[:, IDX_P] @ u
    S_ref = float(u @ Pi[IDX_P, IDX_P] @ u) + qj + R
    assert np.allclose(a, a_ref)
    assert abs(S - S_ref) < 1e-11
    assert np.allclose(Pp, Pi - np.outer(a_ref, a_ref) / S_ref)

    # ---- 2) persistent stride: with one pair per epoch, only epoch 3 fires ----
    cfg = default_config()
    apply_entity_count(cfg, 2, (1, 1, 0))
    cfg["sim_duration"] = 0.3
    cfg["separation_check"] = False
    cfg["dcl_measurement_stride"] = 3
    validate_config(cfg)
    truth = TruthSimulator(cfg); truth.reset(0)
    est = LocalOnlyESKF(cfg); est.reset(truth.get_truth_states(), seed=0)
    p = est.positions()
    rho = float(np.linalg.norm(p[0] - p[1]))
    edge = RangeEdge(0, 1, False, -1, None, rho + 0.2, rho, True)
    traces = []
    for _ in range(3):
        est.update([edge])
        traces.append(float(np.trace(est.get_eval_cov())))
    assert traces[0] == traces[1] and traces[2] < traces[1], traces
    ss = est.get_stride_summary()
    assert ss["relative_processed_total"] == 1
    assert ss["relative_skipped_total"] == 2
    comm = est.get_comm_summary()
    assert comm["C_measurement_scalars"] == 1
    assert comm["C_estimation_scalars"] == 18
    assert not est.has_cross_objects() and est.cross is None

    # ---- 3) short end-to-end smoke run remains finite and PSD ----
    cfg2 = default_config(); cfg2["sim_duration"] = 0.5
    cfg2["separation_check"] = False
    validate_config(cfg2)
    truth2 = TruthSimulator(cfg2); truth2.reset(2)
    sensor2 = SensorSimulator(cfg2); sensor2.reset(5002)
    dcl = LocalOnlyESKF(cfg2); dcl.reset(truth2.get_truth_states(), seed=2)
    n = cfg2["imu_per_uwb"]
    for _ in range(int(round(cfg2["sim_duration"] / cfg2["sync_interval"]))):
        imu = [sensor2.generate_imu(truth2.step()) for _ in range(n)]
        dcl.predict(imu, n * cfg2["dt_imu"])
        dcl.update(sensor2.generate_ranges(truth2.snapshot()))
    P = dcl.get_eval_cov_full()
    assert np.all(np.isfinite(P))
    assert float(np.min(np.linalg.eigvalsh(symmetrize(P)))) > -1e-8

    print("[M16 DCL/LocalOnlyESKF] OK  (stride=%d; processed=%d skipped=%d; "
          "C_est=%d scalars; no cross covariance)"
          % (dcl.measurement_stride,
             dcl.processed_relative_measurements,
             dcl.skipped_relative_measurements,
             dcl.get_comm_summary()["C_estimation_scalars"]))
