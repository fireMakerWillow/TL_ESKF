"""M26 UwbOnlyKF -- dense 3K position random-walk EKF.

The state is the stacked position vector ``p in R^(3K)``.  There is no velocity,
attitude, or bias state and therefore this module is an ordinary EKF, not an
ESKF.  Prediction follows the design contract

    p^- = p^+,        P^- = P^+ + q_p T I,

where ``q_p`` is ``uwb_only_pos_rw_psd``; when that configuration entry is
``None`` the operating-point value ``speed**2 * sync_interval`` is used.
"""

from __future__ import annotations

import numpy as np

from config import POS_DIM
from l1_linalg import joseph_update, psd_project, symmetrize
from l1_range_model import predict_range, range_row_position
from l1_so3 import NominalState
from l2_estimator_base import CooperativeESKF

_MIN_RANGE = 1e-9


class UwbOnlyKF(CooperativeESKF):
    name = "uwb_only"
    evaluated_channels = ("pos",)
    provides_full_cov = False

    def reset(self, truth_states, seed=None):
        self._validate_truth_states(truth_states)
        seeded = self.init_nominal_from_truth(truth_states, seed)
        self._R_frozen = [s.R.copy() for s in seeded]
        self.p = np.concatenate([s.p for s in seeded])
        self.nominal = [NominalState(
            p=s.p.copy(), v=np.zeros(3), R=s.R.copy(),
            b_a=np.zeros(3), b_g=np.zeros(3)) for s in seeded]
        self.P = (float(self.cfg["init_pos_std"]) ** 2
                  * np.eye(POS_DIM * self.K))
        self.flops = 0
        self.epochs = 0
        self.last_edge_count = 0
        self.last_innovations = []
        self.effective_q_p = self._resolve_q_p()
        return self

    def _validate_truth_states(self, truth_states):
        if len(truth_states) != self.K:
            raise ValueError(f"expected {self.K} truth states, got {len(truth_states)}")

    def _resolve_q_p(self):
        value = self.cfg.get("uwb_only_pos_rw_psd")
        if value is None:
            value = float(self.cfg["speed"]) ** 2 * float(self.cfg["sync_interval"])
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("uwb_only_pos_rw_psd must be non-negative and finite")
        return value

    def positions(self):
        return [self.p[i * POS_DIM:(i + 1) * POS_DIM] for i in range(self.K)]

    def predict(self, imu_seq, dt=None):
        self.validate_predict_input(imu_seq, dt)
        T = len(imu_seq) * self.dt
        self.effective_q_p = self._resolve_q_p()
        self.P = symmetrize(
            self.P + self.effective_q_p * T * np.eye(POS_DIM * self.K))
        self.flops += POS_DIM * self.K
        return self.p.copy()

    def update(self, range_edges):
        if self.nominal is None:
            raise RuntimeError("reset() must be called before update()")
        n = POS_DIM * self.K
        dp = np.zeros(n)
        positions = [x.copy() for x in self.positions()]
        self.last_innovations = []
        applied = 0

        for rec in range_edges:
            d_hat, u = predict_range(rec, positions, self.anchors)
            if d_hat < _MIN_RANGE or not np.any(u):
                continue
            h = range_row_position(rec, positions, self.anchors, self.K)
            a = self.P @ h
            S = float(h @ a) + self.R_range
            if not np.isfinite(S) or S <= 0.0:
                raise np.linalg.LinAlgError(
                    f"non-positive innovation variance S={S} on {rec}")
            gain = a / S
            innovation = float(rec.meas) - d_hat - float(h @ dp)
            dp += gain * innovation
            if self.joseph:
                self.P = joseph_update(self.P, h, self.R_range, gain)
            else:
                self.P = symmetrize(self.P - np.outer(gain, a))
            self.last_innovations.append({
                "i": rec.i, "j": (-1 if rec.is_anchor else rec.j),
                "u": u.copy(), "S": S, "y": float(rec.meas) - d_hat})
            self.flops += 2 * n * n
            applied += 1

        self.p += dp
        self.P = psd_project(symmetrize(self.P), floor=0.0)
        self.last_edge_count = applied
        self.epochs += 1
        self.charge_epoch(measurement=len(range_edges))
        self._sync_nominal()
        return dp

    def _sync_nominal(self):
        for i in range(self.K):
            s = slice(i * POS_DIM, (i + 1) * POS_DIM)
            self.nominal[i] = NominalState(
                p=self.p[s].copy(), v=np.zeros(3), R=self._R_frozen[i].copy(),
                b_a=np.zeros(3), b_g=np.zeros(3))

    def get_eval_cov(self):
        return self.P.copy()

    def get_reported_cov(self):
        return self.P.copy()

    def get_local_covariances(self):
        return [self.P[i * POS_DIM:(i + 1) * POS_DIM,
                       i * POS_DIM:(i + 1) * POS_DIM].copy()
                for i in range(self.K)]


if __name__ == "__main__":
    from config import apply_entity_count, default_config, validate_config
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    cfg = default_config(); apply_entity_count(cfg, 3, (1, 1, 1))
    cfg["sim_duration"] = 1.0
    validate_config(cfg)
    truth = TruthSimulator(cfg); truth.reset(0)
    sensor = SensorSimulator(cfg); sensor.reset(5000)
    est = UwbOnlyKF(cfg); est.reset(truth.get_truth_states(), seed=0)
    p_before = est.p.copy(); P_before = est.P.copy()
    imu = [sensor.generate_imu(truth.step())
           for _ in range(cfg["imu_per_uwb"])]
    est.predict(imu, cfg["dt_uwb"])
    assert np.array_equal(est.p, p_before)
    assert np.all(np.diag(est.P) >= np.diag(P_before))
    est.update(sensor.generate_ranges(truth.snapshot()))
    assert est.P.shape == (3 * cfg["num_entities"],) * 2
    assert np.allclose(est.P, est.P.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(est.P)) >= -1e-10
    assert np.linalg.norm(est.P[0:3, 3:6]) > 0.0
    assert not hasattr(est, "get_eval_cov_full")
    assert not hasattr(est, "get_full_cov")
    assert est.evaluated_channels == ("pos",)
    print("[M26 UwbOnlyKF] OK  (dense 3K random walk, q_p=%.3g)" %
          est.effective_q_p)
