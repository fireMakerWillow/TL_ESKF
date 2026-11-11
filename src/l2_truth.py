"""M13 TruthSimulator (L2 domain subsystem, truth side).

The single source of truth for K entities at the IMU rate.  Causality runs one
way only:

    M5 circular formation  ->  (a^n, omega^b) and f^b = R^T(a^n - g)
                           ->  M4 strapdown integration  ==>  true (R, v, p)
    M6 true bias random walk  --(consumed by M14 alone)-->  measurements

Four prohibitions, unchanged from v2 and unaffected by the v3 main-line
collapse, because they are what makes a NEES verdict mean anything at all:

    1. never construct or reference F, Phi or Q;
    2. never bias-compensate f^b or omega^b inside the integrator;
    3. never inject noise into (R, v, p);
    4. never read ``sigma_acc``, ``sigma_gyro``, ``sigma_range`` or any
       ``filter_sigma_*``.

Integrated, not sampled.  The truth is the strapdown INTEGRAL of the commanded
specific force, not the closed-form circle, even though M5 can evaluate the
circle analytically.  The reason is the T1 acceptance test: feeding noise-free,
bias-free measurements back through M4 must reproduce the truth to 1e-9.  That
identity holds only if the truth was produced by the same integrator that
consumes the measurements.  The gap between the integrated path and the analytic
circle is the mechanization discretisation error, which
:meth:`analytic_deviation` reports so it can never silently grow.

RNG streams.  The truth stream and the measurement stream are spawned from
independent ``SeedSequence`` children, so sweeping the measurement noise leaves
the trajectory bit-identical (T3), and sweeping any filter parameter leaves it
bit-identical as well (T2).
"""

import numpy as np

from config import validate_config
from l1_mechanization import mechanize
from l1_so3 import Exp, NominalState
from l1_trajectory import CircleFormationTrajectory, TrajectoryGenerator
from l1_true_bias import TrueBiasProcess


class TruthRecord:
    """One IMU-epoch snapshot of the truth for all entities."""

    __slots__ = ("t", "states", "f_b", "omega_b", "b_a", "b_g")

    def __init__(self, t, states, f_b, omega_b, b_a, b_g):
        self.t = float(t)
        self.states = states          # list[NominalState]: true (p, v, R)
        self.f_b = f_b                # (K,3) true specific force, body frame
        self.omega_b = omega_b        # (K,3) true angular rate, body frame
        self.b_a = b_a                # (K,3) true accelerometer bias
        self.b_g = b_g                # (K,3) true gyroscope bias

    def positions(self):
        return np.array([s.p for s in self.states])


class TruthSimulator:
    """Deterministic truth generator (given its seed)."""

    def __init__(self, config):
        validate_config(config)
        self.cfg = config
        self.K = int(config["num_entities"])
        self.g = np.asarray(config["gravity"], dtype=float)
        self.traj = TrajectoryGenerator(config)
        self.bias = TrueBiasProcess(config)
        self.states = None
        self._t = 0.0
        self._truth_rng = None

    # ------------------------------------------------------------------
    def reset(self, seed):
        """Initialise the truth stream and return the t = 0 record."""
        cfg = self.cfg
        self._truth_rng = np.random.default_rng(
            np.random.SeedSequence(int(seed)).spawn(1)[0])
        self.bias.reset(self._truth_rng)
        self.traj.reset(self._truth_rng)

        self.states = []
        for i in range(self.K):
            start = self.traj.initial_state(i)
            if start is None:                     # legacy generators (tests only)
                start = self._fallback_initial_state(i)
            start.b_a = self.bias.b_a[i].copy()
            start.b_g = self.bias.b_g[i].copy()
            self.states.append(start)
        self._t = 0.0
        return self.snapshot()

    def _fallback_initial_state(self, i):
        """Random start for legacy/random trajectories.

        Random/agile starts respect the soft x/y boundary and are distributed
        around the trajectory generator's per-entity reference heights instead
        of being concentrated around one common z plane.
        """
        cfg = self.cfg
        rng = self._truth_rng

        L = float(cfg["field_size"])
        margin = float(cfg.get("xy_boundary_margin", 0.0))
        if 0.0 < margin < 0.5 * L:
            x = rng.uniform(margin, L - margin)
            y = rng.uniform(margin, L - margin)
        else:
            x = rng.uniform(0.0, L)
            y = rng.uniform(0.0, L)

        if hasattr(self.traj, "reference_height"):
            z_ref = float(self.traj.reference_height(i))
            z = z_ref + rng.normal(0.0, float(cfg.get("z_initial_spread", 2.0)))
            z = float(np.clip(z, float(cfg.get("z_min", 0.0)),
                              float(cfg.get("z_max", 0.2 * L))))
        else:
            z = float(rng.uniform(0.0, 0.2 * L))
        p = np.array([x, y, z], dtype=float)

        if cfg["traj_type"] == "static":
            v = np.zeros(3)
        else:
            d = rng.standard_normal(3)
            d[2] *= float(cfg.get("random_vertical_scale", 0.20))
            v = float(cfg["speed"]) * d / (np.linalg.norm(d) + 1e-12)
        return NominalState(p=p, v=v, R=Exp(0.05 * rng.standard_normal(3)))

    # ------------------------------------------------------------------
    def step(self):
        """Advance every entity one IMU step and return the TruthRecord."""
        if self.states is None:
            raise RuntimeError("reset() must be called before step()")
        cfg = self.cfg
        dt = float(cfg["dt_imu"])
        f_b = np.zeros((self.K, 3))
        omega_b = np.zeros((self.K, 3))
        for i in range(self.K):
            st = self.states[i]
            _, w_b, fb = self.traj.command(i, st.R, st.v, st.p, self._t)
            f_b[i] = fb
            omega_b[i] = w_b
            # TRUE inputs: no bias subtraction, no noise, no Q anywhere
            self.states[i] = mechanize(st, fb, w_b, dt, self.g,
                                       cfg["nominal_int_order"])
        b_a, b_g = self.bias.step(dt)
        # the carried bias of a truth state IS the true bias, so that M10's
        # boxminus reports a meaningful bias error against an estimator
        for i in range(self.K):
            self.states[i].b_a = b_a[i].copy()
            self.states[i].b_g = b_g[i].copy()
        self._t += dt
        return TruthRecord(self._t, [s.copy() for s in self.states],
                           f_b, omega_b, b_a, b_g)

    def snapshot(self):
        """Current truth without advancing; the input columns are zeroed."""
        if self.states is None:
            raise RuntimeError("reset() must be called before snapshot()")
        b_a = np.array([s.b_a for s in self.states])
        b_g = np.array([s.b_g for s in self.states])
        return TruthRecord(self._t, [s.copy() for s in self.states],
                           np.zeros((self.K, 3)), np.zeros((self.K, 3)), b_a, b_g)

    # ------------------------------------------------------------------
    def get_truth_states(self):
        return [s.copy() for s in self.states]

    def get_truth_positions(self):
        return np.array([s.p for s in self.states])

    def time(self):
        return self._t

    def analytic_deviation(self):
        """Distance from the integrated truth to M5's closed-form circle.

        Only defined on the main line.  This is the mechanization discretisation
        error and nothing else -- it must stay small compared with the position
        RMSE the experiment reports, otherwise the "truth" carries an integrator
        artefact of the same size as the effect being measured.
        """
        if not isinstance(self.traj, CircleFormationTrajectory):
            return float("nan")
        return float(max(
            np.linalg.norm(self.states[i].p - self.traj.position(i, self._t))
            for i in range(self.K)))

    def pairwise_distance_spread(self, distances=None):
        """Range of every inter-entity distance seen so far (anti-degeneracy)."""
        d = np.asarray(distances, dtype=float)
        return float(np.min(np.ptp(d, axis=0))) if d.size else float("nan")


def run_truth(config, seed, epochs=None):
    """Convenience: run whole UWB epochs and return (simulator, records)."""
    sim = TruthSimulator(config)
    sim.reset(seed)
    n = int(config["imu_per_uwb"])
    if epochs is None:
        epochs = int(round(float(config["sim_duration"])
                           / float(config["sync_interval"])))
    records = []
    for _ in range(int(epochs)):
        records.append([sim.step() for _ in range(n)])
    return sim, records


if __name__ == "__main__":
    from config import default_config, validate_config, apply_entity_count

    cfg = default_config(); cfg["sim_duration"] = 4.0; validate_config(cfg)
    K = cfg["num_entities"]
    dt = cfg["dt_imu"]

    # ---- 1) the same seed reproduces the truth bit for bit ----
    def path(config, seed=0, n=120, entity=0):
        sim = TruthSimulator(config); sim.reset(seed)
        return np.array([sim.step().states[entity].p for _ in range(n)])

    assert np.array_equal(path(cfg), path(cfg))

    # ---- 2) the truth ignores every filter and measurement parameter (T2) ----
    base = path(cfg)
    for key, value in [("sigma_range", 99.0), ("sigma_acc", 9.0),
                       ("sigma_gyro", 1.0), ("filter_sigma_acc_bias", 9.0),
                       ("filter_sigma_gyro_bias", 1.0), ("phi_order", "van_loan"),
                       ("reg_epsilon", 1e-3), ("whiten_lambda", 1e-2)]:
        c = default_config(); c["sim_duration"] = 4.0; c[key] = value
        validate_config(c)
        assert np.array_equal(path(c), base), key

    # ---- 3) the truth starts exactly on the analytic circle ----
    sim = TruthSimulator(cfg); rec0 = sim.reset(0)
    traj = CircleFormationTrajectory(cfg)
    for i in range(K):
        assert np.allclose(rec0.states[i].p, traj.position(i, 0.0), atol=1e-12)
        assert np.allclose(rec0.states[i].v, traj.velocity(i, 0.0), atol=1e-12)
        assert abs(np.linalg.norm(rec0.states[i].v) - cfg["speed"]) < 1e-12
    assert rec0.t == 0.0 and np.allclose(rec0.f_b, 0.0)

    # ---- 4) it stays on it: speed, radius and the analytic deviation ----
    steps = int(round(cfg["sim_duration"] / dt))
    for _ in range(steps):
        rec = sim.step()
    deviation = sim.analytic_deviation()
    assert deviation < 1e-3 * cfg["circle_radius"], deviation
    for i in range(K):
        assert abs(np.linalg.norm(sim.states[i].v) - cfg["speed"]) < 1e-3
        radius = np.linalg.norm(sim.states[i].p[:2] - traj.center_of(i)[:2])
        assert abs(radius - cfg["circle_radius"]) < 1e-2, radius
    # ... and inside the field
    P = sim.get_truth_positions()
    assert P[:, :2].min() > -1.0 and P[:, :2].max() < cfg["field_size"] + 1.0

    # ---- 5) perfect-IMU reconstruction (the kernel of T1) ----
    c3 = default_config(); c3["sim_duration"] = 2.0
    for key in ("true_bias_init_std_acc", "true_bias_init_std_gyro",
                "true_bias_rw_acc", "true_bias_rw_gyro"):
        c3[key] = 0.0
    validate_config(c3)
    s3 = TruthSimulator(c3); init = s3.reset(3)
    rebuilt = [st.copy() for st in init.states]
    worst = 0.0
    for _ in range(int(round(c3["sim_duration"] / c3["dt_imu"]))):
        r = s3.step()
        for i in range(c3["num_entities"]):
            rebuilt[i] = mechanize(rebuilt[i], r.f_b[i], r.omega_b[i],
                                   c3["dt_imu"], c3["gravity"],
                                   c3["nominal_int_order"])
            worst = max(worst,
                        float(np.linalg.norm(rebuilt[i].p - r.states[i].p)),
                        float(np.linalg.norm(rebuilt[i].v - r.states[i].v)),
                        float(np.linalg.norm(rebuilt[i].R - r.states[i].R)))
    assert worst < 1e-9, worst

    # ---- 6) the true bias is carried on the truth states and drifts ----
    s4 = TruthSimulator(cfg); s4.reset(11)
    b0 = s4.snapshot().b_a.copy()
    for _ in range(400):
        rec = s4.step()
    assert np.allclose(rec.b_a, [st.b_a for st in rec.states])
    assert np.linalg.norm(rec.b_a - b0) > 0.0
    # the bias never leaks into the integrator: a huge bias must not move the
    # truth at all (prohibition 2)
    c5 = default_config(); c5["sim_duration"] = 1.0
    c5["true_bias_init_std_acc"] = 5.0; c5["true_bias_init_std_gyro"] = 0.5
    validate_config(c5)
    biased = path(c5, seed=0, n=100)
    c6 = default_config(); c6["sim_duration"] = 1.0
    c6["true_bias_init_std_acc"] = 0.0; c6["true_bias_init_std_gyro"] = 0.0
    c6["true_bias_rw_acc"] = 0.0; c6["true_bias_rw_gyro"] = 0.0
    validate_config(c6)
    assert np.array_equal(biased, path(c6, seed=0, n=100))

    # ---- 7) anti-degeneracy holds on the realised (integrated) truth ----
    epochs = int(round(cfg["sim_duration"] / cfg["sync_interval"]))
    sim7, records = run_truth(cfg, seed=2, epochs=epochs)
    assert len(records) == epochs and len(records[0]) == cfg["imu_per_uwb"]
    assert abs(sim7.time() - cfg["sim_duration"]) < 1e-9
    s7 = TruthSimulator(cfg); s7.reset(2)
    dists = []
    for _ in range(steps):
        s7.step()
        p = s7.get_truth_positions()
        dists.append([np.linalg.norm(p[i] - p[j])
                      for i in range(K) for j in range(i + 1, K)])
    spread = s7.pairwise_distance_spread(dists)
    assert spread > 0.0

    # ---- 8) the K = 10 extension and the legacy path both run ----
    big = default_config(); apply_entity_count(big, 10, (2, 6, 2))
    big["sim_duration"] = 1.0; validate_config(big)
    assert TruthSimulator(big).reset(0).positions().shape == (10, 3)
    legacy = default_config(); legacy["traj_type"] = "static"
    legacy["sim_duration"] = 1.0; validate_config(legacy)
    ls = TruthSimulator(legacy); ls.reset(5)
    p_static = ls.get_truth_positions().copy()
    for _ in range(200):
        ls.step()
    assert np.allclose(ls.get_truth_positions(), p_static, atol=1e-9)

    # ---- 9) stepping before reset is an error, not a silent zero state ----
    try:
        TruthSimulator(cfg).step()
        raise AssertionError("stepping before reset was allowed")
    except RuntimeError:
        pass

    print("[M13 TruthSimulator] OK  (perfect-IMU reconstruction %.1e m, "
          "deviation from the analytic circle %.2e m after %.0f s, "
          "distance spread %.1f m)"
          % (worst, deviation, cfg["sim_duration"], spread))
