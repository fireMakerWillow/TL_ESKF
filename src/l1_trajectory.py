"""M5 CircleFormationTrajectory (L1 basic capability) -- REWRITTEN for v3.

Designs the *true* motion of each entity, independent of any estimation model.
The v3 main line is a circular formation of radius R: entity i orbits the centre
o_i = o_0 + delta o_i with phase phi_i(t) = phi_0i + s_i omega t, where
omega = v / R and s_i = +-1 is the rotation sense.  Everything is analytic and
deterministic, so every (v, T) grid point has an exactly reproducible truth.

    p_i(t)   = o_i + R [cos phi_i, sin phi_i, 0]
    v_i(t)   = v   s_i [-sin phi_i, cos phi_i, 0]        (|v_i| = v)
    a_i^n(t) = -(v^2/R) [cos phi_i, sin phi_i, 0]        (|a_i| = v^2/R)
    R_i(t)   : body x along the velocity, body z up
    omega^b  = [0, 0, s_i omega]
    f^b      = R^T (a^n - g)      (exact inverse of M4's a^n = R f^b + g)

Anti-degeneracy (plan section 5.6): a rigidly co-phased formation keeps every
inter-entity distance constant and destroys the cooperative geometry.  Distinct
phases, mixed rotation senses and per-entity centre offsets each break it; the
defaults enable all three, and :func:`min_pairwise_distance_spread` provides the
pre-flight check used by M20.

The single motion-intensity knob is v, which simultaneously sets the attitude
rate omega = v/R and the specific force a_c = v^2/R.  The attitude-coupling
channel therefore obeys the closed-form prediction

    ||Phi_bar_i[0:3, 6:9]|| ~ 0.5 a_c T^2 = v^2 T^2 / (2 R),

which is the analytic reference curve of the M22 (v, T) sweep.

No process noise is ever injected here.
"""

import numpy as np

from config import circle_parameters, circle_omega, validate_config
from l1_so3 import NominalState

_LEGACY_TYPES = {"static", "const_vel", "circle", "figure8",
                 "random_accel", "agile"}


class CircleFormationTrajectory:
    """Analytic circular-formation command generator (v3 main line)."""

    def __init__(self, config):
        self.cfg = config
        self.K = int(config["num_entities"])
        self.g = np.asarray(config["gravity"], dtype=float)
        self.R = float(config["circle_radius"])
        self.center = np.asarray(config["circle_center"], dtype=float)
        self.phases, self.directions, self.offsets = circle_parameters(config)
        self.omega = circle_omega(config)
        self.speed = float(config["speed"])

    # ------------------------------------------------------------------
    # analytic kinematics
    # ------------------------------------------------------------------
    def _phase(self, i, t):
        return self.phases[i] + self.directions[i] * self.omega * float(t)

    def center_of(self, i):
        return self.center + self.offsets[i]

    def position(self, i, t):
        phi = self._phase(i, t)
        return self.center_of(i) + self.R * np.array(
            [np.cos(phi), np.sin(phi), 0.0])

    def velocity(self, i, t):
        phi = self._phase(i, t)
        s = self.directions[i]
        return self.speed * s * np.array([-np.sin(phi), np.cos(phi), 0.0])

    def acceleration(self, i, t):
        phi = self._phase(i, t)
        return -(self.speed ** 2 / self.R) * np.array(
            [np.cos(phi), np.sin(phi), 0.0])

    def attitude(self, i, t):
        """Body x along velocity, body z up; exact rotation matrix."""
        v = self.velocity(i, t)
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:                     # v = 0 -> identity heading
            x_b = np.array([1.0, 0.0, 0.0])
        else:
            x_b = v / norm
        z_b = np.array([0.0, 0.0, 1.0])
        y_b = np.cross(z_b, x_b)
        return np.column_stack([x_b, y_b, z_b])

    def angular_rate(self, i, t):
        """Body-frame rate: pure yaw at s_i * omega (body z = navigation z)."""
        return np.array([0.0, 0.0, self.directions[i] * self.omega])

    def initial_state(self, i):
        """Exact analytic initial (p, v, R) so the truth starts on the circle."""
        return NominalState(p=self.position(i, 0.0),
                            v=self.velocity(i, 0.0),
                            R=self.attitude(i, 0.0))

    # ------------------------------------------------------------------
    # M13 interface
    # ------------------------------------------------------------------
    def reset(self, rng):
        """Bind the truth RNG (unused: the trajectory is deterministic)."""
        self._rng = rng

    def command(self, i, R_i, v_i, p_i, t):
        """Return the TRUE (a^n, omega^b, f^b) command of entity i at time t.

        ``a^n`` and ``omega^b`` are analytic functions of t only.  The specific
        force is built with the *integrated* attitude ``R_i`` so that M4's
        a^n = R f^b + g reproduces the designed acceleration exactly.
        """
        a_n = self.acceleration(i, t)
        w_b = self.angular_rate(i, t)
        f_b = np.asarray(R_i, dtype=float).T @ (a_n - self.g)
        return a_n, w_b, f_b


class LegacyManeuverTrajectory:
    """Legacy/random maneuver generator with smooth spatial constraints.

    For ``random_accel`` and ``agile`` this version avoids two pathologies:

    1. unbounded random-acceleration integration;
    2. collapse of every entity toward one common z reference.

    Each entity receives a distinct reference height ``z_ref[i]`` sampled in a
    stratified way across a configurable vertical band.  The restoring term is
    intentionally weak: it prevents long-term vertical drift, while random
    acceleration still produces genuine 3-D motion around the individual
    reference layer.
    """

    def __init__(self, config):
        self.cfg = config
        self.K = int(config["num_entities"])
        self.g = np.asarray(config["gravity"], dtype=float)
        self._rng = None
        self._a_hold = None
        self._w_hold = None
        self._hold_left = None
        self._z_ref = None

    def reset(self, rng):
        self._rng = rng
        self._a_hold = np.zeros((self.K, 3))
        self._w_hold = np.zeros((self.K, 3))
        self._hold_left = np.zeros(self.K, dtype=int)
        self._z_ref = self._make_z_references()

    @staticmethod
    def _clip(vec, mag):
        if mag <= 0:
            return np.zeros(3)
        n = np.linalg.norm(vec)
        if n < 1e-12:
            return np.zeros(3)
        return vec * min(1.0, mag / n)

    def _make_z_references(self):
        """Create distinct, reproducible reference heights for all entities.

        A stratified set guarantees vertical spread.  A seed-dependent
        permutation prevents entity index from being permanently tied to a
        specific height, while a small jitter avoids perfectly regular layers.
        """
        cfg = self.cfg
        center = float(cfg.get("z_reference", 25.0))
        spread = max(0.0, float(cfg.get("z_reference_spread", 30.0)))
        zmin = float(cfg.get("z_min", 5.0))
        zmax = float(cfg.get("z_max", 45.0))
        margin = max(0.0, float(cfg.get("z_boundary_margin", 3.0)))

        lo_allowed = zmin + margin
        hi_allowed = zmax - margin
        if hi_allowed <= lo_allowed:
            lo_allowed, hi_allowed = zmin, zmax

        half = 0.5 * spread
        lo = max(lo_allowed, center - half)
        hi = min(hi_allowed, center + half)
        if hi < lo:
            lo = hi = float(np.clip(center, lo_allowed, hi_allowed))

        if self.K <= 1 or spread <= 1e-12:
            refs = np.full(self.K, 0.5 * (lo + hi), dtype=float)
        else:
            refs = np.linspace(lo, hi, self.K, dtype=float)
            refs = refs[self._rng.permutation(self.K)]

        jitter = max(0.0, float(cfg.get("z_reference_jitter", 0.0)))
        if jitter > 0.0 and self.K:
            refs = refs + self._rng.normal(0.0, jitter, size=self.K)
        return np.clip(refs, lo_allowed, hi_allowed)

    def reference_height(self, i):
        if self._z_ref is None:
            return float(self.cfg.get("z_reference", 25.0))
        return float(self._z_ref[int(i)])

    def reference_heights(self):
        if self._z_ref is None:
            return np.full(self.K, float(self.cfg.get("z_reference", 25.0)))
        return self._z_ref.copy()

    def _resample_segment(self, i):
        cfg = self.cfg
        na = (self._rng.standard_normal(3)
              * float(cfg.get("maneuver_accel_std", 1.0)))
        na[2] *= float(cfg.get("random_vertical_scale", 0.20))
        nw = self._rng.standard_normal(3)
        nw[:2] *= 0.2
        self._a_hold[i] = self._clip(na, cfg["max_accel"])
        self._w_hold[i] = self._clip(nw, cfg["max_turn_rate"])

        if cfg["traj_type"] == "agile":
            base = min(0.10, float(cfg.get("random_hold_time", 0.50)))
        else:
            base = float(cfg.get("random_hold_time", 0.50))
        self._hold_left[i] = max(1, int(round(base / cfg["dt_imu"])))

    def initial_state(self, i):
        return None                     # M13 supplies a reproducible random start

    def _apply_target_speed(self, a_n, v_i):
        """Restore the full 3-D speed magnitude toward the requested target.

        The correction is parallel to the current XYZ velocity vector:

            a_speed = k_v * (v_target - ||v||) * v_hat

        Therefore it regulates only the speed magnitude and does not directly
        prescribe a new motion direction.  Random acceleration, spatial
        boundary recovery, and the per-entity height controller still shape
        the actual 3-D trajectory.
        """
        cfg = self.cfg
        raw_target = cfg.get("target_speed", None)
        target = float(cfg["speed"] if raw_target is None else raw_target)
        gain = float(cfg.get("target_speed_gain", 0.80))

        v = np.asarray(v_i, dtype=float)
        speed_3d = float(np.linalg.norm(v))

        if speed_3d > 1e-9:
            direction = v / speed_3d
        else:
            # Initial speed is normally non-zero.  If an entity nearly stops,
            # use the current random commanded-acceleration direction rather
            # than imposing a fixed global heading.
            a = np.asarray(a_n, dtype=float)
            a_norm = float(np.linalg.norm(a))
            if a_norm > 1e-9:
                direction = a / a_norm
            else:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)

        a_n += gain * (target - speed_3d) * direction
        return a_n

    def _apply_xy_boundary(self, a_n, v_i, p_i):
        cfg = self.cfg
        L = float(cfg["field_size"])
        margin = float(cfg.get("xy_boundary_margin", 20.0))
        k = float(cfg.get("xy_boundary_k", 0.20))
        damping = float(cfg.get("xy_boundary_damping", 0.80))

        for axis in (0, 1):
            lo = margin
            hi = L - margin
            pos = float(p_i[axis])
            vel = float(v_i[axis])

            if pos < lo:
                # Restore inward and damp only outward velocity.
                a_n[axis] += k * (lo - pos) + damping * max(-vel, 0.0)
            elif pos > hi:
                a_n[axis] -= k * (pos - hi) + damping * max(vel, 0.0)
        return a_n

    def _apply_height_control(self, i, a_n, v_i, p_i):
        cfg = self.cfg
        z = float(p_i[2])
        vz = float(v_i[2])
        zref = self.reference_height(i)

        # Weak per-entity restoration: prevents secular drift without forcing
        # all entities into one plane.
        a_n[2] += (
            -float(cfg.get("z_restore_k", 0.035)) * (z - zref)
            -float(cfg.get("z_velocity_damping", 0.25)) * vz
        )

        # Stronger protection near the permitted vertical boundaries.
        zmin = float(cfg.get("z_min", 5.0))
        zmax = float(cfg.get("z_max", 45.0))
        margin = float(cfg.get("z_boundary_margin", 3.0))
        kb = float(cfg.get("z_boundary_k", 0.60))

        low_soft = zmin + margin
        high_soft = zmax - margin
        if z < low_soft:
            a_n[2] += kb * (low_soft - z)
        elif z > high_soft:
            a_n[2] -= kb * (z - high_soft)
        return a_n

    def command(self, i, R_i, v_i, p_i, t):
        cfg = self.cfg
        ttype = cfg["traj_type"]

        if ttype in ("static", "const_vel"):
            a_n = np.zeros(3)
            w_b = np.zeros(3)

        elif ttype in ("circle", "figure8"):
            rate = cfg["max_turn_rate"]
            w_b = np.array([0.0, 0.0, rate])
            vxy = np.array([v_i[0], v_i[1], 0.0])
            a_n = rate * np.array([-vxy[1], vxy[0], 0.0])
            if ttype == "figure8":
                a_n = a_n * np.sign(np.sin(0.2 * t) + 1e-9)

        else:
            if self._hold_left[i] <= 0:
                self._resample_segment(i)

            a_n = self._a_hold[i].copy()
            w_b = self._w_hold[i].copy()
            self._hold_left[i] -= 1

            # Keep the M22 speed axis physically meaningful: random
            # acceleration creates maneuvering, while this term regulates the
            # full 3-D speed magnitude around target_speed.
            a_n = self._apply_target_speed(a_n, v_i)

            a_n = self._apply_xy_boundary(a_n, v_i, p_i)
            a_n = self._apply_height_control(i, a_n, v_i, p_i)

        a_n = self._clip(a_n, cfg["max_accel"])
        w_b = self._clip(w_b, cfg["max_turn_rate"])
        f_b = np.asarray(R_i, dtype=float).T @ (a_n - self.g)
        return a_n, w_b, f_b

def TrajectoryGenerator(config):
    """Factory: the circular formation on the main line, legacy types for tests."""
    ttype = config["traj_type"]
    if ttype == "circle_formation":
        return CircleFormationTrajectory(config)
    if ttype in _LEGACY_TYPES:
        return LegacyManeuverTrajectory(config)
    raise ValueError(f"unsupported traj_type {ttype!r}")


def min_pairwise_distance_spread(config, duration=None, samples=200):
    """Smallest over pairs of (max - min) inter-entity distance over a horizon.

    This is the anti-degeneracy statistic of plan section 5.6.  M20 aborts the
    experiment when it drops below 0.2 R.
    """
    traj = CircleFormationTrajectory(config)
    K = traj.K
    if K < 2:
        return np.inf
    duration = float(config["sim_duration"] if duration is None else duration)
    times = np.linspace(0.0, duration, int(max(samples, 2)))
    positions = np.stack(
        [np.stack([traj.position(i, t) for t in times]) for i in range(K)]
    )                                                  # (K, S, 3)
    worst = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d = np.linalg.norm(positions[i] - positions[j], axis=1)
            worst = min(worst, float(np.ptp(d)))
    return worst


if __name__ == "__main__":
    from config import default_config, coupling_prediction, apply_entity_count
    from l1_mechanization import mechanize

    cfg = default_config(); validate_config(cfg)
    traj = CircleFormationTrajectory(cfg)
    v, R = cfg["speed"], cfg["circle_radius"]

    # 1) speed and centripetal acceleration are constant along the circle
    for i in (0, 3, 7):
        for t in (0.0, 1.3, 7.7):
            assert abs(np.linalg.norm(traj.velocity(i, t)) - v) < 1e-12
            assert abs(np.linalg.norm(traj.acceleration(i, t)) - v ** 2 / R) < 1e-12
            Rm = traj.attitude(i, t)
            assert np.allclose(Rm.T @ Rm, np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(Rm) - 1.0) < 1e-12

    # 2) f^b fed back through M4 closes the circle after one revolution
    period = 2 * np.pi * R / v
    dt = cfg["dt_imu"]
    steps = int(round(period / dt))
    st = traj.initial_state(0)
    p0 = st.p.copy()
    t = 0.0
    for _ in range(steps):
        _, w_b, f_b = traj.command(0, st.R, st.v, st.p, t)
        st = mechanize(st, f_b, w_b, dt, cfg["gravity"], cfg["nominal_int_order"])
        t += dt
    closure = float(np.linalg.norm(st.p - p0))
    assert closure < 0.02 * R, closure

    # 3) the whole formation stays inside the field
    for i in range(cfg["num_entities"]):
        for t in np.linspace(0, period, 60):
            p = traj.position(i, t)
            assert -1e-9 <= p[0] <= cfg["field_size"] + 1e-9
            assert -1e-9 <= p[1] <= cfg["field_size"] + 1e-9

    # 4) anti-degeneracy: pair distances vary by at least 0.2 R over the run,
    #    at every swept speed and both entity counts
    spread = min_pairwise_distance_spread(cfg)
    assert spread >= 0.2 * R, spread
    worst = (None, np.inf)
    for K_test, sizes in ((8, (2, 4, 2)), (10, (2, 6, 2))):
        for v_test in cfg["sweep_v_values"]:
            probe = default_config()
            apply_entity_count(probe, K_test, sizes)
            probe["speed"] = v_test
            validate_config(probe)
            s = min_pairwise_distance_spread(probe)
            if s < worst[1]:
                worst = ((K_test, v_test), s)
            assert s >= 0.2 * R, (K_test, v_test, s)

    # 5) a rigid co-phased formation is detected as degenerate
    bad = default_config()
    bad["circle_phase_offsets"] = np.zeros(bad["num_entities"])
    bad["circle_directions"] = np.ones(bad["num_entities"])
    bad["circle_center_offsets"] = np.zeros((bad["num_entities"], 3))
    try:
        validate_config(bad)
        raise AssertionError("degenerate circular formation accepted")
    except ValueError:
        pass

    # 6) omega and a_c are functions of v alone; the coupling prediction follows
    assert abs(circle_omega(cfg) - v / R) < 1e-12
    assert abs(coupling_prediction(cfg) - v ** 2 * cfg["sync_interval"] ** 2
               / (2 * R)) < 1e-15

    # 7) the legacy factory still works (unit-test path only)
    legacy = default_config(); legacy["traj_type"] = "const_vel"
    validate_config(legacy)
    gen = TrajectoryGenerator(legacy)
    gen.reset(np.random.default_rng(0))
    assert isinstance(gen, LegacyManeuverTrajectory)
    a_n, w_b, f_b = gen.command(0, np.eye(3), np.zeros(3), np.zeros(3), 0.0)
    assert np.allclose(a_n, 0.0) and np.allclose(f_b, -legacy["gravity"])

    print("[M5 CircleFormationTrajectory] OK  (closure=%.3f m over one lap, "
          "min distance spread=%.1f m >= %.1f m; worst grid point %s -> %.1f m)"
          % (closure, spread, 0.2 * R, worst[0], worst[1]))
