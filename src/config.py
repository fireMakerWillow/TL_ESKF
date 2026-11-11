"""M0 unified configuration (v3, aligned with the restructured problem statement).

All tunable parameters live here.  Every module reads parameters only through
``config[key]`` and never re-hardcodes a value that already appears here.

Error-state layout for one entity (15-dim, 3D), state order [dp, dv, dtheta, dba, dbg]:
    index 0:3   -> position error   dp      (m)
    index 3:6   -> velocity error   dv      (m/s)
    index 6:9   -> attitude error   dtheta  (rad, body-frame right small angle:
                                             R = R_hat Exp(dtheta))
    index 9:12  -> accel-bias error dba     (m/s^2)
    index 12:15 -> gyro-bias error  dbg     (rad/s)

v3 main line (see M_DLCL_ESKF_v3 section 0):
    * entities are partitioned into classes A / B / C with 4 / 2 / 0 visible
      anchors; all metrics are reported per class;
    * mobile entities are fully connected (no comm_range gating);
    * the trajectory is a circular formation of radius R with per-entity phase,
      direction and centre offsets (anti-degeneracy);
    * the motion-intensity axis collapses to the circle speed v, and the sync
      interval T forms the second axis of the single 2-D sweep (v, T);
    * lever arm is out of scope (``lever_arm_mode='none'``), and the
      cross-block propagation is the regression form only; obsolete estimator
      paths are physically removed rather than retained as dormant code.

Derived keys (``entity_class_map``, ``imu_per_uwb``, ``dt_uwb``) are filled in by
:func:`validate_config`, which every subsystem calls on construction.
"""

import numpy as np

# ---- fixed error-state layout (from the problem description: 3D, 15-dim) ----
POS_DIM = 3
STATE_DIM = 15
IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_TH = slice(6, 9)
IDX_BA = slice(9, 12)
IDX_BG = slice(12, 15)

# Channel names used by the four-channel decomposition of M12 and by M10.
Z_CHANNELS = ("vel", "att", "acc_bias", "gyro_bias")
Z_CHANNEL_SLICES = {
    "vel": slice(0, 3),          # offsets inside the 12-dim non-position block
    "att": slice(3, 6),
    "acc_bias": slice(6, 9),
    "gyro_bias": slice(9, 12),
}

ENTITY_CLASSES = ("A", "B", "C")

# v3 main-line baseline set (section 6 of the restructured plan).
MAIN_LINE_ESTIMATORS = ("centralized", "dual_layer", "local_only",
                        "ci_trace", "ci_det")
SENSOR_ABLATION_ESTIMATORS = ("imu_only", "uwb_only")
ESTIMATOR_NAMES = MAIN_LINE_ESTIMATORS + SENSOR_ABLATION_ESTIMATORS


def default_config():
    """Return a fresh copy of the default configuration dict."""
    cfg = {
        # ---- dimensions / scene (section 1.1) ----
        "num_entities": 10,              # K = |A| + |B| + |C|
        "class_sizes": (2, 6, 2),        # (|V_A|, |V_B|, |V_C|): A=2, B=6, C=2
        "entity_class_map": None,        # derived by validate_config, then frozen
        "pos_dim": POS_DIM,              # 3  (problem description)
        "state_dim": STATE_DIM,          # 15 (problem description)
        "num_anchors": 4,                # fixed at 4, non-coplanar
        "anchor_positions": np.array([   # four non-coplanar anchors
            [0.0,   0.0,   100.0],
            [200.0, 0.0,   0.0],
            [0.0,   200.0, 0.0],
            [200.0, 200.0, 100.0],        # raised, removes the mirror ambiguity
        ]),
        "anchor_visible_by_class": {"A": [0, 1, 2, 3], "B": [0, 1], "C": []},
        # Class B still sees exactly two anchors, but different B entities are
        # assigned balanced mixed-height pairs instead of all using [0, 1].
        # With the default geometry the cycle is (0,1), (3,2), (0,2), (3,1),
        # repeated as needed; anchor usage differs by at most one at any prefix.
        "b_anchor_assignment": "balanced_mixed_height",
        "field_size": 200.0,             # m, square field (replaces v2 area_size)
        "entity_topology": "full",       # full | range_gated | knn
        "comm_range": np.inf,            # used only by range_gated
        "knn_k": 4,                      # exact even degree in fixed circular topology
        "los_prob": 1.0,                 # DEMOTED: all-LOS main line
        "nlos_bias_mean": 0.0,           # DEMOTED
        "gravity": np.array([0.0, 0.0, -9.81]),

        # ---- trajectory: circular formation (section 1.2) ----
        "traj_type": "circle_formation",
        "circle_radius": 70.0,           # R (m), fixed within one (v,T) sweep
        "circle_center": np.array([100.0, 100.0, 5.0]),
        "speed": 5.0,                    # v (m/s), sweep axis; omega=v/R, a_c=v^2/R
        "circle_phase_offsets": None,    # None -> evenly spaced, pairwise distinct
        "circle_directions": None,       # None -> alternating +1 / -1
        "circle_center_offsets": None,   # None -> deterministic ring of offsets
        "max_accel": 30.0,               # safety cap; raised for sigma_a=8
        "max_turn_rate": 2.0,            # DEMOTED: safety cap only

        # ---- bounded random/agile motion (used by quick experiments) ----
        "random_hold_time": 0.50,
        "random_vertical_scale": 0.20,
        "maneuver_accel_std": 1.0,       # sigma_a (m/s^2), horizontal maneuver std

        # Full 3-D target-speed controller for random/agile motion.
        # None -> use config["speed"].  M22 explicitly sets target_speed to
        # the current sweep-v value at every grid point.
        "target_speed": None,
        "target_speed_gain": 0.80,

        "xy_boundary_margin": 30.0,
        "xy_boundary_k": 0.20,
        "xy_boundary_damping": 0.80,
        "random_velocity_damping": 0.03,

        # Height control deliberately uses DIFFERENT references per entity.
        # z_reference is the centre of the reference-height band, not a common
        # altitude to which all entities are pulled.
        "z_reference": 25.0,
        "z_reference_spread": 30.0,      # total span -> roughly 10 ... 40 m
        "z_reference_jitter": 1.0,       # seed-dependent small perturbation
        "z_initial_spread": 2.0,         # initial z around each entity's target
        "z_min": 5.0,
        "z_max": 45.0,
        "z_restore_k": 0.035,            # weak long-term height restoring force
        "z_velocity_damping": 0.25,
        "z_boundary_margin": 3.0,        # start boundary protection before z limit
        "z_boundary_k": 0.60,

        # ---- time (section 1.3) ----
        "dt_imu": 0.005,                 # must divide every T on the sweep grid
        "sync_interval": 0.10,           # T (s), sweep axis
        "imu_per_uwb": None,             # derived: n = T / dt_imu
        "dt_uwb": None,                  # derived alias of sync_interval
        "sim_duration": 60.0,            # s; long enough that the slowest swept
                                         # speed (v=2 m/s) still covers enough of
                                         # the orbit for the M20 anti-degeneracy
                                         # check to pass

        # ---- true inertial-sensor physics (measurement side only) ----
        "true_bias_init_std_acc": 0.02,    # m/s^2
        "true_bias_init_std_gyro": 0.002,  # rad/s
        "true_bias_rw_acc": 1e-3,          # m/s^2 / sqrt(s)
        "true_bias_rw_gyro": 1e-4,         # rad/s / sqrt(s)

        # ---- measurement noise densities ----
        "sigma_acc": 0.05,               # m/s^2 / sqrt(Hz)
        "sigma_gyro": 0.005,             # rad/s / sqrt(Hz)
        "sigma_range": 0.10,             # m  (R = sigma_range**2)

        # ---- filter model Q (kept SEPARATE from truth) ----
        "filter_sigma_acc_bias": 1e-3,
        "filter_sigma_gyro_bias": 1e-4,

        # ---- lever arm: out of scope on the main line (section 12.1) ----
        "lever_arm_mode": "none",
        "lever_arm_vectors": None,

        # ---- initial covariance (std devs) ----
        "init_pos_std": 0.5,
        "init_vel_std": 0.2,
        "init_att_std": 0.02,
        "init_acc_bias_std": 0.02,
        "init_gyro_bias_std": 0.002,

        # ---- ESKF conventions / numerics (section 1.4) ----
        "att_error_convention": "right_body",
        "nominal_int_order": "1",
        "phi_order": "1",
        "reset_jacobian": "first_order",   # G_theta = I - 0.5 [c_theta]_x
        "recompute_L_after_reset": True,
        "reg_epsilon": 1e-9,
        "pg_cross_mode": "regression",     # DEMOTED: 'keep' is ablation only
        "ci_objective": "trace",           # 'trace' | 'det' (M17 instantiation)
        "dcl_measurement_stride": 1,      # process every inter-entity range
        "uwb_only_pos_rw_psd": None,       # q_p; None -> auto q_p = v^2 T
        "joseph_form": True,

        # ---- metrics and diagnostics (section 1.5) ----
        "whiten_lambda": 1e-6,             # W_i = (P + lambda diag(P))^{-1/2}
        "diag_tau": 1e-12,                 # relative floor of the eta ratios
        "rmte_definition": "diagonal_trace",
        "diag_sample_points": ["post_uwb", "pre_uwb", "at_innovation"],
        "diag_in_summary": False,        # enable sampled diagnostics in summary mode
        "diag_sample_stride_s": 0.5,     # diagnostic sampling interval
        "diag_max_pairs": 12,            # 0 means all entity pairs
        "report_by_class": True,
        "nees_confidence": 0.95,
        "runtime_repeats": 3,
        "enable_runtime_profile": True,
        "float_bytes": 8,
        "count_flops": True,
        "separation_check": True,
        "run_preflight": True,              # M20; disable only for tiny smoke tests

        # ---- experiment ----
        "mc_runs": 20,
        "seed_base": 1000,
        "estimators": list(MAIN_LINE_ESTIMATORS),
        "ablation_estimators": list(SENSOR_ABLATION_ESTIMATORS),
        # M21: the main comparison scenario now uses K=10 with A/B/C = 2/6/2.
        "comparison_entity_counts": [10],
        "comparison_class_sizes": {10: (2, 6, 2)},
        "comparison_run_extension": False,
        "comparison_require_full_set": True,
        "comparison_burn_in_frac": 0.5,
        "comparison_warmup": True,

        # ---- (v, T) applicability-domain sweep (section 10) ----
        "sweep_v_values": [2.0, 5.0, 10.0, 15.0, 20.0],
        "sweep_T_values": [0.01, 0.05, 0.10, 0.20, 0.50],
        "zone_rmse_tol": 0.05,             # relative dRMSE tolerance
        "zone_class_c_rmse_tol": 0.10,     # C-class relative dRMSE tolerance
        "zone_nees_tol": 0.25,             # excess outside the MC ANEES band
        "zone_eta_tol": 0.10,              # mean per-cycle eta_dyn tolerance
        "zone_eta_s_tol": 0.25,            # mean innovation eta_S tolerance
        "zone_calibration_tol": 0.25,      # |RMSE/RMTE - 1|
        "m22_burn_in_frac": 0.5,
        "m22_horizon_mode": "fixed_duration",
        "m22_fixed_duration_s": 60.0,
        "m22_target_angle_rad": 1.5707963267948966,  # 90 deg

        # ---- output ----
        "output_dir": "results",
        "save_plots": True,
        "plot_format": "png",
    }
    return cfg


# ----------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------
def derive_class_sizes(config):
    """Return (nA, nB, nC); auto-split when ``class_sizes`` is None."""
    K = int(config["num_entities"])
    sizes = config.get("class_sizes")
    if sizes is None:
        # New default policy: keep two fully anchored A entities and two
        # anchor-free C entities; assign the remaining entities to B.
        if K >= 4:
            return (2, K - 4, 2)
        nA = min(2, K)
        return (nA, K - nA, 0)
    sizes = tuple(int(s) for s in sizes)
    if len(sizes) != 3 or any(s < 0 for s in sizes):
        raise ValueError("class_sizes must be three non-negative integers")
    return sizes


def derive_entity_class_map(config):
    """Return the frozen list of per-entity class labels ('A'/'B'/'C')."""
    nA, nB, nC = derive_class_sizes(config)
    return ["A"] * nA + ["B"] * nB + ["C"] * nC


def class_indices(config):
    """Return ``{'A': [...], 'B': [...], 'C': [...]}`` entity index lists."""
    cmap = config["entity_class_map"] or derive_entity_class_map(config)
    return {c: [i for i, lab in enumerate(cmap) if lab == c]
            for c in ENTITY_CLASSES}


def _balanced_mixed_height_pairs(config):
    """Return a deterministic cycle of balanced two-anchor pairs for class B.

    The four-anchor experiment has two elevated and two lower anchors.  Pairing
    one anchor from each height group gives every B entity vertical diversity,
    while the greedy schedule balances how often every anchor is used.  The
    routine is geometry-derived rather than tied to specific anchor indices.
    """
    anchors = np.asarray(config["anchor_positions"], dtype=float)
    n = int(config["num_anchors"])
    if n < 2:
        return []

    order = np.argsort(anchors[:, 2], kind="stable")
    split = n // 2
    low = [int(x) for x in order[:split]]
    high = [int(x) for x in order[split:]]
    candidates = [(h, l) for h in high for l in low if h != l]
    if not candidates:
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Build one balanced period.  At each step prefer the pair that minimises
    # the resulting anchor-use imbalance, then the pair used least often, then
    # the larger vertical separation, and finally lexicographic order.
    period_len = max(len(candidates), n)
    anchor_use = np.zeros(n, dtype=int)
    pair_use = {tuple(sorted(pair)): 0 for pair in candidates}
    period = []
    for _ in range(period_len):
        best = None
        best_key = None
        for pair in candidates:
            a, b = map(int, pair)
            trial = anchor_use.copy()
            trial[a] += 1
            trial[b] += 1
            canonical = tuple(sorted((a, b)))
            key = (
                int(np.max(trial) - np.min(trial)),
                int(np.max(trial)),
                int(pair_use[canonical]),
                -float(abs(anchors[a, 2] - anchors[b, 2])),
                canonical,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (a, b)
        a, b = best
        period.append([int(a), int(b)])
        anchor_use[a] += 1
        anchor_use[b] += 1
        pair_use[tuple(sorted((a, b)))] += 1
    return period


def anchor_visible_entities(config):
    """Return the frozen per-entity anchor visibility map.

    A entities see all four anchors, C entities see none, and B entities see
    two anchors each.  In ``balanced_mixed_height`` mode the B pairs are spread
    deterministically across the anchor set instead of every B entity sharing
    the same pair.
    """
    cmap = config["entity_class_map"] or derive_entity_class_map(config)
    table = config["anchor_visible_by_class"]
    mode = str(config.get("b_anchor_assignment", "shared"))
    out = {}
    b_entities = [i for i, label in enumerate(cmap) if label == "B"]
    b_rank = {entity: rank for rank, entity in enumerate(b_entities)}
    cycle = (_balanced_mixed_height_pairs(config)
             if mode == "balanced_mixed_height" else [])
    for i, label in enumerate(cmap):
        if label == "B" and cycle:
            out[i] = list(cycle[b_rank[i] % len(cycle)])
        else:
            out[i] = list(table[label])
    return out


def sync_interval(config):
    """Return the UWB sync interval T (s)."""
    return float(config["sync_interval"])


def circle_omega(config):
    """Angular rate omega = v / R of the circular formation."""
    return float(config["speed"]) / float(config["circle_radius"])


def centripetal_accel(config):
    """Centripetal acceleration a_c = v^2 / R."""
    return float(config["speed"]) ** 2 / float(config["circle_radius"])


def coupling_prediction(config, T=None):
    """v-dependent part of the attitude coupling: 0.5 a_c T^2 = v^2 T^2 / (2R).

    This is the curve written in the restructured plan.  It is NOT the full
    coupling magnitude -- see :func:`coupling_prediction_full`.
    """
    T = sync_interval(config) if T is None else float(T)
    return 0.5 * centripetal_accel(config) * T ** 2


def specific_force_magnitude(config):
    """|f^b| = |a^n - g| along the circle: sqrt(a_c^2 + |g|^2).

    The body-frame specific force that the IMU actually measures must cancel
    gravity as well as produce the centripetal acceleration, and the two are
    orthogonal on a level circle.
    """
    a_c = centripetal_accel(config)
    g = float(np.linalg.norm(np.asarray(config["gravity"], dtype=float)))
    return float(np.hypot(a_c, g))


def coupling_prediction_full(config, T=None):
    """FULL analytic prediction of ||Phi_bar[0:3, 6:9]||_F.

    Over one sync interval the accumulated position-attitude block is
    ``-0.5 R [f^b]_x T^2``, whose Frobenius norm is ``sqrt(2) * 0.5 |f^b| T^2``
    because ``|| [x]_x ||_F = sqrt(2) |x|``.  Since ``|f^b| = |a^n - g|``, the
    coupling is driven by the TOTAL specific force, not by the centripetal part
    alone: this is the classical INS tilt-to-horizontal-acceleration channel and
    gravity dominates it whenever a_c << |g|.

    Consequence for the M22 sweep: T is an exact quadratic axis, whereas v only
    enters through sqrt(a_c^2 + g^2) and is therefore a WEAK axis over the
    current grid (a_c = v^2/R reaches |g| only at v = sqrt(gR) ~ 26 m/s for
    R = 70 m).  Use :func:`coupling_excess` when the v-dependence itself is the
    quantity of interest.
    """
    T = sync_interval(config) if T is None else float(T)
    return float(np.sqrt(2.0) * 0.5 * specific_force_magnitude(config) * T ** 2)


def coupling_excess(config, T=None):
    """Coupling in excess of the v = 0 (hover) baseline -- the v-sensitive part."""
    T = sync_interval(config) if T is None else float(T)
    g = float(np.linalg.norm(np.asarray(config["gravity"], dtype=float)))
    a_c = centripetal_accel(config)
    return float(np.sqrt(2.0) * 0.5 * (np.hypot(a_c, g) - g) * T ** 2)


def auto_pos_rw_psd(config):
    """M26 contract: q_p ~ v^2 T, refreshed at every (v, T) grid point."""
    return float(config["speed"]) ** 2 * sync_interval(config)


def lever_arms(config):
    """Return the (K,3) lever-arm array, zeros when unset or mode 'none'."""
    K = int(config["num_entities"])
    if config["lever_arm_mode"] == "none" or config["lever_arm_vectors"] is None:
        return np.zeros((K, 3))
    la = np.asarray(config["lever_arm_vectors"], dtype=float)
    if la.shape == (3,):
        la = np.tile(la, (K, 1))
    if la.shape != (K, 3):
        raise ValueError(
            "lever_arm_vectors must be None, shape (3,), or shape (num_entities, 3)"
        )
    if not np.all(np.isfinite(la)):
        raise ValueError("lever_arm_vectors contains NaN or Inf")
    return la


def circle_parameters(config):
    """Return (phases, directions, centre offsets) for the circular formation.

    Defaults are deliberately anti-degenerate (section 5.6 of the plan):
    evenly spaced distinct phases, alternating rotation senses, and per-entity
    centre offsets spread evenly on a ring of radius 0.30 R, so that every
    inter-entity distance varies with time.  0.30 R is the smallest scale that
    clears the 0.2 R spread threshold on the whole (v, T) x K grid while still
    leaving the formation inside the field (R + 0.30 R < field_size / 2).  The
    realised spread also depends on how much of the orbit the run covers, i.e.
    on v * sim_duration / R -- see the note on ``sim_duration`` in
    :func:`default_config`.
    """
    K = int(config["num_entities"])
    R = float(config["circle_radius"])

    phases = config.get("circle_phase_offsets")
    if phases is None:
        phases = 2.0 * np.pi * np.arange(K) / max(K, 1)
    phases = np.asarray(phases, dtype=float).ravel()

    directions = config.get("circle_directions")
    if directions is None:
        directions = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(K)])
    directions = np.asarray(directions, dtype=float).ravel()

    offsets = config.get("circle_center_offsets")
    if offsets is None:
        scale = 0.30 * R
        offsets = np.zeros((K, 3))
        for i in range(K):
            ang = 2.0 * np.pi * i / max(K, 1) + 0.4
            offsets[i] = scale * np.array([
                np.cos(ang), np.sin(ang), 0.15 * np.sin(2.0 * ang)
            ])
    offsets = np.asarray(offsets, dtype=float)

    if phases.shape != (K,):
        raise ValueError("circle_phase_offsets must have length num_entities")
    if directions.shape != (K,):
        raise ValueError("circle_directions must have length num_entities")
    if offsets.shape != (K, 3):
        raise ValueError("circle_center_offsets must have shape (K, 3)")
    if not np.all(np.isin(directions, (-1.0, 1.0))):
        raise ValueError("circle_directions entries must be +1 or -1")
    return phases, directions, offsets


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
_WARNED_MESSAGES = set()


def _warn(message):
    """Print each advisory configuration warning at most once per process."""
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    print(f"[config][WARNING] {message}")


def validate_config(config):
    """Validate the public configuration contract and fill derived keys.

    Derived keys written back into ``config``: ``entity_class_map``,
    ``imu_per_uwb`` and ``dt_uwb``.  Deviations from the v3 main line are not
    rejected outright (the code paths are retained for the follow-up work of
    sections 12.1/12.2) but they raise an explicit warning.
    """
    enums = {
        "traj_type": {"circle_formation", "static", "const_vel", "circle",
                      "figure8", "agile", "random_accel"},
        "lever_arm_mode": {"none", "A", "B"},
        "att_error_convention": {"right_body"},
        "nominal_int_order": {"1", "midpoint"},
        "phi_order": {"1", "2", "van_loan"},
        "reset_jacobian": {"identity", "first_order"},
        "pg_cross_mode": {"keep", "regression"},
        "ci_objective": {"trace", "det"},
        "entity_topology": {"full", "range_gated", "knn"},
    }
    for key, allowed in enums.items():
        value = config.get(key)
        if value not in allowed:
            raise ValueError(
                f"invalid config[{key!r}]={value!r}; expected one of {sorted(allowed)}"
            )

    def finite_scalar(key):
        value = float(config[key])
        if not np.isfinite(value):
            raise ValueError(f"config[{key!r}] must be finite")
        return value

    def positive(key):
        value = finite_scalar(key)
        if value <= 0.0:
            raise ValueError(f"config[{key!r}] must be positive")
        return value

    def nonnegative(key):
        value = finite_scalar(key)
        if value < 0.0:
            raise ValueError(f"config[{key!r}] must be non-negative")
        return value

    dt_imu = positive("dt_imu")
    T = positive("sync_interval")
    sim_duration = positive("sim_duration")
    positive("field_size")
    positive("circle_radius")
    positive("reg_epsilon")
    nonnegative("speed")
    nonnegative("max_accel")
    nonnegative("max_turn_rate")
    nonnegative("random_hold_time")
    nonnegative("random_vertical_scale")
    positive("maneuver_accel_std")
    target_speed = config.get("target_speed", None)
    if target_speed is not None:
        target_speed = float(target_speed)
        if not np.isfinite(target_speed) or target_speed < 0.0:
            raise ValueError("config['target_speed'] must be None or a non-negative finite scalar")
    nonnegative("target_speed_gain")
    nonnegative("xy_boundary_margin")
    nonnegative("xy_boundary_k")
    nonnegative("xy_boundary_damping")
    nonnegative("random_velocity_damping")
    finite_scalar("z_reference")
    nonnegative("z_reference_spread")
    nonnegative("z_reference_jitter")
    nonnegative("z_initial_spread")
    finite_scalar("z_min")
    finite_scalar("z_max")
    nonnegative("z_restore_k")
    nonnegative("z_velocity_damping")
    nonnegative("z_boundary_margin")
    nonnegative("z_boundary_k")
    nonnegative("nlos_bias_mean")
    nonnegative("whiten_lambda")
    nonnegative("diag_tau")
    positive("diag_sample_stride_s")
    diag_max_pairs = config.get("diag_max_pairs", 12)
    if int(diag_max_pairs) != diag_max_pairs or int(diag_max_pairs) < 0:
        raise ValueError("config['diag_max_pairs'] must be a non-negative integer")
    config["diag_max_pairs"] = int(diag_max_pairs)

    stride_raw = config.get("dcl_measurement_stride", 1)
    if int(stride_raw) != stride_raw or int(stride_raw) < 1:
        raise ValueError(
            "config['dcl_measurement_stride'] must be a positive integer")
    config["dcl_measurement_stride"] = int(stride_raw)

    for key in (
        "true_bias_init_std_acc", "true_bias_init_std_gyro",
        "true_bias_rw_acc", "true_bias_rw_gyro",
        "sigma_acc", "sigma_gyro", "sigma_range",
        "filter_sigma_acc_bias", "filter_sigma_gyro_bias",
        "init_pos_std", "init_vel_std", "init_att_std",
        "init_acc_bias_std", "init_gyro_bias_std",
    ):
        nonnegative(key)

    if float(config["xy_boundary_margin"]) * 2.0 >= float(config["field_size"]):
        raise ValueError("xy_boundary_margin must be less than field_size / 2")
    if float(config["z_min"]) >= float(config["z_max"]):
        raise ValueError("z_min must be smaller than z_max")
    if float(config["z_boundary_margin"]) * 2.0 >= (
            float(config["z_max"]) - float(config["z_min"])):
        raise ValueError("z_boundary_margin must be less than half the z range")

    K_raw = config["num_entities"]
    if int(K_raw) != K_raw or int(K_raw) <= 0:
        raise ValueError("config['num_entities'] must be a positive integer")
    K = int(K_raw)

    A_raw = config["num_anchors"]
    if int(A_raw) != A_raw or int(A_raw) < 0:
        raise ValueError("config['num_anchors'] must be a non-negative integer")
    num_anchors = int(A_raw)

    if int(config["pos_dim"]) != POS_DIM or int(config["state_dim"]) != STATE_DIM:
        raise ValueError("this implementation requires pos_dim=3 and state_dim=15")

    # ---- (M24-1) class sizes must sum to K, and the map is frozen once ----
    sizes = derive_class_sizes(config)
    if sum(sizes) != K:
        raise ValueError(
            f"sum(class_sizes)={sum(sizes)} must equal num_entities={K}"
        )
    config["class_sizes"] = tuple(sizes)
    derived_map = derive_entity_class_map(config)
    existing = config.get("entity_class_map")
    if existing is None:
        config["entity_class_map"] = derived_map
    else:
        existing = list(existing)
        if len(existing) != K or any(c not in ENTITY_CLASSES for c in existing):
            raise ValueError("entity_class_map must be K labels drawn from A/B/C")
        counts = tuple(existing.count(c) for c in ENTITY_CLASSES)
        if counts != tuple(sizes):
            raise ValueError(
                f"entity_class_map class counts {counts} disagree with "
                f"class_sizes {tuple(sizes)}"
            )
        config["entity_class_map"] = existing

    # ---- (M24-2) anchor geometry: shape and non-coplanarity ----
    anchors = np.asarray(config["anchor_positions"], dtype=float)
    if anchors.shape != (num_anchors, POS_DIM):
        raise ValueError("anchor_positions must have shape (num_anchors, 3)")
    if not np.all(np.isfinite(anchors)):
        raise ValueError("anchor_positions contains NaN or Inf")
    if num_anchors == 4:
        volume = float(np.linalg.det(
            np.stack([anchors[1] - anchors[0],
                      anchors[2] - anchors[0],
                      anchors[3] - anchors[0]])
        ))
        if abs(volume) < 1e-6:
            raise ValueError("the four anchors must not be coplanar")
    elif num_anchors != 0:
        _warn(f"num_anchors={num_anchors} deviates from the main line (4).")

    # ---- (M24-3) anchor visibility per class ----
    table = config["anchor_visible_by_class"]
    if set(table) != set(ENTITY_CLASSES):
        raise ValueError("anchor_visible_by_class must have keys A, B, C")
    for label, visible in table.items():
        idx = list(visible)
        if any(int(a) != a or a < 0 or a >= num_anchors for a in idx):
            raise ValueError(
                f"anchor_visible_by_class[{label!r}] holds out-of-range anchors"
            )
        if len(set(idx)) != len(idx):
            raise ValueError(f"anchor_visible_by_class[{label!r}] has duplicates")
    if num_anchors == 4:
        expected = {"A": 4, "B": 2, "C": 0}
        actual = {label: len(table[label]) for label in ENTITY_CLASSES}
        if actual != expected:
            _warn(f"anchor visibility {actual} deviates from the main line "
                  f"{expected} (see restructured plan section 2.1).")
    b_mode = str(config.get("b_anchor_assignment", "shared"))
    if b_mode not in ("shared", "balanced_mixed_height"):
        raise ValueError(
            "b_anchor_assignment must be 'shared' or 'balanced_mixed_height'")
    realised_visibility = anchor_visible_entities(config)
    for entity, visible in realised_visibility.items():
        label = config["entity_class_map"][entity]
        expected_count = {"A": 4, "B": 2, "C": 0}[label]
        if num_anchors == 4 and len(visible) != expected_count:
            raise ValueError(
                f"entity {entity} class {label} sees {len(visible)} anchors; "
                f"expected {expected_count}")

    # ---- (M24-4) at least two entities must see anchors (gauge fixing) ----
    anchored = sum(1 for visible in realised_visibility.values() if visible)
    if num_anchors > 0 and anchored < min(2, K):
        raise ValueError(
            "at least two entities must have a non-empty anchor set, otherwise "
            "the formation yaw gauge stays unobservable "
            "(K=1 is exempt: one entity seeing four anchors has no gauge freedom)"
        )

    # ---- (M24-4b) RMTE definition: enum + main-line check ----
    rmte_def = config.get("rmte_definition", "diagonal_trace")
    if rmte_def not in ("diagonal_trace", "pairwise_relative"):
        raise ValueError(
            f"rmte_definition must be 'diagonal_trace' or 'pairwise_relative', "
            f"got {rmte_def!r}")
    if rmte_def != "diagonal_trace":
        _warn("rmte_definition='%s' deviates from the main-line formula of "
              "[M10] (RMTE = sqrt(mean_i tr(H P_L,i H^T)))." % rmte_def)

    # ---- (M24-5) the time grid: dt_imu must divide T ----
    ratio = T / dt_imu
    if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-9):
        raise ValueError(
            f"sync_interval={T:.12g} must be an integer multiple of dt_imu={dt_imu:.12g}"
        )
    n = int(round(ratio))
    if n <= 0:
        raise ValueError("sync_interval must be at least one IMU step")
    config["imu_per_uwb"] = n
    config["dt_uwb"] = T
    epochs = sim_duration / T
    if not np.isclose(epochs, round(epochs), rtol=1e-10, atol=1e-9):
        raise ValueError("sim_duration must be an integer multiple of sync_interval")

    for T_value in config.get("sweep_T_values", []):
        r = float(T_value) / dt_imu
        if not np.isclose(r, round(r), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"sweep T={T_value} is not an integer multiple of dt_imu={dt_imu}"
            )

    topology = config["entity_topology"]
    if topology == "knn":
        k_nn = config.get("knn_k", 4)
        if int(k_nn) != k_nn:
            raise ValueError("knn_k must be an integer")
        k_nn = int(k_nn)
        if not 2 <= k_nn < K:
            raise ValueError("knn_k must lie in [2, K-1] for circular topology")
        if k_nn % 2 != 0:
            raise ValueError(
                "knn_k must be even: the fixed circular topology connects "
                "k/2 predecessors and k/2 successors")
        config["knn_k"] = k_nn
        if "knn_connectivity_repair" in config:
            _warn("knn_connectivity_repair is obsolete and ignored: the fixed "
                  "circular k-regular topology is connected by construction.")

    gravity = np.asarray(config["gravity"], dtype=float)
    if gravity.shape != (POS_DIM,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")

    los_prob = finite_scalar("los_prob")
    if not 0.0 <= los_prob <= 1.0:
        raise ValueError("config['los_prob'] must lie in [0, 1]")

    # ---- (M24-6) circular-formation anti-degeneracy ----
    phases, directions, offsets = circle_parameters(config)
    if config["traj_type"] == "circle_formation" and K > 1:
        distinct_phase = len(set(np.round(np.mod(phases, 2 * np.pi), 9))) == K
        mixed_direction = len(set(directions.tolist())) > 1
        shifted_centre = not np.allclose(offsets, 0.0)
        if not (distinct_phase or mixed_direction or shifted_centre):
            raise ValueError(
                "degenerate circular formation: give distinct phases, mixed "
                "directions, or non-zero centre offsets (plan section 5.6)"
            )

    mc_runs = config["mc_runs"]
    if int(mc_runs) != mc_runs or int(mc_runs) <= 0:
        raise ValueError("config['mc_runs'] must be a positive integer")
    confidence = finite_scalar("nees_confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("config['nees_confidence'] must lie in (0, 1)")
    for flag in ("separation_check", "run_preflight", "enable_runtime_profile"):
        if not isinstance(config.get(flag), (bool, np.bool_)):
            raise ValueError(f"config[{flag!r}] must be boolean")

    repeats = config["runtime_repeats"]
    if int(repeats) != repeats or int(repeats) < 1:
        raise ValueError("config['runtime_repeats'] must be a positive integer")

    burn = float(config.get("comparison_burn_in_frac", 0.5))
    if not 0.0 <= burn < 1.0:
        raise ValueError("comparison_burn_in_frac must lie in [0,1)")
    counts = list(config.get("comparison_entity_counts", [K]))
    if not counts or len({int(x) for x in counts}) != len(counts):
        raise ValueError("comparison_entity_counts must be non-empty and unique")
    class_table = config.get("comparison_class_sizes", {})
    for raw in counts:
        k_cmp = int(raw)
        if k_cmp <= 0 or k_cmp != raw:
            raise ValueError("comparison_entity_counts must contain positive integers")
        sizes_cmp = class_table.get(k_cmp, class_table.get(str(k_cmp)))
        if sizes_cmp is not None and (len(sizes_cmp) != 3 or
                                      sum(map(int, sizes_cmp)) != k_cmp):
            raise ValueError(
                f"comparison_class_sizes[{k_cmp}] must contain three sizes summing to K")

    estimators = list(config["estimators"])
    if not estimators or len(set(estimators)) != len(estimators):
        raise ValueError("config['estimators'] must be a non-empty list without duplicates")
    unknown = sorted(set(estimators) - set(ESTIMATOR_NAMES))
    if unknown:
        raise ValueError(f"unknown estimator names: {unknown}")
    for name in list(config.get("ablation_estimators", [])):
        if name not in ESTIMATOR_NAMES:
            raise ValueError(f"unknown ablation estimator name: {name}")

    fb = config["float_bytes"]
    if int(fb) != fb or int(fb) <= 0:
        raise ValueError("config['float_bytes'] must be a positive integer")

    # ---- (M24-7) main-line deviation warnings ----
    if config["lever_arm_mode"] != "none":
        _warn("lever_arm_mode != 'none' departs from the main line "
              "(restructured plan section 12.1): H E H^T = 0 no longer holds.")
    if config["pg_cross_mode"] != "regression":
        _warn("pg_cross_mode != 'regression' is an internal ablation only "
              "(restructured plan section 3.4); keep it out of the paper tables.")
    if config["entity_topology"] != "full":
        _warn("entity_topology != 'full' departs from the main line "
              "(restructured plan section 12.2).")

    lever_arms(config)
    return config


def apply_entity_count(config, K, class_sizes=None):
    """Set ``num_entities`` and a consistent class split (helper for tests)."""
    config["num_entities"] = int(K)
    if class_sizes is None:
        if K >= 4:
            class_sizes = (2, K - 4, 2)
        else:
            nA = min(2, K)
            class_sizes = (nA, K - nA, 0)
    config["class_sizes"] = tuple(class_sizes)
    config["entity_class_map"] = None
    config["circle_phase_offsets"] = None
    config["circle_directions"] = None
    config["circle_center_offsets"] = None
    if config.get("lever_arm_vectors") is not None:
        config["lever_arm_vectors"] = None
    return config


if __name__ == "__main__":
    cfg = default_config()
    validate_config(cfg)
    assert cfg["state_dim"] == 15 and cfg["pos_dim"] == 3
    assert cfg["imu_per_uwb"] == int(round(cfg["sync_interval"] / cfg["dt_imu"]))
    assert cfg["entity_class_map"].count("A") == 2
    assert cfg["entity_class_map"].count("B") == 6
    assert cfg["entity_class_map"].count("C") == 2
    assert cfg["anchor_positions"].shape == (cfg["num_anchors"], 3)
    assert "true_bias_rw_acc" in cfg and "filter_sigma_acc_bias" in cfg
    assert cfg["att_error_convention"] == "right_body"
    assert lever_arms(cfg).shape == (cfg["num_entities"], 3)
    assert len(anchor_visible_entities(cfg)[0]) == 4
    assert anchor_visible_entities(cfg)[cfg["num_entities"] - 1] == []

    # Explicit K=10 application preserves the new (2, 6, 2) split.
    big = default_config()
    apply_entity_count(big, 10, (2, 6, 2))
    validate_config(big)
    assert big["entity_class_map"].count("B") == 6
    assert big["entity_class_map"].count("C") == 2

    # a mismatching class split is rejected
    bad = default_config(); bad["class_sizes"] = (2, 5, 2)
    try:
        validate_config(bad)
        raise AssertionError("inconsistent class_sizes accepted")
    except ValueError:
        pass

    # coplanar anchors are rejected
    flat = default_config()
    flat["anchor_positions"] = np.array([[0.0, 0, 0], [200.0, 0, 0],
                                         [0.0, 200, 0], [200.0, 200, 0]])
    try:
        validate_config(flat)
        raise AssertionError("coplanar anchors accepted")
    except ValueError:
        pass

    # the sweep grid must be compatible with dt_imu
    grid = default_config(); grid["dt_imu"] = 0.003
    try:
        validate_config(grid)
        raise AssertionError("incompatible (dt_imu, T) grid accepted")
    except ValueError:
        pass

    # the formation must fit inside the field
    _, _, offs = circle_parameters(cfg)
    reach = cfg["circle_radius"] + float(np.max(np.linalg.norm(offs[:, :2], axis=1)))
    assert cfg["circle_center"][0] - reach >= 0.0
    assert cfg["circle_center"][0] + reach <= cfg["field_size"]

    # the coupling reference curves (see coupling_prediction_full)
    assert coupling_prediction_full(cfg) > 10.0 * coupling_prediction(cfg)
    assert abs(specific_force_magnitude(cfg)
               - np.hypot(centripetal_accel(cfg), 9.81)) < 1e-12
    assert coupling_excess(cfg) < coupling_prediction_full(cfg)

    print("[config] OK: K=%d classes=%s n=%d T=%.3fs R=%.1fm v=%.1fm/s "
          "duration=%.0fs" % (cfg["num_entities"], cfg["class_sizes"],
                              cfg["imu_per_uwb"], cfg["sync_interval"],
                              cfg["circle_radius"], cfg["speed"],
                              cfg["sim_duration"]))
