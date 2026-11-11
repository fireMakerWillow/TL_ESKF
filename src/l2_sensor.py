"""M14 SensorSimulator / MeasurementAssembler (L2 domain subsystem).

Turns a TruthRecord into raw measurements.  It is the ONLY consumer of the true
bias, and it reaches geometry ONLY through M3:

    IMU  a_m = f^b_true + b_a_true + n_a,  n_a ~ N(0, sigma_acc^2 / dt I)
         w_m = omega^b_true + b_g_true + n_g
    UWB  z   = || endpoint_i - endpoint_j || + n_r,   n_r ~ N(0, sigma_range^2)

M8 is NOT called here, not even its ``simulate_ranges`` helper.  M8 is the
ESTIMATOR's view of a range -- predicted distance plus Jacobian -- and letting
the truth path borrow it would couple the two sides of the very separation the
NEES verdict depends on.  M8's simulator exists for its own unit tests; the
experiment's measurements come from M3 geometry applied to true positions.

Topology (v3 main line, plan section 2.1)
-----------------------------------------
    inter-entity : FULL mesh, unconditionally K(K-1)/2 edges every epoch.  No
                   ``comm_range`` gate -- that is demoted to the section 12.2
                   ablation and only fires under ``entity_topology='range_gated'``.
    anchors      : derived from the FROZEN class map, 4 for A, 2 for B, none
                   for C.  Class C never sees an anchor, which is the whole
                   reason the per-class reporting of M10 exists.

At K = 8 that is 28 + 12 = 40 scalars per epoch, and the count is asserted every
epoch (T19): a silently thinned graph would change the information content of
the experiment without changing any obviously wrong number.

The discrete IMU-noise standard deviation ``sigma / sqrt(dt)`` is the form that
makes the propagated covariance independent of the sampling rate, and it is
dimensionally consistent with M7's ``Q = G Qc G^T dt``.  The measurement RNG is
a separate substream from the truth RNG.
"""

import numpy as np

from config import (anchor_visible_entities, class_indices, lever_arms,
                    validate_config)
from l1_geometry import (Edge, antenna_position, anchor_edges, edge_counts,
                         inter_entity_edges, los_unit)
from l1_true_bias import corrupt_imu


class RangeEdge:
    """One realised UWB measurement.

    Field convention follows M3's :class:`~l1_geometry.Edge`: ``j = -1`` and
    ``anchor`` is an integer INDEX on an anchor edge (v2 stored the anchor's
    coordinates in the same attribute).  ``anchor_position`` carries the
    coordinates when they are needed.
    """

    __slots__ = ("i", "j", "is_anchor", "anchor", "anchor_position",
                 "meas", "d_true", "los")

    def __init__(self, i, j=-1, is_anchor=False, anchor=-1, anchor_position=None,
                 meas=float("nan"), d_true=float("nan"), los=True):
        self.i = int(i)
        self.j = int(j)
        self.is_anchor = bool(is_anchor)
        self.anchor = int(anchor)
        self.anchor_position = (None if anchor_position is None
                                else np.asarray(anchor_position, dtype=float))
        self.meas = float(meas)
        self.d_true = float(d_true)      # diagnostics only, never fed to a filter
        self.los = bool(los)

    def to_edge(self):
        """The M3 edge this measurement belongs to (topology without values)."""
        return Edge(self.i, self.j, self.is_anchor, self.anchor)

    def endpoints_positions(self, positions, anchors):
        p_i = np.asarray(positions[self.i], dtype=float)
        if self.is_anchor:
            return p_i, np.asarray(anchors[self.anchor], dtype=float)
        return p_i, np.asarray(positions[self.j], dtype=float)

    def __repr__(self):
        target = (f"anchor {self.anchor}" if self.is_anchor
                  else f"entity {self.j}")
        return f"RangeEdge(entity {self.i} -> {target}, z={self.meas:.3f})"


class SensorSimulator:
    """Measurement assembler: IMU samples and the UWB range set of one epoch."""

    def __init__(self, config):
        validate_config(config)
        self.cfg = config
        self.K = int(config["num_entities"])
        self.anchors = np.asarray(config["anchor_positions"], dtype=float)
        self.lever = lever_arms(config)                # (K,3), zeros on the line
        self.expected_counts = edge_counts(config)
        self._meas_rng = None

    # ------------------------------------------------------------------
    def reset(self, seed):
        """Bind the measurement substream, independent of M13's truth stream."""
        self._meas_rng = np.random.default_rng(
            np.random.SeedSequence(int(seed)).spawn(1)[0])
        return self

    def _require_rng(self):
        if self._meas_rng is None:
            raise RuntimeError("reset() must be called before generating data")
        return self._meas_rng

    # ------------------------------------------------------------------
    # IMU
    # ------------------------------------------------------------------
    def generate_imu(self, record):
        """Per-entity ``(a_m, w_m)`` for one IMU epoch."""
        rng = self._require_rng()
        cfg = self.cfg
        dt = float(cfg["dt_imu"])
        out = []
        for i in range(len(record.states)):
            out.append(corrupt_imu(record.f_b[i], record.omega_b[i],
                                   record.b_a[i], record.b_g[i],
                                   float(cfg["sigma_acc"]),
                                   float(cfg["sigma_gyro"]), dt, rng))
        return out

    # ------------------------------------------------------------------
    # UWB
    # ------------------------------------------------------------------
    def _endpoint(self, states, i):
        """True antenna endpoint of entity i, through M3 only."""
        return antenna_position(states[i].p, states[i].R, self.lever[i])

    def _nlos_bias(self, los):
        """Demoted path: a positive exponential bias on an NLOS link."""
        mu = float(self.cfg["nlos_bias_mean"])
        if los or mu <= 0.0:
            return 0.0
        return float(self._meas_rng.exponential(mu))

    def generate_ranges(self, record, check_topology=True):
        """All UWB ranges of one epoch, anchor edges first then the mesh."""
        rng = self._require_rng()
        cfg = self.cfg
        states = record.states
        sigma = float(cfg["sigma_range"])
        los_prob = float(cfg["los_prob"])
        endpoints = [self._endpoint(states, i) for i in range(self.K)]

        topology = anchor_edges(cfg) + inter_entity_edges(cfg, endpoints)
        out = []
        for edge in topology:
            if edge.is_anchor:
                p_i, p_j = endpoints[edge.i], self.anchors[edge.anchor]
            else:
                p_i, p_j = endpoints[edge.i], endpoints[edge.j]
            _, d_true = los_unit(p_i, p_j)                 # M3 geometry, truth
            los = True if los_prob >= 1.0 else bool(rng.random() <= los_prob)
            meas = d_true + sigma * rng.standard_normal() + self._nlos_bias(los)
            out.append(RangeEdge(
                edge.i, edge.j, edge.is_anchor, edge.anchor,
                self.anchors[edge.anchor] if edge.is_anchor else None,
                meas, d_true, los))

        if check_topology:
            self.assert_topology(out)
        return out

    # ------------------------------------------------------------------
    def assert_topology(self, edges):
        """Validate the realised graph and class-dependent anchor policy."""
        topology = self.cfg["entity_topology"]
        n_anchor = sum(1 for e in edges if e.is_anchor)
        inter = [e for e in edges if not e.is_anchor]
        want = self.expected_counts

        if n_anchor != want["anchor"]:
            raise AssertionError(
                f"topology self-check failed: {n_anchor} anchor edges, "
                f"expected {want['anchor']}")

        pairs = [(min(e.i, e.j), max(e.i, e.j)) for e in inter]
        if len(set(pairs)) != len(pairs):
            raise AssertionError("duplicate entity-entity edge detected")

        if topology == "full" and len(inter) != want["inter"]:
            raise AssertionError(
                f"topology self-check failed: {len(inter)} inter edges, "
                f"expected {want['inter']}")

        if topology == "knn":
            k = int(self.cfg["knn_k"])
            if len(inter) != want["inter"]:
                raise AssertionError(
                    f"circular topology has {len(inter)} edges; "
                    f"expected {want['inter']}")
            degrees = np.zeros(self.K, dtype=int)
            adjacency = [set() for _ in range(self.K)]
            for i, j in pairs:
                degrees[i] += 1
                degrees[j] += 1
                adjacency[i].add(j)
                adjacency[j].add(i)
            if not np.all(degrees == k):
                raise AssertionError(
                    f"circular topology degree must equal k={k}: "
                    f"{degrees.tolist()}")
            seen, stack = {0}, [0]
            while stack:
                i = stack.pop()
                for j in adjacency[i]:
                    if j not in seen:
                        seen.add(j)
                        stack.append(j)
            if len(seen) != self.K:
                raise AssertionError("fixed circular topology is disconnected")

        groups = class_indices(self.cfg)
        for e in edges:
            if e.is_anchor and e.i in groups["C"]:
                raise AssertionError(
                    f"class-C entity {e.i} was given an anchor edge")
        return True

    def topology_report(self):
        """Static graph policy; dynamic realised counts are logged by M20."""
        groups = class_indices(self.cfg)
        table = self.cfg["anchor_visible_by_class"]
        per_entity = anchor_visible_entities(self.cfg)
        report = {
            "entity_topology": self.cfg["entity_topology"],
            "inter_edges_static_reference": self.expected_counts["inter"],
            "inter_edges_min": self.expected_counts.get("inter_min"),
            "inter_edges_max": self.expected_counts.get("inter_max"),
            "dynamic_inter_edges": self.expected_counts.get("dynamic", False),
            "anchor_edges_per_epoch": self.expected_counts["anchor"],
            "total_edges_per_epoch": self.expected_counts["total"],
            "class_sizes": {c: len(groups[c]) for c in groups},
            "anchors_per_class": {c: len(table[c]) for c in table},
            "anchor_assignment_mode_B": self.cfg.get(
                "b_anchor_assignment", "shared"),
            "anchor_visibility_by_entity": {
                str(i): list(per_entity[i]) for i in sorted(per_entity)
            },
            "lever_arm_mode": self.cfg["lever_arm_mode"],
        }
        if self.cfg["entity_topology"] == "knn":
            report.update({
                "knn_k": int(self.cfg["knn_k"]),
                "topology_family": "fixed_circular_k_regular",
                "circular_neighbors_each_side": int(self.cfg["knn_k"]) // 2,
                "circular_entity_order": "index_0_to_K_minus_1",
                "position_dependent_topology": False,
                "connectivity_by_construction": True,
            })
        return report


if __name__ == "__main__":
    from config import default_config, validate_config, apply_entity_count
    from l2_truth import TruthSimulator

    cfg = default_config(); cfg["sim_duration"] = 4.0; validate_config(cfg)
    K = cfg["num_entities"]
    groups = class_indices(cfg)

    truth = TruthSimulator(cfg); truth.reset(0)
    sensor = SensorSimulator(cfg); sensor.reset(10_000)
    rec = truth.step()

    # ---- 1) IMU shape ----
    imu = sensor.generate_imu(rec)
    assert len(imu) == K and imu[0][0].shape == (3,) and imu[0][1].shape == (3,)

    # ---- 2) topology: full mesh + class-derived anchors (T19) ----
    edges = sensor.generate_ranges(rec)
    n_anchor = sum(1 for e in edges if e.is_anchor)
    n_inter = len(edges) - n_anchor
    assert n_inter == K * (K - 1) // 2
    assert n_anchor == 4 * len(groups["A"]) + 2 * len(groups["B"])
    assert len(edges) == n_anchor + n_inter
    # class C never receives an anchor edge
    assert all(e.i not in groups["C"] for e in edges if e.is_anchor)
    # every unordered entity pair appears exactly once
    pairs = {(e.i, e.j) for e in edges if not e.is_anchor}
    assert len(pairs) == K * (K - 1) // 2 and all(i < j for i, j in pairs)
    # anchor edges carry both the index and the coordinates, consistently
    for e in edges:
        if e.is_anchor:
            assert np.allclose(e.anchor_position, cfg["anchor_positions"][e.anchor])
            assert e.j == -1
        else:
            assert e.anchor == -1 and e.anchor_position is None
        assert e.to_edge().key() in {x.to_edge().key() for x in edges}

    # ---- 3) the mesh does NOT depend on distance on the main line ----
    far = truth.snapshot()
    far.states[0].p = far.states[0].p + np.array([1e4, 0.0, 0.0])
    assert len(sensor.generate_ranges(far)) == n_anchor + n_inter
    gated = default_config(); gated["entity_topology"] = "range_gated"
    gated["comm_range"] = 1.0; gated["sim_duration"] = 1.0
    validate_config(gated)
    tg = TruthSimulator(gated); rg = tg.reset(0)
    sg = SensorSimulator(gated); sg.reset(1)
    assert sum(1 for e in sg.generate_ranges(rg) if not e.is_anchor) == 0

    # ---- 4) IMU residual statistics: var = sigma^2 / dt ----
    dt = cfg["dt_imu"]
    truth.reset(1); sensor.reset(2)
    res = []
    for _ in range(1500):
        r = truth.step()
        ms = sensor.generate_imu(r)
        for i in range(K):
            res.append(ms[i][0] - r.f_b[i] - r.b_a[i])
    var_emp = float(np.var(np.array(res)))
    var_theory = cfg["sigma_acc"] ** 2 / dt
    assert abs(var_emp - var_theory) / var_theory < 0.05, (var_emp, var_theory)

    # ---- 5) range residual statistics ----
    truth.reset(3); sensor.reset(4)
    rr = []
    for _ in range(200):
        r = truth.step()
        rr.extend(e.meas - e.d_true for e in sensor.generate_ranges(r))
    rr = np.asarray(rr)
    assert abs(float(np.std(rr)) - cfg["sigma_range"]) < 0.02
    assert abs(float(np.mean(rr))) < 4.0 * cfg["sigma_range"] / np.sqrt(rr.size)

    # ---- 6) zero noise -> the measurement is exactly the true quantity ----
    c0 = default_config(); c0["sim_duration"] = 1.0
    c0["sigma_acc"] = 0.0; c0["sigma_gyro"] = 0.0; c0["sigma_range"] = 0.0
    validate_config(c0)
    t0 = TruthSimulator(c0); s0 = SensorSimulator(c0)
    r0 = t0.reset(0); s0.reset(1)
    r0 = t0.step()
    m0 = s0.generate_imu(r0)
    assert np.allclose(m0[0][0], r0.f_b[0] + r0.b_a[0], atol=1e-15)
    assert np.allclose(m0[0][1], r0.omega_b[0] + r0.b_g[0], atol=1e-15)
    for e in s0.generate_ranges(r0):
        assert abs(e.meas - e.d_true) < 1e-12
    # and the true distance really is the true geometry
    P = r0.positions()
    for e in s0.generate_ranges(r0):
        if e.is_anchor:
            d = np.linalg.norm(P[e.i] - cfg["anchor_positions"][e.anchor])
        else:
            d = np.linalg.norm(P[e.i] - P[e.j])
        assert abs(e.d_true - d) < 1e-12

    # ---- 7) the measurement seed never moves the truth (T3) ----
    def truth_path(seed_t, seed_m, n=60):
        t = TruthSimulator(cfg); t.reset(seed_t)
        s = SensorSimulator(cfg); s.reset(seed_m)
        ps = []
        for _ in range(n):
            r = t.step(); s.generate_imu(r); s.generate_ranges(r)
            ps.append(r.states[0].p.copy())
        return np.array(ps)

    assert np.array_equal(truth_path(0, 111), truth_path(0, 222))
    # ... while it does move the measurements
    truth.reset(0); a = SensorSimulator(cfg); a.reset(5)
    ra = truth.step(); za = [e.meas for e in a.generate_ranges(ra)]
    truth.reset(0); b = SensorSimulator(cfg); b.reset(6)
    rb = truth.step(); zb = [e.meas for e in b.generate_ranges(rb)]
    assert not np.allclose(za, zb)

    # ---- 8) generating before reset is an error ----
    try:
        SensorSimulator(cfg).generate_ranges(rec)
        raise AssertionError("range generation before reset was allowed")
    except RuntimeError:
        pass

    # ---- 9) a thinned edge list is caught by the self-check ----
    try:
        sensor.assert_topology(edges[:-1])
        raise AssertionError("a missing edge went unnoticed")
    except AssertionError as exc:
        assert "topology self-check failed" in str(exc)
    # re-point one anchor edge at a class-C entity: the COUNT still matches, so
    # only the class check can catch it
    forged = [e for e in edges]
    victim = next(k for k, e in enumerate(forged) if e.is_anchor)
    forged[victim] = RangeEdge(groups["C"][0], -1, True, 0,
                               cfg["anchor_positions"][0], 1.0, 1.0)
    try:
        sensor.assert_topology(forged)
        raise AssertionError("an anchor edge on a class-C entity went unnoticed")
    except AssertionError as exc:
        assert "class-C" in str(exc)

    # ---- 10) K = 10 scales the graph as expected ----
    big = default_config(); apply_entity_count(big, 10, (2, 6, 2))
    big["sim_duration"] = 1.0; validate_config(big)
    tb = TruthSimulator(big); rb2 = tb.reset(0)
    sb = SensorSimulator(big); sb.reset(1)
    eb = sb.generate_ranges(rb2)
    assert len(eb) == 45 + 20
    rep = sb.topology_report()
    assert (rep["inter_edges_static_reference"] + rep["anchor_edges_per_epoch"] == 65
            and rep["lever_arm_mode"] == "none")

    print("[M14 SensorSimulator] OK  (K=%d: %d inter + %d anchor = %d ranges/epoch, "
          "IMU var %.3e vs %.3e, range std %.4f m)"
          % (K, n_inter, n_anchor, len(edges), var_emp, var_theory,
             float(np.std(rr))))
