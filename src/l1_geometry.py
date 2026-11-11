"""M3 Geometry (L1 basic capability) -- the measurement graph.

The v3 main line fixes the topology once and for all (plan section 2.1):

    * the K mobile entities are FULLY connected, so every UWB epoch carries
      K(K-1)/2 inter-entity ranges.  ``comm_range`` gating is demoted to an
      ablation and is only honoured when ``entity_topology == 'range_gated'``;
    * anchors are visible by class: 4 for class A, 2 for class B, none for
      class C.  This asymmetry is the whole point of the per-class reporting --
      class C is positioned ONLY through cooperation.

An :class:`Edge` is a lightweight record; M8 turns it into a scalar range with a
Jacobian, M12 iterates over the entity-entity edges for eta_S, and M11 counts
them for the measurement traffic.  ``is_anchor`` is the attribute that every
downstream consumer branches on, so it is always present.
"""

import numpy as np

from config import POS_DIM, anchor_visible_entities, class_indices


def antenna_position(p, R, l):
    """Lever-arm endpoint p^a = p + R l.  l may be zeros for 'none' mode.

    Restored per TASK2_NOTES 1: T1's v3 rewrite dropped this helper, but M14
    (l2_sensor) still needs the true antenna endpoint.
    """
    p = np.asarray(p, dtype=float)
    l = np.asarray(l, dtype=float)
    if np.allclose(l, 0.0):
        return p.copy()
    return p + np.asarray(R, dtype=float) @ l



class Edge:
    """One range measurement channel.

    Attributes
    ----------
    i : int          index of the first (always mobile) entity
    j : int          index of the second entity, or -1 for an anchor edge
    is_anchor : bool True for an entity-to-anchor range
    anchor : int     anchor index (-1 when ``is_anchor`` is False)
    """

    __slots__ = ("i", "j", "is_anchor", "anchor")

    def __init__(self, i, j=-1, is_anchor=False, anchor=-1):
        self.i = int(i)
        self.j = int(j)
        self.is_anchor = bool(is_anchor)
        self.anchor = int(anchor)

    def key(self):
        return ("A", self.i, self.anchor) if self.is_anchor else ("E", self.i, self.j)

    def __repr__(self):
        if self.is_anchor:
            return f"Edge(entity {self.i} -> anchor {self.anchor})"
        return f"Edge(entity {self.i} <-> entity {self.j})"

    def __eq__(self, other):
        return isinstance(other, Edge) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


def circular_regular_pairs(K, k):
    """Return the fixed undirected circular ``k``-regular graph.

    Entity indices define the immutable circular order ``0, 1, ..., K-1, 0``.
    For even ``k = 2r``, every entity is connected to the ``r`` predecessors and
    ``r`` successors in that order.  The result therefore has exactly degree
    ``k`` at every vertex, exactly ``K*k/2`` edges, and is connected for
    ``k >= 2``.  No physical positions or connectivity-repair edges are used.
    """
    K = int(K)
    k = int(k)
    if K < 3:
        raise ValueError("fixed circular topology requires at least 3 entities")
    if not 2 <= k < K:
        raise ValueError("k must lie in [2, K-1]")
    if k % 2 != 0:
        raise ValueError("k must be even for the fixed circular topology")

    radius = k // 2
    pairs = set()
    for i in range(K):
        for offset in range(1, radius + 1):
            j = (i + offset) % K
            pairs.add((min(i, j), max(i, j)))

    expected = K * k // 2
    if len(pairs) != expected:
        raise AssertionError(
            f"circular topology construction produced {len(pairs)} edges; "
            f"expected {expected}")
    return sorted(pairs)

def inter_entity_edges(config, positions=None):
    """Return entity-entity edges for full, range-gated, or circular topology.

    The historical ``entity_topology='knn'`` configuration name is retained for
    result compatibility, but it now denotes a fixed circular k-regular graph,
    not physical nearest-neighbour selection.  For ``knn_k = 2r``, entity ``i``
    connects to the ``r`` preceding and ``r`` following indices modulo ``K``.
    """
    K = int(config["num_entities"])
    topology = config["entity_topology"]

    if topology == "full":
        return [Edge(i, j) for i in range(K) for j in range(i + 1, K)]

    if topology == "knn":
        k = int(config.get("knn_k", 4))
        return [Edge(i, j) for i, j in circular_regular_pairs(K, k)]

    if topology != "range_gated":
        raise ValueError(f"unsupported entity_topology {topology!r}")
    if positions is None:
        raise ValueError("positions are required for entity_topology='range_gated'")
    pos = np.asarray(positions, dtype=float)
    if pos.shape != (K, POS_DIM):
        raise ValueError(f"positions must have shape {(K, POS_DIM)}, got {pos.shape}")
    limit = float(config["comm_range"])
    return [
        Edge(i, j)
        for i in range(K)
        for j in range(i + 1, K)
        if float(np.linalg.norm(pos[i] - pos[j])) <= limit
    ]


def anchor_edges(config):
    """Return the entity-anchor edges implied by the frozen class map."""
    table = anchor_visible_entities(config)
    edges = []
    for i in sorted(table):
        for a in table[i]:
            edges.append(Edge(i, -1, True, a))
    return edges


def build_edges(config, positions=None):
    """Full edge list of one UWB epoch: anchor edges first, then entity pairs."""
    return anchor_edges(config) + inter_entity_edges(config, positions)


def edge_counts(config):
    """Return static count information for the configured measurement graph."""
    n_a = len(anchor_edges(config))
    K = int(config["num_entities"])
    topology = config["entity_topology"]
    full = K * (K - 1) // 2
    if topology == "full":
        n_e = n_min = n_max = full
        dynamic = False
    elif topology == "range_gated":
        n_e, n_min, n_max, dynamic = full, 0, full, True
    elif topology == "knn":
        k = int(config.get("knn_k", 4))
        n_e = n_min = n_max = K * k // 2
        dynamic = False
    else:
        raise ValueError(f"unsupported entity_topology {topology!r}")
    return {
        "anchor": n_a,
        "inter": n_e,
        "total": n_a + n_e,
        "inter_min": n_min,
        "inter_max": n_max,
        "dynamic": dynamic,
    }


def los_unit(p_i, p_j, eps=1e-12):
    """Unit line-of-sight vector from ``p_j`` to ``p_i`` and the distance."""
    d = np.asarray(p_i, dtype=float).ravel() - np.asarray(p_j, dtype=float).ravel()
    n = float(np.linalg.norm(d))
    if n < eps:
        return np.zeros(POS_DIM), 0.0
    return d / n, n


def edge_endpoints(edge, positions, anchors):
    """Return the two endpoint positions of an edge (anchor-aware)."""
    p_i = np.asarray(positions[edge.i], dtype=float).ravel()
    if edge.is_anchor:
        return p_i, np.asarray(anchors[edge.anchor], dtype=float).ravel()
    return p_i, np.asarray(positions[edge.j], dtype=float).ravel()


def anchor_rank(config):
    """Rank of the centred anchor cloud; 3 means the anchors span 3-D."""
    A = np.asarray(config["anchor_positions"], dtype=float)
    if A.shape[0] < 2:
        return 0
    return int(np.linalg.matrix_rank(A - A.mean(axis=0), tol=1e-9))


def anchored_entities(config):
    """Indices of the entities that see at least one anchor."""
    table = anchor_visible_entities(config)
    return [i for i in sorted(table) if table[i]]


def check_observability(config):
    """Necessary structural conditions for the formation gauge to be fixed.

    Returns a dict of booleans plus a ``ok`` verdict.  This is a NECESSARY-only
    check (a rank test on the true Jacobian happens in M20's pre-flight); it
    catches the cheap mistakes: coplanar anchors, a single anchored entity, or a
    class-C entity with no cooperative neighbour.
    """
    K = int(config["num_entities"])
    counts = edge_counts(config)
    groups = class_indices(config)
    anchored = anchored_entities(config)
    checks = {
        "anchors_span_3d": anchor_rank(config) >= 3,
        "two_anchored_entities": len(anchored) >= 2,
        "class_C_has_neighbours": (len(groups["C"]) == 0
                                   or (K - 1) >= 1),
        "enough_scalars": counts["total"] >= POS_DIM * K - 0,
    }
    checks["ok"] = all(checks.values())
    return checks


if __name__ == "__main__":
    from config import default_config, validate_config, apply_entity_count

    cfg = default_config(); validate_config(cfg)
    K = cfg["num_entities"]

    # ---- counts: full connectivity + 4|A| + 2|B| anchor edges ----
    counts = edge_counts(cfg)
    assert counts["inter"] == K * (K - 1) // 2
    assert counts["anchor"] == 4 * len(class_indices(cfg)["A"]) + 2 * len(class_indices(cfg)["B"])
    edges = build_edges(cfg)
    assert len(edges) == counts["total"]
    assert sum(1 for e in edges if e.is_anchor) == counts["anchor"]

    # ---- class C really has no anchor edge ----
    groups = class_indices(cfg)
    anchor_owners = {e.i for e in edges if e.is_anchor}
    assert anchor_owners.isdisjoint(set(groups["C"]))
    assert set(groups["A"]) | set(groups["B"]) == anchor_owners
    assert anchored_entities(cfg) == sorted(anchor_owners)

    # ---- every class-C entity is still reachable through entity edges ----
    for c in groups["C"]:
        assert any((not e.is_anchor) and (e.i == c or e.j == c) for e in edges)

    # ---- edges are unique and hashable ----
    assert len(set(edges)) == len(edges)

    # ---- LOS geometry ----
    u, d = los_unit(np.array([3.0, 4.0, 0.0]), np.zeros(3))
    assert abs(d - 5.0) < 1e-12 and np.allclose(u, [0.6, 0.8, 0.0])
    u0, d0 = los_unit(np.zeros(3), np.zeros(3))
    assert d0 == 0.0 and np.allclose(u0, 0.0)

    # ---- endpoints resolve anchors correctly ----
    positions = [np.array([float(i), 0.0, 0.0]) for i in range(K)]
    e_a = next(e for e in edges if e.is_anchor)
    p, q = edge_endpoints(e_a, positions, cfg["anchor_positions"])
    assert np.allclose(q, cfg["anchor_positions"][e_a.anchor])
    e_e = next(e for e in edges if not e.is_anchor)
    p, q = edge_endpoints(e_e, positions, cfg["anchor_positions"])
    assert np.allclose(q, positions[e_e.j])

    # ---- observability structural check ----
    obs = check_observability(cfg)
    assert obs["ok"], obs
    assert anchor_rank(cfg) == 3

    # ---- range gating is honoured only on the ablation path ----
    gated = default_config(); gated["entity_topology"] = "range_gated"
    gated["comm_range"] = 2.5
    validate_config(gated)
    far = [np.array([10.0 * i, 0.0, 0.0]) for i in range(K)]
    assert len(inter_entity_edges(gated, far)) == 0
    assert len(inter_entity_edges(cfg, far)) == K * (K - 1) // 2      # main line ignores it

    # ---- fixed circular topology: exact even degree and connectivity ----
    ring = default_config(); apply_entity_count(ring, 5, (1, 3, 1))
    ring["entity_topology"] = "knn"; ring["knn_k"] = 2
    validate_config(ring)
    ring_pairs = {(e.i, e.j) for e in inter_entity_edges(ring)}
    assert ring_pairs == {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    ring["knn_k"] = 4; validate_config(ring)
    assert len(inter_entity_edges(ring)) == 10  # complete graph for K=5
    odd = default_config(); odd["entity_topology"] = "knn"; odd["knn_k"] = 3
    try:
        validate_config(odd)
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("odd knn_k must be rejected")

    # ---- K = 10 scales as expected ----
    big = default_config(); apply_entity_count(big, 10, (2, 6, 2)); validate_config(big)
    big_counts = edge_counts(big)
    assert big_counts["anchor"] == 4 * 2 + 2 * 6 == 20
    assert big_counts["inter"] == 45
    assert big_counts["total"] == 65

    print("[M3 Geometry] OK  (K=%d: %d anchor + %d inter = %d ranges per epoch; "
          "class C anchor-free)" % (K, counts["anchor"], counts["inter"],
                                    counts["total"]))
