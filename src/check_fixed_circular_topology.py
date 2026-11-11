"""Regression checks for the fixed circular even-degree topology."""

import copy
import numpy as np

from config import apply_entity_count, default_config, validate_config
from l1_geometry import circular_regular_pairs, edge_counts, inter_entity_edges
from l1_resource import measurement_scalars
from l2_sensor import SensorSimulator
from l2_truth import TruthSimulator


def degrees(K, pairs):
    out = np.zeros(K, dtype=int)
    adjacency = [set() for _ in range(K)]
    for i, j in pairs:
        out[i] += 1
        out[j] += 1
        adjacency[i].add(j)
        adjacency[j].add(i)
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in adjacency[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return out, seen


def ring_config(K, k):
    cfg = default_config()
    if K == 5:
        sizes = (1, 3, 1)
    elif K >= 4:
        sizes = (2, K - 4, 2)
    else:
        sizes = (K, 0, 0)
    apply_entity_count(cfg, K, sizes)
    cfg["entity_topology"] = "knn"
    cfg["knn_k"] = k
    cfg["sim_duration"] = 1.0
    validate_config(cfg)
    return cfg


def main():
    cfg = ring_config(5, 2)
    pairs = circular_regular_pairs(5, 2)
    assert set(pairs) == {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    assert [(e.i, e.j) for e in inter_entity_edges(cfg)] == pairs

    for K, k in ((5, 2), (5, 4), (10, 2), (10, 4), (10, 6),
                 (10, 8), (20, 4), (40, 4)):
        cfg = ring_config(K, k)
        pairs = [(e.i, e.j) for e in inter_entity_edges(cfg)]
        deg, seen = degrees(K, pairs)
        assert len(pairs) == K * k // 2
        assert np.all(deg == k), (K, k, deg.tolist())
        assert seen == set(range(K)), (K, k, sorted(seen))
        counts = edge_counts(cfg)
        assert counts["inter"] == counts["inter_min"] == counts["inter_max"]
        assert counts["inter"] == K * k // 2
        assert counts["dynamic"] is False
        assert measurement_scalars(cfg)["inter"] == K * k // 2

        # Fixed topology must not depend on physical positions.
        pos_a = np.zeros((K, 3))
        pos_b = np.random.default_rng(7).normal(size=(K, 3)) * 1e6
        assert inter_entity_edges(cfg, pos_a) == inter_entity_edges(cfg, pos_b)

    # Sensor topology check and report.
    cfg = ring_config(10, 4)
    truth = TruthSimulator(cfg); truth.reset(0)
    sensor = SensorSimulator(cfg); sensor.reset(1)
    edges = sensor.generate_ranges(truth.step())
    inter = [e for e in edges if not e.is_anchor]
    assert len(inter) == 20
    report = sensor.topology_report()
    assert report["topology_family"] == "fixed_circular_k_regular"
    assert report["circular_neighbors_each_side"] == 2
    assert report["position_dependent_topology"] is False
    assert report["connectivity_by_construction"] is True

    # Odd degree and invalid upper bound must be rejected.
    for K, k in ((10, 1), (10, 3), (10, 9), (5, 5)):
        bad = default_config()
        apply_entity_count(bad, K, (1, K - 2, 1) if K >= 2 else (K, 0, 0))
        bad["entity_topology"] = "knn"
        bad["knn_k"] = k
        try:
            validate_config(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid circular degree accepted: K={K}, k={k}")

    print("[fixed circular topology] OK")


if __name__ == "__main__":
    main()
