"""Checks balanced B anchors and the experiment-design v2 manifest."""

from config import (anchor_visible_entities, apply_entity_count,
                    default_config, validate_config)
from run_large_scale_parallel import build_conditions


def main():
    for K, sizes in ((5, (1, 3, 1)), (10, (2, 6, 2)),
                     (20, (4, 12, 4)), (40, (8, 24, 8)),
                     (10, (1, 7, 2)), (10, (2, 4, 4))):
        cfg = default_config()
        apply_entity_count(cfg, K, sizes)
        validate_config(cfg)
        vis = anchor_visible_entities(cfg)
        b = [vis[i] for i, label in enumerate(cfg["entity_class_map"])
             if label == "B"]
        counts = [sum(anchor in pair for pair in b) for anchor in range(4)]
        assert all(len(pair) == 2 for pair in b)
        assert all(abs(cfg["anchor_positions"][pair[0], 2]
                       - cfg["anchor_positions"][pair[1], 2]) > 0 for pair in b)
        assert max(counts) - min(counts) <= 1, (sizes, counts, b)

    conditions = build_conditions()
    assert len(conditions) == 40
    assert sum("E1_maneuver_sync" in c["experiment_modules"] for c in conditions) == 25
    assert sum("E1b_speed_control" in c["experiment_modules"] for c in conditions) == 2
    assert sum("E2_network_size" in c["experiment_modules"] for c in conditions) == 4
    assert sum("E3_class_composition" in c["experiment_modules"] for c in conditions) == 7
    assert sum("E4_topology" in c["experiment_modules"] for c in conditions) == 5
    nominal = [c for c in conditions if c["is_nominal"]]
    assert len(nominal) == 1 and nominal[0]["condition_id"] == "C008"
    sparse = [c for c in conditions if c["topology"] == "knn"]
    assert sparse
    assert all(int(c["knn_k"]) % 2 == 0 for c in sparse)
    assert all(2 <= int(c["knn_k"]) < int(c["K"]) for c in sparse)
    print("[large-scale revision] OK: balanced B anchors + 25-point E1 + "
          "40 unique conditions + even circular degrees")


if __name__ == "__main__":
    main()
