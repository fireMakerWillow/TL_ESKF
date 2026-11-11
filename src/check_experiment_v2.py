"""End-to-end smoke test for experiment design v2 and collect_results."""

import json
import tempfile
from pathlib import Path

from collect_results import bundle, collect
from run_large_scale_parallel import (
    _atomic_json, _run_one, build_conditions, methods_for_condition,
)


def main():
    conditions = build_conditions()
    assert len(conditions) == 40
    assert sum("E1_maneuver_sync" in c["experiment_modules"] for c in conditions) == 25
    nominal = next(c for c in conditions if c["condition_id"] == "C008")
    assert nominal["is_nominal"]
    assert methods_for_condition(nominal) == (
        "centralized", "dual_layer", "ci_trace", "local_only",
        "ci_det", "imu_only", "uwb_only",
    )

    with tempfile.TemporaryDirectory(prefix="experiment_v2_smoke_") as temp:
        root = Path(temp) / "results"
        root.mkdir(parents=True)
        manifest = {
            "duration_s": 0.2,
            "burn_in_s": 0.0,
            "conditions": [nominal],
            "expected_seeds_by_condition": {"C008": [1010]},
        }
        _atomic_json(root / "experiment_manifest.json", manifest)
        result = _run_one({
            "condition": nominal,
            "seed": 1010,
            "output": str(root),
            "methods": methods_for_condition(nominal),
            "duration_s": 0.2,
            "burn_in_s": 0.0,
            "diagnostics": True,
            "diag_stride_s": 0.1,
            "diag_max_pairs": 12,
        })
        assert result["ok"], result
        payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        assert set(payload["methods"]) == set(methods_for_condition(nominal))
        assert payload["motion_summary"]["specific_force_mean"] is not None
        assert payload["diagnostics_summary"]["pre_uwb"]["all"]["sample_count"] >= 1

        output = root / "analysis_tables"
        collected = collect(root, output, "core", compress=True,
                            methods_keep=("centralized", "dual_layer", "ci_trace",
                                          "local_only", "imu_only", "uwb_only"))
        assert collected["status_counts"] == {"complete": 1}
        assert collected["short_conditions"] == {}
        assert collected["tables"]["metrics"]["rows"] > 0
        assert collected["tables"]["motion"]["rows"] > 0
        assert collected["tables"]["diagnostics"]["rows"] > 0
        archive = bundle(output, collected)
        assert archive.is_file()

    print("[experiment-v2] OK: 25-point E1, 40 conditions, runner + collector")


if __name__ == "__main__":
    main()
