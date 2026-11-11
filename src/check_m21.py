"""Acceptance gate for M21 Monte-Carlo comparison."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from config import (
    ENTITY_CLASSES,
    MAIN_LINE_ESTIMATORS,
    SENSOR_ABLATION_ESTIMATORS,
    apply_entity_count,
    default_config,
    validate_config,
)
from l4_comparison import _scenario_specs, run_scenario


def _smoke_config(directory, mc_runs=2):
    cfg = default_config()
    apply_entity_count(cfg, 4, (1, 1, 2))
    cfg["mc_runs"] = int(mc_runs)
    cfg["sim_duration"] = 0.2
    cfg["output_dir"] = str(directory)
    cfg["save_plots"] = False
    cfg["run_preflight"] = False
    cfg["comparison_warmup"] = False
    cfg["comparison_run_extension"] = False
    cfg["comparison_entity_counts"] = [4]
    cfg["comparison_class_sizes"] = {4: (1, 1, 2)}
    validate_config(cfg)
    return cfg


def _numeric_signature(payload):
    signature = {}
    for name, row in payload["methods"].items():
        signature[name] = {
            "pos_rmse": row["pos_rmse"],
            "rmte": row["rmte"],
            "anees": row["anees"],
            "class": {
                label: (
                    row["class_metrics"][label]["pos_rmse"],
                    row["class_metrics"][label]["rmte"],
                    row["class_metrics"][label]["anees"],
                ) for label in ENTITY_CLASSES
            },
        }
    return signature


def main():
    # 1) Default experiment definition is K=10 with A/B/C = 2/6/2.
    cfg_default = default_config()
    validate_config(cfg_default)
    assert _scenario_specs(cfg_default) == [(10, (2, 6, 2))]

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cfg = _smoke_config(root / "first", mc_runs=2)
        payload = run_scenario(cfg, verbose=False)
        methods = payload["methods"]
        expected = set(MAIN_LINE_ESTIMATORS + SENSOR_ABLATION_ESTIMATORS)
        assert set(methods) == expected
        assert payload["metadata"]["architecture_methods"] == list(MAIN_LINE_ESTIMATORS)
        assert payload["metadata"]["sensor_ablation_methods"] == list(SENSOR_ABLATION_ESTIMATORS)
        assert payload["metadata"]["seeds"] == [cfg["seed_base"], cfg["seed_base"] + 1]
        assert payload["metadata"]["nees_sampling"].startswith("post-burn-in time mean")

        # 2) Pooled global RMSE must exactly recombine from A/B/C strata.
        weights = dict(zip(ENTITY_CLASSES, cfg["class_sizes"]))
        for name, row in methods.items():
            recombined = np.sqrt(sum(
                weights[label] * row["class_metrics"][label]["pos_rmse"] ** 2
                for label in ENTITY_CLASSES
            ) / cfg["num_entities"])
            assert np.isclose(recombined, row["pos_rmse"], rtol=1e-12, atol=1e-12)
            assert row["anees_num_trials"] == cfg["mc_runs"]
            assert set(row["class_metrics"]) == set(ENTITY_CLASSES)
            assert np.isfinite(row["rmte"])
            assert np.isfinite(row["runtime_total_s"])
            assert row["runtime_samples"] == cfg["mc_runs"]
            assert row["memory_array_bytes"] > 0
            assert row["storage_analytic_bytes"] > 0

        # 3) Measurement traffic is a sensor-network constant; estimation
        # traffic must expose method differences.
        measurement = [methods[n]["comm_measurement_bytes"] for n in methods]
        assert np.allclose(measurement, measurement[0])
        estimation = {n: methods[n]["comm_estimation_bytes"] for n in methods}
        assert estimation["imu_only"] == 0
        assert estimation["dual_layer"] < estimation["centralized"]
        assert len(set(estimation.values())) > 2

        # 4) Required audit files and table split.
        required = {
            "comparison_results.json",
            "comparison_results.csv",
            "comparison_per_run.csv",
            "comparison_architectures.csv",
            "comparison_sensor_ablations.csv",
            "comparison_config_snapshot.json",
        }
        assert required <= {p.name for p in (root / "first").iterdir()}
        with open(root / "first" / "comparison_results.json", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        assert set(on_disk["methods"]) == expected
        with open(root / "first" / "comparison_per_run.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == cfg["mc_runs"] * len(expected)
        assert {int(row["seed"]) for row in rows} == set(payload["metadata"]["seeds"])

        # 5) Same seeds/config reproduce every numerical result.  Runtime is
        # intentionally excluded because wall-clock measurements are stochastic.
        cfg2 = copy.deepcopy(cfg)
        cfg2["output_dir"] = str(root / "second")
        payload2 = run_scenario(cfg2, verbose=False)
        sig1, sig2 = _numeric_signature(payload), _numeric_signature(payload2)
        for name in expected:
            assert np.isclose(sig1[name]["pos_rmse"], sig2[name]["pos_rmse"])
            assert np.isclose(sig1[name]["rmte"], sig2[name]["rmte"])
            assert np.isclose(sig1[name]["anees"], sig2[name]["anees"])
            for label in ENTITY_CLASSES:
                assert np.allclose(sig1[name]["class"][label],
                                   sig2[name]["class"][label])

    print("[M21 acceptance] PASS")
    print("  default scenario  : K=10 (2,6,2)")
    print("  methods           : 5 architectures + 2 sensor ablations")
    print("  metrics           : pooled RMSE + RMTE + seed-wise TMNNEES")
    print("  reporting         : full network + A/B/C + runtime/memory/comm split")
    print("  reproducibility   : deterministic numerical signature by seed")


if __name__ == "__main__":
    main()
