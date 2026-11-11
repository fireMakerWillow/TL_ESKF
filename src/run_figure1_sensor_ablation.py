"""Run the formal Figure 1 sensor-ablation supplement at condition C008.

This script deliberately reuses ``run_large_scale_parallel._build_config`` and
``build_conditions``.  Therefore the truth trajectory, sensor streams, entity
class permutation, anchor visibility, noise, initial errors, numerical options,
120 s duration, and 20 s burn-in are identical to the formal C008 large-scale
baseline.  Only the estimator set changes to ``imu_only`` and ``uwb_only``.

Outputs
-------
<output>/runs/C008/seed_XXXX.json
    One atomic, resumable result per seed.
<output>/sensor_ablation_per_seed_metrics.csv
    Long table of every finite streaming metric.
<output>/sensor_ablation_aggregate_metrics.csv
    Across-seed statistics in the same schema used by the large-scale analysis.
<output>/comparison_sensor_ablations.csv
    Wide compatibility table accepted by plot_reduced_paper_figures.py.
<output>/comparison_results.json
    JSON compatibility summary accepted by the plotting script.
<output>/reference_match_report.json
    Optional comparison with the existing formal C008 run files.
"""

from __future__ import annotations

# Prevent each process from spawning a multithreaded BLAS team.
import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

from run_large_scale_parallel import (
    _build_config as _build_large_scale_config,
    _canonical_hash,
    build_conditions,
)


METHODS = ("imu_only", "uwb_only")
OFFICIAL_SEEDS = tuple(range(1010, 1040))
CONDITION_ID = "C008"
DURATION_S = 120.0
BURN_IN_S = 20.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_large_scale_120s"


AGGREGATE_COLUMNS = [
    "condition_id",
    "experiment_module",
    "target_speed",
    "sync_interval",
    "num_entities",
    "num_A",
    "num_B",
    "num_C",
    "topology_type",
    "topology_k",
    "method",
    "metric_kind",
    "metric_path",
    "n",
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "min",
    "max",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_resamples",
    "bootstrap_confidence",
]


PER_SEED_COLUMNS = [
    "condition_id",
    "experiment_module",
    "target_speed",
    "sync_interval",
    "num_entities",
    "num_A",
    "num_B",
    "num_C",
    "topology_type",
    "topology_k",
    "seed",
    "method",
    "metric_kind",
    "metric_path",
    "value",
    "unit",
    "config_hash",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _parse_seeds(text: str) -> List[int]:
    text = text.strip()
    if ":" in text:
        first, last = text.split(":", 1)
        seeds = list(range(int(first), int(last) + 1))
    else:
        seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed list contains duplicates")
    return seeds


def _baseline_condition() -> dict:
    matches = [item for item in build_conditions() if item["condition_id"] == CONDITION_ID]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {CONDITION_ID}, got {len(matches)}")
    condition = matches[0]
    expected = {
        "K": 10,
        "class_sizes": [2, 6, 2],
        "speed": 5.0,
        "sync_interval": 0.10,
        "topology": "full",
        "knn_k": None,
    }
    for key, value in expected.items():
        if condition.get(key) != value:
            raise RuntimeError(
                f"{CONDITION_ID} protocol changed: {key}={condition.get(key)!r}, "
                f"expected {value!r}"
            )
    return condition


def _build_config(condition: Mapping[str, Any], seed: int):
    """Reuse the frozen large-scale config and change only the estimator list."""
    from config import validate_config

    cfg = _build_large_scale_config(dict(condition), int(seed))
    cfg["estimators"] = list(METHODS)
    validate_config(cfg)

    # These assertions make accidental protocol drift fail immediately.
    assert float(cfg["sim_duration"]) == DURATION_S
    assert float(cfg["large_scale_burn_in_s"]) == BURN_IN_S
    assert bool(cfg["large_scale_summary_only"])
    assert cfg["traj_type"] == "random_accel"
    assert float(cfg["target_speed"]) == 5.0
    assert float(cfg["sync_interval"]) == 0.10
    assert int(cfg["num_entities"]) == 10
    assert tuple(cfg["class_sizes"]) == (2, 6, 2)
    assert cfg["entity_topology"] == "full"
    assert cfg["b_anchor_assignment"] == "balanced_mixed_height"
    return cfg


def _config_record(cfg: Mapping[str, Any]) -> dict:
    from config import anchor_visible_entities

    visibility = anchor_visible_entities(cfg)
    return {
        "duration_s": DURATION_S,
        "burn_in_s": BURN_IN_S,
        "dt_imu_s": float(cfg["dt_imu"]),
        "anchor_positions": cfg["anchor_positions"],
        "anchor_visibility_by_entity": {
            str(i): visibility[i] for i in sorted(visibility)
        },
        "entity_class_map": list(cfg["entity_class_map"]),
        "b_anchor_assignment": cfg["b_anchor_assignment"],
    }


def _task_path(output: Path, seed: int) -> Path:
    return output / "runs" / CONDITION_ID / f"seed_{seed:04d}.json"


def _is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload.get("status") == "complete"
    except Exception:
        return False


def _run_one(task: Mapping[str, Any]) -> dict:
    condition = dict(task["condition"])
    seed = int(task["seed"])
    output = Path(task["output"])
    path = _task_path(output, seed)
    try:
        from l3_runner import run_trial

        cfg = _build_config(condition, seed)
        logs, _ = run_trial(cfg, seed, list(METHODS), run_preflight=False)
        config_record = _config_record(cfg)
        config_hash = _canonical_hash(
            {"condition": condition, "seed": seed, "config": config_record}
        )
        payload = {
            "status": "complete",
            "experiment": "figure1_sensor_ablation",
            "condition": condition,
            "seed": seed,
            "methods": {
                name: {"metrics": logs[name]["stream_metrics"]}
                for name in METHODS
            },
            "motion_summary": logs["_meta"]["motion_summary"],
            "topology_summary": logs["_meta"]["topology_summary"],
            "topology_policy": logs["_meta"]["topology"],
            "config": config_record,
            # This is intentionally calculated exactly like the formal runner.
            "config_hash": config_hash,
        }
        _atomic_json(path, payload)
        return {"ok": True, "seed": seed, "path": str(path)}
    except Exception as exc:
        payload = {
            "status": "failed",
            "experiment": "figure1_sensor_ablation",
            "condition": condition,
            "seed": seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(path, payload)
        return {
            "ok": False,
            "seed": seed,
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _metric_kind(path: str) -> str:
    leaf = path.rsplit(".", 1)[-1]
    return leaf if leaf in {"rmse", "rmte", "tmnnees"} else "scalar"


def _iter_group_metric_rows(metrics: Mapping[str, Any]) -> Iterator[Tuple[str, float, str]]:
    """Yield (metric_path, value, unit) for all finite group metrics."""
    groups = metrics.get("groups", {})
    if not isinstance(groups, Mapping):
        return
    for group_name, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        states = group.get("states", {})
        if isinstance(states, Mapping):
            for state_name, state in states.items():
                if not isinstance(state, Mapping):
                    continue
                unit = str(state.get("unit", ""))
                block = state.get("block", {})
                if isinstance(block, Mapping):
                    for metric_name in ("rmse", "rmte", "tmnnees"):
                        value = block.get(metric_name)
                        if _finite_number(value):
                            yield (
                                f"metrics.groups.{group_name}.states.{state_name}.block.{metric_name}",
                                float(value),
                                "1" if metric_name == "tmnnees" else unit,
                            )
                components = state.get("components", {})
                if isinstance(components, Mapping):
                    for component_name, component in components.items():
                        if not isinstance(component, Mapping):
                            continue
                        for metric_name in ("rmse", "rmte", "tmnnees"):
                            value = component.get(metric_name)
                            if _finite_number(value):
                                yield (
                                    f"metrics.groups.{group_name}.states.{state_name}.components.{component_name}.{metric_name}",
                                    float(value),
                                    "1" if metric_name == "tmnnees" else unit,
                                )
        full_state = group.get("full_state", {})
        if isinstance(full_state, Mapping):
            value = full_state.get("tmnnees")
            if _finite_number(value):
                yield (
                    f"metrics.groups.{group_name}.full_state.tmnnees",
                    float(value),
                    "1",
                )


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metadata(condition: Mapping[str, Any]) -> dict:
    n_a, n_b, n_c = map(int, condition["class_sizes"])
    return {
        "condition_id": condition["condition_id"],
        "experiment_module": "Figure1_sensor_ablation",
        "target_speed": float(condition["speed"]),
        "sync_interval": float(condition["sync_interval"]),
        "num_entities": int(condition["K"]),
        "num_A": n_a,
        "num_B": n_b,
        "num_C": n_c,
        "topology_type": str(condition["topology"]),
        "topology_k": condition.get("knn_k"),
    }


def _load_complete_runs(output: Path, seeds: Sequence[int]) -> Tuple[List[dict], List[int]]:
    complete: List[dict] = []
    missing_or_failed: List[int] = []
    for seed in seeds:
        path = _task_path(output, int(seed))
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("status") != "complete":
                missing_or_failed.append(int(seed))
            else:
                complete.append(payload)
        except Exception:
            missing_or_failed.append(int(seed))
    return complete, missing_or_failed


def _per_seed_rows(runs: Sequence[Mapping[str, Any]], condition: Mapping[str, Any]) -> List[dict]:
    base = _metadata(condition)
    rows: List[dict] = []
    for run in runs:
        seed = int(run["seed"])
        config_hash = str(run.get("config_hash", ""))
        methods = run.get("methods", {})
        for method in METHODS:
            method_payload = methods.get(method, {})
            metrics = method_payload.get("metrics", {}) if isinstance(method_payload, Mapping) else {}
            for path, value, unit in _iter_group_metric_rows(metrics):
                rows.append({
                    **base,
                    "seed": seed,
                    "method": method,
                    "metric_kind": _metric_kind(path),
                    "metric_path": path,
                    "value": value,
                    "unit": unit,
                    "config_hash": config_hash,
                })
    rows.sort(key=lambda row: (row["method"], row["metric_path"], row["seed"]))
    return rows


def _stable_seed(base_seed: int, *parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return (int(base_seed) + int.from_bytes(digest, "little")) % (2**63 - 1)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> Tuple[float | None, float | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0 or int(resamples) <= 0:
        return None, None
    if n == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(int(seed))
    alpha = (1.0 - float(confidence)) / 2.0
    means = np.empty(int(resamples), dtype=float)
    batch = min(1000, int(resamples))
    offset = 0
    while offset < int(resamples):
        count = min(batch, int(resamples) - offset)
        indices = rng.integers(0, n, size=(count, n), endpoint=False)
        means[offset:offset + count] = x[indices].mean(axis=1)
        offset += count
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _stats(values: Sequence[float]) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "q25": None, "q75": None, "min": None, "max": None,
        }
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _aggregate_rows(
    per_seed_rows: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any],
    *,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> List[dict]:
    base = _metadata(condition)
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in per_seed_rows:
        key = (str(row["method"]), str(row["metric_kind"]), str(row["metric_path"]))
        grouped.setdefault(key, []).append(float(row["value"]))

    rows: List[dict] = []
    for (method, metric_kind, metric_path), values in sorted(grouped.items()):
        stats = _stats(values)
        ci_low, ci_high = _bootstrap_mean_ci(
            np.asarray(values, dtype=float),
            resamples=int(bootstrap_resamples),
            confidence=float(bootstrap_confidence),
            seed=_stable_seed(
                int(bootstrap_seed),
                "aggregate",
                condition["condition_id"],
                method,
                metric_path,
            ),
        )
        rows.append({
            **base,
            "method": method,
            "metric_kind": metric_kind,
            "metric_path": metric_path,
            **stats,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_confidence": float(bootstrap_confidence),
        })
    return rows


def _aggregate_lookup(aggregate_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["method"]), str(row["metric_path"])): row
        for row in aggregate_rows
    }


def _wide_summary_rows(
    aggregate_rows: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any],
    seed_count: int,
) -> List[dict]:
    lookup = _aggregate_lookup(aggregate_rows)
    base = _metadata(condition)

    def get(method: str, path: str, field: str = "mean") -> Any:
        row = lookup.get((method, path))
        return None if row is None else row.get(field)

    rows = []
    for method in METHODS:
        row: Dict[str, Any] = {
            **base,
            "method": method,
            "role": "sensor_ablation",
            "K": int(condition["K"]),
            "speed": float(condition["speed"]),
            "duration_s": DURATION_S,
            "burn_in_s": BURN_IN_S,
            "mc_runs": int(seed_count),
        }
        overall_prefix = "metrics.groups.all.states.position.block"
        for short_name, metric_name in (
            ("pos_rmse", "rmse"),
            ("rmte", "rmte"),
            ("pos_tmnnees", "tmnnees"),
        ):
            path = f"{overall_prefix}.{metric_name}"
            row[short_name] = get(method, path)
            row[f"{short_name}_std"] = get(method, path, "std")
            row[f"{short_name}_ci_low"] = get(method, path, "bootstrap_ci_low")
            row[f"{short_name}_ci_high"] = get(method, path, "bootstrap_ci_high")
        # Common aliases accepted by the current plotting script.
        row["rmse"] = row["pos_rmse"]
        row["tmnnees"] = row["pos_tmnnees"]
        row["time_mean_nnees"] = row["pos_tmnnees"]

        for label in ("A", "B", "C"):
            prefix = f"metrics.groups.class_{label}.states.position.block"
            for short_name, metric_name in (
                ("pos_rmse", "rmse"),
                ("rmte", "rmte"),
                ("time_mean_nnees", "tmnnees"),
            ):
                path = f"{prefix}.{metric_name}"
                row[f"{short_name}_{label}"] = get(method, path)
                row[f"{short_name}_{label}_std"] = get(method, path, "std")
                row[f"{short_name}_{label}_ci_low"] = get(
                    method, path, "bootstrap_ci_low"
                )
                row[f"{short_name}_{label}_ci_high"] = get(
                    method, path, "bootstrap_ci_high"
                )
        rows.append(row)
    return rows


def _comparison_json(
    wide_rows: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
) -> dict:
    methods = {}
    for row in wide_rows:
        method = str(row["method"])
        methods[method] = {
            key: value
            for key, value in row.items()
            if key not in {
                "condition_id", "experiment_module", "target_speed",
                "sync_interval", "num_entities", "num_A", "num_B",
                "num_C", "topology_type", "topology_k", "method",
                "K", "speed",
            }
        }
    return {
        "metadata": {
            **_metadata(condition),
            "duration_s": DURATION_S,
            "burn_in_s": BURN_IN_S,
            "sensor_ablation_methods": list(METHODS),
            "seeds": [int(seed) for seed in seeds],
            "mc_runs": len(seeds),
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_confidence": float(bootstrap_confidence),
            "aggregation": (
                "Each seed first produces post-burn-in streaming metrics; "
                "the reported value is the arithmetic mean across independent seeds."
            ),
        },
        "methods": methods,
    }


def _reference_match_report(
    runs: Sequence[Mapping[str, Any]],
    reference_root: Path | None,
) -> dict:
    report = {
        "reference_root": str(reference_root) if reference_root is not None else None,
        "checked": 0,
        "missing": [],
        "config_hash_matches": 0,
        "config_hash_mismatches": [],
        "motion_summary_matches": 0,
        "motion_summary_mismatches": [],
        "all_available_references_match": True,
    }
    if reference_root is None:
        return report

    for run in runs:
        seed = int(run["seed"])
        reference_path = reference_root / "runs" / CONDITION_ID / f"seed_{seed:04d}.json"
        if not reference_path.is_file():
            report["missing"].append(seed)
            continue
        try:
            with reference_path.open("r", encoding="utf-8") as handle:
                reference = json.load(handle)
        except Exception as exc:
            report["config_hash_mismatches"].append({
                "seed": seed, "reason": f"cannot read reference: {exc}"
            })
            report["all_available_references_match"] = False
            continue

        report["checked"] += 1
        if reference.get("config_hash") == run.get("config_hash"):
            report["config_hash_matches"] += 1
        else:
            report["config_hash_mismatches"].append({
                "seed": seed,
                "ablation": run.get("config_hash"),
                "reference": reference.get("config_hash"),
            })
            report["all_available_references_match"] = False

        if _canonical_hash(reference.get("motion_summary")) == _canonical_hash(
            run.get("motion_summary")
        ):
            report["motion_summary_matches"] += 1
        else:
            report["motion_summary_mismatches"].append(seed)
            report["all_available_references_match"] = False
    return report


def _write_aggregate_outputs(
    output: Path,
    condition: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    reference_root: Path | None,
) -> Tuple[List[int], dict]:
    runs, missing_or_failed = _load_complete_runs(output, seeds)
    per_seed_rows = _per_seed_rows(runs, condition)
    aggregate_rows = _aggregate_rows(
        per_seed_rows,
        condition,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    wide_rows = _wide_summary_rows(aggregate_rows, condition, len(runs))

    _atomic_csv(
        output / "sensor_ablation_per_seed_metrics.csv",
        PER_SEED_COLUMNS,
        per_seed_rows,
    )
    _atomic_csv(
        output / "sensor_ablation_aggregate_metrics.csv",
        AGGREGATE_COLUMNS,
        aggregate_rows,
    )

    wide_columns: List[str] = []
    for row in wide_rows:
        for key in row:
            if key not in wide_columns:
                wide_columns.append(key)
    _atomic_csv(output / "comparison_sensor_ablations.csv", wide_columns, wide_rows)

    comparison = _comparison_json(
        wide_rows,
        condition,
        [int(run["seed"]) for run in runs],
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
    )
    _atomic_json(output / "comparison_results.json", comparison)

    reference_report = _reference_match_report(runs, reference_root)
    _atomic_json(output / "reference_match_report.json", reference_report)

    summary = {
        "condition": condition,
        "requested_seeds": [int(seed) for seed in seeds],
        "completed_seeds": sorted(int(run["seed"]) for run in runs),
        "missing_or_failed_seeds": missing_or_failed,
        "methods": list(METHODS),
        "duration_s": DURATION_S,
        "burn_in_s": BURN_IN_S,
        "complete_run_count": len(runs),
        "expected_run_count": len(seeds),
        "per_seed_metric_rows": len(per_seed_rows),
        "aggregate_metric_rows": len(aggregate_rows),
        "reference_match": reference_report,
    }
    _atomic_json(output / "ablation_summary.json", summary)
    return missing_or_failed, reference_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Formal Figure 1 sensor ablation: C008, 120 s, 20 s burn-in, "
            "seeds 1010-1039, methods imu_only and uwb_only."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "figure1_sensor_ablation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(30, max(1, os.cpu_count() or 1)),
        help="Maximum process count; at most one task per seed.",
    )
    parser.add_argument("--seeds", default="1010:1039")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None, help="Debug only.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=24680)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Existing formal result root used to verify that C008 config hashes "
            "and motion summaries match. Missing reference files are allowed."
        ),
    )
    parser.add_argument(
        "--no-reference-check",
        action="store_true",
        help="Skip comparison with existing C008 files.",
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.bootstrap_resamples < 0:
        parser.error("--bootstrap-resamples cannot be negative")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        parser.error("--bootstrap-confidence must lie in (0,1)")

    condition = _baseline_condition()
    seeds = _parse_seeds(args.seeds)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_root = (
        None
        if args.no_reference_check
        else args.reference_root.expanduser().resolve()
    )

    manifest = {
        "experiment_purpose": (
            "Supplement Figure 1 with formal IMU-only and UWB-only sensor "
            "ablations under the exact C008 baseline condition."
        ),
        "condition": condition,
        "duration_s": DURATION_S,
        "burn_in_s": BURN_IN_S,
        "methods": list(METHODS),
        "seeds": seeds,
        "task_count": len(seeds),
        "method_run_count": len(seeds) * len(METHODS),
        "requested_workers": int(args.workers),
        "effective_worker_limit": min(30, int(args.workers)),
        "shared_data_within_seed": True,
        "config_source": "run_large_scale_parallel._build_config",
        "reference_root": str(reference_root) if reference_root else None,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_confidence": float(args.bootstrap_confidence),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    _atomic_json(output / "experiment_manifest.json", manifest)

    if args.aggregate_only:
        missing, reference_report = _write_aggregate_outputs(
            output,
            condition,
            seeds,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
            reference_root=reference_root,
        )
        print(
            f"[ablation] aggregate-only: complete={len(seeds) - len(missing)}/"
            f"{len(seeds)}, missing_or_failed={missing}",
            flush=True,
        )
        if not reference_report["all_available_references_match"]:
            print("[ablation] ERROR: available C008 reference files do not match.", flush=True)
            return 2
        return 1 if missing else 0

    tasks = []
    resume = not args.no_resume
    for seed in seeds:
        path = _task_path(output, seed)
        if resume and _is_complete(path):
            continue
        tasks.append({"condition": condition, "seed": seed, "output": str(output)})
    if args.max_tasks is not None:
        tasks = tasks[:max(0, int(args.max_tasks))]

    workers = min(30, int(args.workers), max(1, len(tasks)))
    print(
        f"[ablation] condition={CONDITION_ID}, seeds={len(seeds)}, "
        f"pending={len(tasks)}, methods/task={len(METHODS)}, workers={workers}",
        flush=True,
    )
    print(
        "[ablation] protocol: K=10, A/B/C=2/6/2, v=5 m/s, T=0.10 s, "
        "full topology, duration=120 s, burn-in=20 s",
        flush=True,
    )

    if args.dry_run:
        print(f"[ablation] manifest -> {output / 'experiment_manifest.json'}", flush=True)
        return 0

    failed = 0
    if tasks:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_run_one, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                failed += int(not result["ok"])
                message = "ok" if result["ok"] else result.get("error", "failed")
                print(
                    f"[ablation] {index}/{len(tasks)} seed={result['seed']} {message}",
                    flush=True,
                )

    missing, reference_report = _write_aggregate_outputs(
        output,
        condition,
        seeds,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
        reference_root=reference_root,
    )
    print(
        f"[ablation] finished: worker_failures={failed}, "
        f"complete={len(seeds) - len(missing)}/{len(seeds)}, output={output}",
        flush=True,
    )
    if reference_report["checked"]:
        print(
            f"[ablation] reference check: checked={reference_report['checked']}, "
            f"config_matches={reference_report['config_hash_matches']}, "
            f"motion_matches={reference_report['motion_summary_matches']}",
            flush=True,
        )
    if not reference_report["all_available_references_match"]:
        print("[ablation] ERROR: available C008 reference files do not match.", flush=True)
        return 2
    return 1 if failed or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
