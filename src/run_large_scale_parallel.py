"""Run experiment-design v2 with resumable per-(condition, seed) tasks.

The protocol contains 25 E1 maneuver/synchronisation grid points and 40 unique
physical conditions after de-duplication.  E1 conditions use 50 seeds; all other
conditions use 30.  The nominal condition C008 additionally runs CI-det and the
IMU-only/UWB-only sensor ablations.  Sampled structural diagnostics are enabled
for E1/E3/E4 conditions on seeds 1010..1014 and disabled for E2-only conditions.
"""

from __future__ import annotations

import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CORE_METHODS = ("centralized", "dual_layer", "ci_trace", "local_only")
NOMINAL_EXTRA_METHODS = ("ci_det", "imu_only", "uwb_only")
OFFICIAL_BASE_SEEDS = tuple(range(1010, 1040))
OFFICIAL_E1_SEEDS = tuple(range(1010, 1060))
OFFICIAL_DIAGNOSTIC_SEEDS = tuple(range(1010, 1015))
DIAGNOSTIC_MODULES = {"E1_maneuver_sync", "E3_class_composition", "E4_topology"}


def _jsonable(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return None
    return value


def _canonical_hash(payload) -> str:
    text = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _condition_key(c: dict) -> Tuple:
    return (
        int(c["K"]), tuple(c["class_sizes"]), float(c["target_speed"]),
        float(c["maneuver_accel_std"]), float(c["sync_interval"]),
        str(c["topology"]),
        int(c["knn_k"]) if c["topology"] == "knn" else None,
    )


def _is_nominal(c: dict) -> bool:
    return (
        int(c["K"]) == 10
        and tuple(c["class_sizes"]) == (2, 6, 2)
        and float(c["target_speed"]) == 5.0
        and float(c["maneuver_accel_std"]) == 1.0
        and float(c["sync_interval"]) == 0.10
        and c["topology"] == "full"
    )


def build_conditions() -> List[dict]:
    """Build the confirmed 25-point E1 grid and de-duplicate all scenarios."""
    raw = []

    def add(module, K, class_sizes, target_speed, sigma_a, T, topology, knn_k=None):
        raw.append({
            "experiment_module": module,
            "K": int(K),
            "class_sizes": tuple(class_sizes),
            "target_speed": float(target_speed),
            "maneuver_accel_std": float(sigma_a),
            "sync_interval": float(T),
            "topology": str(topology),
            "knn_k": (None if knn_k is None else int(knn_k)),
        })

    # E1: confirmed 5 x 5 = 25 maneuver/synchronisation grid.
    for sigma_a in (0.5, 1.0, 2.0, 4.0, 8.0):
        for T in (0.02, 0.05, 0.10, 0.20, 0.40):
            add("E1_maneuver_sync", 10, (2, 6, 2), 5.0, sigma_a, T, "full")

    # E1b: speed control at fixed maneuver intensity.
    for speed in (2.0, 10.0):
        add("E1b_speed_control", 10, (2, 6, 2), speed, 1.0, 0.10, "full")

    # E2: network size, fixed circular degree 4.
    for K, sizes in ((5, (1, 3, 1)), (10, (2, 6, 2)),
                     (20, (4, 12, 4)), (30, (6, 18, 6))):
        add("E2_network_size", K, sizes, 5.0, 1.0, 0.10, "knn", 4)

    # E3: class composition.
    for sizes in ((1, 7, 2), (2, 6, 2), (3, 5, 2), (4, 4, 2),
                  (2, 8, 0), (2, 4, 4), (1, 3, 6)):
        add("E3_class_composition", 10, sizes, 5.0, 1.0, 0.10, "full")

    # E4: full and fixed circular even-degree graphs.
    add("E4_topology", 10, (2, 6, 2), 5.0, 1.0, 0.10, "full")
    for degree in (8, 6, 4, 2):
        add("E4_topology", 10, (2, 6, 2), 5.0, 1.0, 0.10, "knn", degree)

    merged: Dict[Tuple, dict] = {}
    for item in raw:
        if item["topology"] == "knn":
            if item["knn_k"] % 2 or not (2 <= item["knn_k"] < item["K"]):
                raise AssertionError(f"invalid circular degree: {item}")
        key = _condition_key(item)
        module = item["experiment_module"]
        if key not in merged:
            merged[key] = {**item, "experiment_modules": [module]}
        elif module not in merged[key]["experiment_modules"]:
            merged[key]["experiment_modules"].append(module)

    conditions = []
    for index, item in enumerate(merged.values(), start=1):
        c = dict(item)
        c["condition_id"] = f"C{index:03d}"
        c["class_sizes"] = list(c["class_sizes"])
        c["experiment_module"] = "+".join(c["experiment_modules"])
        c["is_nominal"] = _is_nominal(c)
        c["diagnostics_enabled"] = (
            int(c["K"]) <= 10
            and bool(DIAGNOSTIC_MODULES.intersection(c["experiment_modules"]))
        )
        c["official_seed_count"] = (
            50 if "E1_maneuver_sync" in c["experiment_modules"] else 30)
        c["methods"] = list(methods_for_condition(c))
        conditions.append(c)

    if len(conditions) != 40:
        raise AssertionError(f"expected 40 unique conditions, got {len(conditions)}")
    e1 = [c for c in conditions if "E1_maneuver_sync" in c["experiment_modules"]]
    if len(e1) != 25:
        raise AssertionError(f"expected 25 E1 grid points, got {len(e1)}")
    nominal = [c for c in conditions if c["is_nominal"]]
    if len(nominal) != 1 or nominal[0]["condition_id"] != "C008":
        raise AssertionError(f"nominal condition must be C008, got {nominal}")
    return conditions


def methods_for_condition(condition: dict) -> Tuple[str, ...]:
    methods = list(CORE_METHODS)
    if bool(condition.get("is_nominal", _is_nominal(condition))):
        methods.extend(NOMINAL_EXTRA_METHODS)
    return tuple(methods)


def official_seeds_for_condition(condition: dict) -> Tuple[int, ...]:
    return (OFFICIAL_E1_SEEDS
            if "E1_maneuver_sync" in condition["experiment_modules"]
            else OFFICIAL_BASE_SEEDS)


def _class_map(K: int, sizes: Iterable[int], seed: int) -> List[str]:
    import numpy as np
    nA, nB, nC = map(int, sizes)
    if nA + nB + nC != int(K):
        raise ValueError("class_sizes do not sum to K")
    rng = np.random.default_rng(int(seed) + 70_000 + 1009 * int(K))
    order = [int(x) for x in rng.permutation(int(K))]
    cmap = [None] * int(K)
    for entity in order[:nA]:
        cmap[entity] = "A"
    for entity in order[nA:nA + nB]:
        cmap[entity] = "B"
    for entity in order[nA + nB:]:
        cmap[entity] = "C"
    return cmap


def _build_config(condition: dict, seed: int, methods, duration_s: float,
                  burn_in_s: float, diagnostics: bool,
                  diag_stride_s: float, diag_max_pairs: int):
    from config import apply_entity_count, default_config, validate_config

    cfg = default_config()
    apply_entity_count(cfg, int(condition["K"]), tuple(condition["class_sizes"]))
    cfg["entity_class_map"] = _class_map(
        int(condition["K"]), condition["class_sizes"], int(seed))
    cfg["traj_type"] = "random_accel"
    cfg["speed"] = float(condition["target_speed"])
    cfg["target_speed"] = float(condition["target_speed"])
    cfg["maneuver_accel_std"] = float(condition["maneuver_accel_std"])
    cfg["sync_interval"] = float(condition["sync_interval"])
    cfg["sim_duration"] = float(duration_s)
    cfg["large_scale_burn_in_s"] = float(burn_in_s)
    cfg["large_scale_summary_only"] = True
    cfg["diag_in_summary"] = bool(diagnostics)
    cfg["diag_sample_stride_s"] = float(diag_stride_s)
    cfg["diag_max_pairs"] = int(diag_max_pairs)
    cfg["entity_topology"] = str(condition["topology"])
    if cfg["entity_topology"] == "knn":
        cfg["knn_k"] = int(condition["knn_k"])
    cfg["estimators"] = list(methods)
    cfg["run_preflight"] = False
    cfg["separation_check"] = False
    cfg["enable_runtime_profile"] = False
    cfg["save_plots"] = False
    cfg["count_flops"] = True
    cfg["b_anchor_assignment"] = "balanced_mixed_height"
    cfg["seed_base"] = int(seed)
    validate_config(cfg)
    return cfg


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False,
                  sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _task_path(output: str, condition_id: str, seed: int) -> Path:
    return Path(output) / "runs" / condition_id / f"seed_{seed:04d}.json"


def _run_one(task: dict) -> dict:
    condition = task["condition"]
    seed = int(task["seed"])
    output = str(task["output"])
    result_path = _task_path(output, condition["condition_id"], seed)
    try:
        from config import anchor_visible_entities
        from l1_resource import storage_floats
        from l3_runner import run_trial

        methods = tuple(task["methods"])
        cfg = _build_config(
            condition, seed, methods,
            duration_s=float(task["duration_s"]),
            burn_in_s=float(task["burn_in_s"]),
            diagnostics=bool(task["diagnostics"]),
            diag_stride_s=float(task["diag_stride_s"]),
            diag_max_pairs=int(task["diag_max_pairs"]),
        )
        logs, _ = run_trial(cfg, seed, list(methods), run_preflight=False)
        visibility = anchor_visible_entities(cfg)
        method_blocks = {}
        for name in methods:
            method_blocks[name] = {
                "metrics": logs[name]["stream_metrics"],
                "theoretical_resources": {
                    **logs[name]["resource"],
                    "storage_floats": int(storage_floats(name, int(condition["K"]))),
                    "storage_bytes": int(storage_floats(name, int(condition["K"]))
                                         * int(cfg.get("float_bytes", 8))),
                    "runtime_measured": False,
                },
            }
        config_record = {
            "duration_s": float(task["duration_s"]),
            "burn_in_s": float(task["burn_in_s"]),
            "dt_imu_s": float(cfg["dt_imu"]),
            "target_speed": float(cfg["target_speed"]),
            "maneuver_accel_std": float(cfg["maneuver_accel_std"]),
            "sync_interval": float(cfg["sync_interval"]),
            "dcl_measurement_stride": int(cfg["dcl_measurement_stride"]),
            "diag_in_summary": bool(cfg["diag_in_summary"]),
            "diag_sample_stride_s": float(cfg["diag_sample_stride_s"]),
            "diag_max_pairs": int(cfg["diag_max_pairs"]),
            "anchor_positions": cfg["anchor_positions"],
            "anchor_visibility_by_entity": {
                str(i): visibility[i] for i in sorted(visibility)
            },
            "entity_class_map": list(cfg["entity_class_map"]),
            "b_anchor_assignment": cfg["b_anchor_assignment"],
            "topology_family": (
                "fixed_circular_k_regular"
                if cfg["entity_topology"] == "knn" else cfg["entity_topology"]),
        }
        payload = {
            "status": "complete",
            "condition": condition,
            "seed": seed,
            "methods": method_blocks,
            "motion_summary": logs["_meta"]["motion_summary"],
            "topology_summary": logs["_meta"]["topology_summary"],
            "topology_policy": logs["_meta"]["topology"],
            "diagnostics_summary": logs["_meta"].get("diagnostics_summary", {}),
            "config": config_record,
            "config_hash": _canonical_hash({
                "condition": condition, "seed": seed, "methods": methods,
                "config": config_record,
            }),
        }
        _atomic_json(result_path, payload)
        return {"ok": True, "condition_id": condition["condition_id"],
                "seed": seed, "path": str(result_path)}
    except Exception as exc:
        failure = {
            "status": "failed", "condition": condition, "seed": seed,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(result_path, failure)
        return {"ok": False, "condition_id": condition["condition_id"],
                "seed": seed, "path": str(result_path),
                "error": f"{type(exc).__name__}: {exc}"}


def _parse_seeds(text: str) -> List[int]:
    text = text.strip()
    if ":" in text:
        a, b = text.split(":", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_condition_ids(text: str | None):
    if not text:
        return None
    return {x.strip() for x in text.split(",") if x.strip()}


def _is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("status") == "complete"
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Experiment design v2: 25-point E1, 40 unique conditions")
    parser.add_argument("--output", default="results_experiment_v2")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seeds", default=None,
                        help="override official per-condition seeds, e.g. 1010 or 1010:1012")
    parser.add_argument("--diag-seeds", default="1010:1014")
    parser.add_argument("--conditions", default=None,
                        help="comma-separated condition IDs, e.g. C008,C029")
    parser.add_argument("--run-only", default=None,
                        help="comma-separated condition IDs to execute while keeping "
                             "the full experiment manifest; intended for targeted "
                             "replacement/reruns such as C031")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--burn-in", type=float, default=20.0)
    parser.add_argument("--diag-stride", type=float, default=0.5)
    parser.add_argument("--diag-max-pairs", type=int, default=12)
    parser.add_argument("--disable-diagnostics", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.burn_in < 0.0 or args.burn_in >= args.duration:
        parser.error("--burn-in must satisfy 0 <= burn-in < duration")
    if args.diag_stride <= 0.0:
        parser.error("--diag-stride must be positive")
    if args.diag_max_pairs < 0:
        parser.error("--diag-max-pairs must be non-negative")

    conditions = build_conditions()
    selected_ids = _parse_condition_ids(args.conditions)
    if selected_ids is not None:
        known = {c["condition_id"] for c in conditions}
        unknown = selected_ids - known
        if unknown:
            parser.error(f"unknown condition IDs: {sorted(unknown)}")
        conditions = [c for c in conditions if c["condition_id"] in selected_ids]

    run_only_ids = _parse_condition_ids(args.run_only)
    if run_only_ids is not None:
        if selected_ids is not None:
            parser.error("--run-only cannot be combined with --conditions")
        known = {c["condition_id"] for c in conditions}
        unknown = run_only_ids - known
        if unknown:
            parser.error(f"unknown --run-only condition IDs: {sorted(unknown)}")
        run_conditions = [c for c in conditions if c["condition_id"] in run_only_ids]
    else:
        run_conditions = conditions

    override_seeds = (_parse_seeds(args.seeds) if args.seeds else None)
    diagnostic_seeds = set(_parse_seeds(args.diag_seeds))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    effective_workers = max(1, min(100, int(args.workers)))

    expected_seeds = {
        c["condition_id"]: list(override_seeds or official_seeds_for_condition(c))
        for c in conditions
    }
    methods_by_condition = {
        c["condition_id"]: list(methods_for_condition(c)) for c in conditions
    }
    task_count = sum(len(v) for v in expected_seeds.values())
    method_run_count = sum(
        len(expected_seeds[c["condition_id"]]) * len(methods_by_condition[c["condition_id"]])
        for c in conditions
    )
    manifest = {
        "schema_version": "experiment-v2.1",
        "experiment_purpose": (
            "Evaluate maneuver intensity and synchronization interval, network "
            "size, class composition and topology using centralized, dual-layer, "
            "CI-trace and DCL local-only estimators on identical measurements."
        ),
        "duration_s": float(args.duration),
        "burn_in_s": float(args.burn_in),
        "core_methods": list(CORE_METHODS),
        "nominal_extra_methods": list(NOMINAL_EXTRA_METHODS),
        "methods_by_condition": methods_by_condition,
        "seed_policy": ("override" if override_seeds is not None else "official_50_E1_30_other"),
        "expected_seeds_by_condition": expected_seeds,
        "diagnostic_seeds": sorted(diagnostic_seeds),
        "diagnostic_stride_s": float(args.diag_stride),
        "diagnostic_max_pairs": int(args.diag_max_pairs),
        "conditions": conditions,
        "condition_count": len(conditions),
        "task_count": task_count,
        "method_run_count": method_run_count,
        "requested_workers": int(args.workers),
        "effective_worker_limit": int(effective_workers),
        "blas_threads_per_process": 1,
        "B_anchor_policy": "balanced mixed-height two-anchor pairs",
    }
    _atomic_json(output / "experiment_manifest.json", manifest)

    tasks = []
    resume = not args.no_resume
    for condition in run_conditions:
        methods = methods_for_condition(condition)
        for seed in expected_seeds[condition["condition_id"]]:
            path = _task_path(str(output), condition["condition_id"], seed)
            if resume and _is_complete(path):
                continue
            diagnostics = (
                not args.disable_diagnostics
                and condition["diagnostics_enabled"]
                and seed in diagnostic_seeds
            )
            tasks.append({
                "condition": condition, "seed": seed, "output": str(output),
                "methods": methods, "duration_s": args.duration,
                "burn_in_s": args.burn_in, "diagnostics": diagnostics,
                "diag_stride_s": args.diag_stride,
                "diag_max_pairs": args.diag_max_pairs,
            })
    if args.max_tasks is not None:
        tasks = tasks[:max(0, int(args.max_tasks))]

    print(f"[experiment-v2] manifest conditions={len(conditions)}, "
          f"run conditions={len(run_conditions)}, expected manifest tasks={task_count}, "
          f"pending tasks={len(tasks)}, method-runs={method_run_count}", flush=True)
    print(f"[experiment-v2] workers={effective_workers}; BLAS threads/worker=1", flush=True)
    print(f"[experiment-v2] nominal=C008; E1 grid=25; unique conditions=40", flush=True)
    if args.dry_run or not tasks:
        print(f"[experiment-v2] manifest -> {output / 'experiment_manifest.json'}", flush=True)
        return 0

    workers = max(1, min(effective_workers, len(tasks)))
    cpu = os.cpu_count() or 1
    if workers > cpu:
        print(f"[experiment-v2] note: workers={workers} exceeds logical CPUs={cpu}", flush=True)

    done = failed = 0
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            done += 1
            failed += int(not result["ok"])
            message = "ok" if result["ok"] else result.get("error", "failed")
            print(f"[experiment-v2] {done}/{len(tasks)} "
                  f"{result['condition_id']} seed={result['seed']} {message}", flush=True)

    print(f"[experiment-v2] complete: tasks={done}, failures={failed}, output={output}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
