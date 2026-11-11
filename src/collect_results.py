#!/usr/bin/env python3
"""Collect a large-scale results tree into a small set of tidy CSV tables.

Replaces the ad-hoc reduction path.  Design rules:

1.  **Per-seed, never pre-aggregated.**  Every row keeps ``condition_id`` and
    ``seed``, so failure rates, paired comparisons and robust statistics can
    all be recomputed downstream.  The previous pipeline only exported
    mean/median/std, which made it impossible to re-derive failure rates or
    do paired analysis after the fact.
2.  **Long format, one value per row.**  A generic flattener walks whatever
    nesting the runner produced, so a schema change adds rows instead of
    breaking the collector.
3.  **Metadata is never dropped.**  Condition parameters (K, class sizes,
    topology degree, sync interval, target speed, maneuver sigma) are written
    once into ``conditions.csv`` and joined by ``condition_id``; they are not
    silently left empty as they were before.
4.  **Failures are data.**  ``runs.csv`` records every expected
    (condition, seed) pair including missing and failed ones, so a short seed
    count can never go unnoticed again.
5.  **Uploadable.**  ``--bundle`` writes one zip; ``--level core`` keeps the
    block-level metrics only, which is what analysis actually needs.

Usage
-----
    python collect_results.py --results-root results_large_scale_120s \
                              --output analysis_tables --level core --bundle

    # everything, including per-axis components and RMTE
    python collect_results.py --results-root ... --level full --bundle
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = "2.0"

# Metric leaves that "core" keeps.  RMTE is collected in full mode only.
CORE_METRICS = ("rmse", "tmnnees")
FULL_METRICS = ("rmse", "rmte", "tmnnees", "nees", "anees")

# Methods that reach the paper.  ci_det is deliberately absent: the
# determinant-optimal CI weight performed far worse than the trace-optimal one
# in the first campaign and CI-trace is the sole CI representative.  Any method
# present in the results but not listed here is skipped and *reported* in
# manifest.json under "excluded_methods" -- never dropped silently.  Use
# --methods all to collect everything.
PAPER_METHODS = ("centralized", "dual_layer", "ci_trace", "local_only",
                 "imu_only", "uwb_only")

# Condition fields promoted to columns in conditions.csv.  Anything else the
# runner puts in the condition dict is preserved in the `extra_json` column.
CONDITION_FIELDS = (
    "condition_id", "experiment_module", "K", "num_entities",
    "class_sizes", "num_A", "num_B", "num_C",
    "target_speed", "maneuver_accel_std", "sync_interval",
    "topology", "entity_topology", "topology_type", "knn_k", "topology_k",
    "duration_s", "burn_in_s",
)


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def flatten(obj: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Yield ``(dotted_path, value)`` for every numeric leaf in ``obj``.

    Lists are indexed as ``name[i]``.  Non-numeric leaves are skipped; strings
    and booleans are handled by the callers that need them.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from flatten(value, f"{prefix}[{index}]")
    elif isinstance(obj, bool):
        yield prefix, float(obj)
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)


def parse_metric_path(path: str) -> dict[str, str]:
    """Split ``groups.class_C.states.position.components.x.rmse`` into columns.

    Falls back to leaving ``group``/``block``/``component`` empty and keeping
    the raw path, so an unrecognised shape is still exported rather than lost.
    """
    parts = path.split(".")
    out = {"group": "", "block": "", "component": "", "metric": "",
           "metric_path": path}
    if len(parts) >= 2 and parts[0] == "groups":
        out["group"] = parts[1]
        rest = parts[2:]
        if rest and rest[0] == "states" and len(rest) >= 3:
            out["block"] = rest[1]
            if rest[2] == "block" and len(rest) >= 4:
                out["component"] = "block"
                out["metric"] = rest[3]
            elif rest[2] == "components" and len(rest) >= 5:
                out["component"] = rest[3]
                out["metric"] = rest[4]
        elif rest and rest[0] == "full_state" and len(rest) >= 2:
            out["block"] = "full_state"
            out["component"] = "block"
            out["metric"] = rest[1]
    return out


def condition_signature(condition: dict, config: dict) -> dict[str, Any]:
    """Normalise the condition record into a stable, fully populated row."""
    row: dict[str, Any] = {}
    merged = dict(condition)
    for key in ("duration_s", "burn_in_s"):
        if key in config:
            merged.setdefault(key, config[key])

    # K / num_entities are written under different names by different scripts.
    k = merged.get("K", merged.get("num_entities"))
    merged["K"] = k
    merged["num_entities"] = k

    # class_sizes -> num_A / num_B / num_C.  The old pipeline left these empty,
    # which made the composition experiment unlabellable.
    sizes = merged.get("class_sizes")
    if sizes is None:
        class_map = config.get("entity_class_map")
        if class_map:
            counts = Counter(str(c).upper()[-1] for c in class_map)
            sizes = [counts.get("A", 0), counts.get("B", 0), counts.get("C", 0)]
    if sizes is not None:
        sizes = list(sizes)
        merged["class_sizes"] = "/".join(str(int(s)) for s in sizes)
        for name, value in zip(("num_A", "num_B", "num_C"), sizes):
            merged[name] = int(value)

    # topology: normalise name and exact degree.
    topo = (merged.get("topology") or merged.get("entity_topology")
            or merged.get("topology_type") or "")
    degree = merged.get("knn_k", merged.get("topology_k"))
    if isinstance(topo, str) and topo.startswith("knn") and degree is None:
        tail = topo.split("_")[-1]
        if tail.isdigit():
            degree = int(tail)
    if str(topo) == "full" and k:
        degree = int(k) - 1
    merged["topology"] = "full" if str(topo) == "full" else "circular"
    merged["topology_k"] = degree
    merged["knn_k"] = degree
    if k and degree is not None:
        merged["edges_per_epoch"] = int(k) * int(degree) // 2

    for field in CONDITION_FIELDS:
        row[field] = merged.get(field, "")
    row["edges_per_epoch"] = merged.get("edges_per_epoch", "")
    extra = {key: value for key, value in merged.items()
             if key not in row and key != "edges_per_epoch"}
    row["extra_json"] = json.dumps(extra, sort_keys=True, ensure_ascii=False)
    return row


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
class Table:
    """Streaming CSV writer, optionally gzipped."""

    def __init__(self, path: Path, columns: list[str], compress: bool):
        self.path = path.with_suffix(path.suffix + ".gz") if compress else path
        self.columns = columns
        opener = gzip.open if compress else open
        self._handle = opener(self.path, "wt", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=columns,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self.rows = 0

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self.rows += 1

    def close(self) -> None:
        self._handle.close()


# --------------------------------------------------------------------------
# main collection
# --------------------------------------------------------------------------
def collect(results_root: Path, output: Path, level: str,
            compress: bool, methods_keep: tuple[str, ...] | None) -> dict:
    runs_dir = results_root / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"no runs/ directory under {results_root}")
    output.mkdir(parents=True, exist_ok=True)

    experiment_manifest_path = results_root / "experiment_manifest.json"
    experiment_manifest = {}
    if experiment_manifest_path.is_file():
        experiment_manifest = json.loads(
            experiment_manifest_path.read_text(encoding="utf-8"))
    expected_seeds = {
        str(cid): [int(seed) for seed in seeds]
        for cid, seeds in (experiment_manifest.get(
            "expected_seeds_by_condition", {}) or {}).items()
    }
    manifest_conditions = {
        str(c.get("condition_id")): c
        for c in (experiment_manifest.get("conditions", []) or [])
        if c.get("condition_id")
    }

    keep_metrics = CORE_METRICS if level == "core" else FULL_METRICS
    keep_components = ("block",) if level == "core" else None

    metrics = Table(output / "metrics.csv",
                    ["condition_id", "seed", "method", "group", "block",
                     "component", "metric", "value", "metric_path"], compress)
    resources = Table(output / "resources.csv",
                      ["condition_id", "seed", "method", "key", "value"],
                      compress)
    motion = Table(output / "motion.csv",
                   ["condition_id", "seed", "key", "value"], compress)
    topology = Table(output / "topology.csv",
                     ["condition_id", "seed", "key", "value"], compress)
    diagnostics = Table(output / "diagnostics.csv",
                        ["condition_id", "seed", "sample_point", "scope",
                         "key", "value"], compress)

    condition_rows: dict[str, dict] = {}
    for cid, condition in manifest_conditions.items():
        condition_rows[cid] = condition_signature(condition, {
            "duration_s": experiment_manifest.get("duration_s", ""),
            "burn_in_s": experiment_manifest.get("burn_in_s", ""),
        })
    run_rows: list[dict] = []
    seen_runs: set[tuple[str, int]] = set()
    counts = Counter()
    excluded: Counter = Counter()
    kept_methods: Counter = Counter()

    for condition_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for seed_file in sorted(condition_dir.glob("seed_*.json")):
            cid_from_path = condition_dir.name
            seed_from_path = int(seed_file.stem.split("_")[-1])
            seen_runs.add((cid_from_path, seed_from_path))
            try:
                payload = json.loads(seed_file.read_text(encoding="utf-8"))
            except Exception as exc:                      # corrupt file
                run_rows.append({"condition_id": cid_from_path,
                                 "seed": seed_from_path,
                                 "status": "unreadable", "error_type":
                                 type(exc).__name__, "error": str(exc),
                                 "methods": "", "path": str(seed_file)})
                counts["unreadable"] += 1
                continue

            condition = payload.get("condition", {}) or {}
            cid = str(condition.get("condition_id", condition_dir.name))
            seed = int(payload.get("seed", seed_file.stem.split("_")[-1]))
            status = str(payload.get("status", "unknown"))
            counts[status] += 1

            if cid not in condition_rows:
                condition_rows[cid] = condition_signature(
                    condition, payload.get("config", {}) or {})

            methods = payload.get("methods", {}) or {}
            run_rows.append({
                "condition_id": cid, "seed": seed, "status": status,
                "error_type": payload.get("error_type", ""),
                "error": payload.get("error", ""),
                "methods": ";".join(sorted(methods)),
                "path": str(seed_file.relative_to(results_root)),
            })
            if status != "complete":
                continue

            for method, block in methods.items():
                if methods_keep is not None and method not in methods_keep:
                    excluded[method] += 1
                    continue
                kept_methods[method] += 1
                for path, value in flatten(block.get("metrics", {}) or {}):
                    parsed = parse_metric_path(path)
                    if parsed["metric"] and parsed["metric"] not in keep_metrics:
                        continue
                    if (keep_components is not None and parsed["component"]
                            and parsed["component"] not in keep_components):
                        continue
                    if not parsed["metric"]:
                        continue          # entity index lists etc.
                    metrics.write({"condition_id": cid, "seed": seed,
                                   "method": method, "value": value, **parsed})
                for path, value in flatten(
                        block.get("theoretical_resources", {}) or {}):
                    resources.write({"condition_id": cid, "seed": seed,
                                     "method": method, "key": path,
                                     "value": value})

            for path, value in flatten(payload.get("motion_summary", {}) or {}):
                motion.write({"condition_id": cid, "seed": seed,
                              "key": path, "value": value})
            for source in ("topology_summary", "topology_policy"):
                for path, value in flatten(payload.get(source, {}) or {}):
                    topology.write({"condition_id": cid, "seed": seed,
                                    "key": f"{source}.{path}", "value": value})

            # Structural diagnostics are optional and only present on the
            # conditions where they were enabled.
            diag = payload.get("diagnostics_summary") or payload.get("diag_summary")
            if diag:
                for sample_point, scopes in diag.items():
                    if not isinstance(scopes, dict):
                        continue
                    for scope, values in scopes.items():
                        if not isinstance(values, dict):
                            for path, value in flatten({scope: values}):
                                diagnostics.write({
                                    "condition_id": cid, "seed": seed,
                                    "sample_point": sample_point,
                                    "scope": "all", "key": path,
                                    "value": value})
                            continue
                        for path, value in flatten(values):
                            diagnostics.write({
                                "condition_id": cid, "seed": seed,
                                "sample_point": sample_point, "scope": scope,
                                "key": path, "value": value})

    # Materialise expected-but-absent runs from the experiment manifest.
    for cid, seeds in sorted(expected_seeds.items()):
        for seed in seeds:
            if (cid, int(seed)) in seen_runs:
                continue
            run_rows.append({
                "condition_id": cid, "seed": int(seed), "status": "missing",
                "error_type": "", "error": "expected result file is absent",
                "methods": "", "path": str(
                    Path("runs") / cid / f"seed_{int(seed):04d}.json"),
            })
            counts["missing"] += 1

    for table in (metrics, resources, motion, topology, diagnostics):
        table.close()

    # conditions.csv
    cond_path = output / "conditions.csv"
    with open(cond_path, "w", newline="", encoding="utf-8") as handle:
        columns = list(CONDITION_FIELDS) + ["edges_per_epoch", "extra_json"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for cid in sorted(condition_rows):
            writer.writerow(condition_rows[cid])

    # runs.csv, plus a per-condition completeness check
    runs_path = output / "runs.csv"
    with open(runs_path, "w", newline="", encoding="utf-8") as handle:
        columns = ["condition_id", "seed", "status", "error_type", "error",
                   "methods", "path"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(run_rows, key=lambda r: (r["condition_id"],
                                                   str(r["seed"]))):
            writer.writerow(row)

    per_condition = Counter(r["condition_id"] for r in run_rows
                            if r["status"] == "complete")
    if expected_seeds:
        expected_counts = {cid: len(seeds) for cid, seeds in expected_seeds.items()}
        short = {
            cid: {"complete": int(per_condition.get(cid, 0)),
                  "expected": int(expected_count)}
            for cid, expected_count in sorted(expected_counts.items())
            if int(per_condition.get(cid, 0)) < int(expected_count)
        }
        expected = None
    else:
        expected = max(per_condition.values()) if per_condition else 0
        expected_counts = {cid: expected for cid in condition_rows}
        short = {
            cid: {"complete": int(n), "expected": int(expected)}
            for cid, n in sorted(per_condition.items()) if n < expected
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "results_root": str(results_root),
        "level": level,
        "compressed": compress,
        "conditions": len(condition_rows),
        "status_counts": dict(counts),
        "methods_collected": dict(sorted(kept_methods.items())),
        "methods_kept_filter": (list(methods_keep) if methods_keep
                                else "all"),
        "excluded_methods": dict(sorted(excluded.items())),
        "complete_seeds_per_condition": dict(sorted(per_condition.items())),
        "expected_seeds_per_condition": dict(sorted(expected_counts.items())),
        "modal_seed_count": expected,
        "short_conditions": short,
        "tables": {},
    }
    for name, table in (("metrics", metrics), ("resources", resources),
                        ("motion", motion), ("topology", topology),
                        ("diagnostics", diagnostics)):
        manifest["tables"][name] = {
            "file": table.path.name, "rows": table.rows,
            "bytes": table.path.stat().st_size}
    for name, path in (("conditions", cond_path), ("runs", runs_path)):
        manifest["tables"][name] = {"file": path.name,
                                    "bytes": path.stat().st_size}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def bundle(output: Path, manifest: dict) -> Path:
    archive = output / "analysis_tables.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != archive:
                zf.write(path, path.name)
    manifest["bundle_bytes"] = archive.stat().st_size
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <results-root>/analysis_tables")
    parser.add_argument("--level", choices=("core", "full"), default="core",
                        help="core: block-level RMSE and NEES only (small). "
                             "full: every component and RMTE as well.")
    parser.add_argument("--methods", default="paper",
                        help="'paper' (default, excludes ci_det), 'all', or a "
                             "comma-separated list of estimator keys.")
    parser.add_argument("--no-gzip", action="store_true")
    parser.add_argument("--bundle", action="store_true",
                        help="also write analysis_tables.zip for upload")
    args = parser.parse_args()

    if args.methods == "all":
        methods_keep = None
    elif args.methods == "paper":
        methods_keep = PAPER_METHODS
    else:
        methods_keep = tuple(m.strip() for m in args.methods.split(",")
                             if m.strip())

    output = args.output or (args.results_root / "analysis_tables")
    manifest = collect(args.results_root, output, args.level,
                       compress=not args.no_gzip, methods_keep=methods_keep)

    print(f"schema {manifest['schema_version']}  level={manifest['level']}")
    print(f"conditions: {manifest['conditions']}   "
          f"status: {manifest['status_counts']}")
    print(f"methods collected: {sorted(manifest['methods_collected'])}")
    if manifest["excluded_methods"]:
        print(f"methods EXCLUDED: {manifest['excluded_methods']} "
              "(use --methods all to keep them)")
    total = 0
    for name, info in manifest["tables"].items():
        rows = info.get("rows")
        total += info["bytes"]
        print(f"  {name:<12} {info['file']:<24} "
              f"{'' if rows is None else f'{rows:>10,} rows'} "
              f"{info['bytes'] / 1e6:>8.2f} MB")
    print(f"  {'TOTAL':<12} {'':<24} {'':>16} {total / 1e6:>8.2f} MB")

    if manifest["short_conditions"]:
        print("\nWARNING: incomplete conditions:")
        for cid, info in manifest["short_conditions"].items():
            print(f"  {cid}: {info['complete']}/{info['expected']} complete")
        print("  -> check runs.csv for the failed or missing seeds.")

    if args.bundle:
        archive = bundle(output, manifest)
        print(f"\nbundle: {archive}  ({archive.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
