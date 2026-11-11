#!/usr/bin/env python3
"""Audit Figure 1 sensor-ablation references without rerunning simulations.

Compares:
  <reference-root>/runs/C008/seed_XXXX.json
  <ablation-root>/runs/C008/seed_XXXX.json

The original ablation runner compares motion_summary by an exact SHA-256 hash.
That is intentionally strict, but even tiny cross-platform floating-point
differences make the hash fail. This script reports exact and tolerance-based
matches, per-field deltas, configuration equality, and topology equality.

It never modifies existing results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_FIELDS = (
    "actual_mean_3d_speed",
    "actual_std_3d_speed",
    "actual_speed_p05",
    "actual_speed_p95",
    "z_span_mean",
    "z_span_min",
    "z_span_final",
)


def parse_seeds(text: str) -> List[int]:
    text = text.strip()
    if ":" in text:
        first, last = text.split(":", 1)
        seeds = list(range(int(first), int(last) + 1))
    else:
        seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise ValueError("No seeds were specified.")
    return seeds


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def canonical(value: Any) -> Any:
    """Normalize JSON-like structures for deterministic equality checks."""
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return canonical(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        # Preserve values; this function is not a tolerance comparison.
        return float(value)
    return value


def exact_json_equal(left: Any, right: Any) -> bool:
    return canonical(left) == canonical(right)


def scalar_delta(
    reference: Any,
    ablation: Any,
    *,
    rtol: float,
    atol: float,
) -> Tuple[bool, bool, float, float]:
    """Return exact, close, absolute delta, relative delta."""
    try:
        ref = float(reference)
        abl = float(ablation)
    except (TypeError, ValueError):
        return False, False, math.nan, math.nan

    if not (math.isfinite(ref) and math.isfinite(abl)):
        exact = (not math.isfinite(ref)) and (not math.isfinite(abl))
        return exact, exact, 0.0 if exact else math.inf, 0.0 if exact else math.inf

    abs_delta = abs(abl - ref)
    scale = max(abs(ref), abs(abl), np.finfo(float).tiny)
    rel_delta = abs_delta / scale
    exact = abl == ref
    close = bool(np.isclose(abl, ref, rtol=rtol, atol=atol))
    return exact, close, abs_delta, rel_delta


def flatten_numeric(
    value: Any,
    prefix: str = "",
) -> Iterable[Tuple[str, float]]:
    """Yield numeric leaves from a nested JSON object."""
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_numeric(value[key], child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from flatten_numeric(item, child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def compare_topology(
    reference: Any,
    ablation: Any,
    *,
    rtol: float,
    atol: float,
) -> Tuple[bool, float, float]:
    """Tolerance comparison for all numeric topology-summary leaves."""
    ref_values = dict(flatten_numeric(reference))
    abl_values = dict(flatten_numeric(ablation))
    if set(ref_values) != set(abl_values):
        return False, math.inf, math.inf

    max_abs = 0.0
    max_rel = 0.0
    all_close = True
    for key in ref_values:
        _, close, abs_delta, rel_delta = scalar_delta(
            ref_values[key], abl_values[key], rtol=rtol, atol=atol
        )
        all_close = all_close and close
        max_abs = max(max_abs, abs_delta)
        max_rel = max(max_rel, rel_delta)
    return all_close, max_abs, max_rel


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit C008 reference matching for Figure 1 ablation."
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("results_large_scale_120s"),
        help="Root containing formal runs/C008 (default: results_large_scale_120s)",
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("results_large_scale_120s/figure1_sensor_ablation"),
        help="Root containing ablation runs/C008",
    )
    parser.add_argument("--condition", default="C008")
    parser.add_argument("--seeds", default="1010:1039")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for motion/topology summaries (default: 1e-8)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="Absolute tolerance for motion/topology summaries (default: 1e-10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; default is <ablation-root>/reference_audit",
    )
    args = parser.parse_args(argv)

    reference_root = args.reference_root.expanduser().resolve()
    ablation_root = args.ablation_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else ablation_root / "reference_audit"
    )
    output.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    seed_summary: List[Dict[str, Any]] = []

    for seed in seeds:
        reference_path = (
            reference_root / "runs" / args.condition / f"seed_{seed:04d}.json"
        )
        ablation_path = (
            ablation_root / "runs" / args.condition / f"seed_{seed:04d}.json"
        )
        if not reference_path.is_file() or not ablation_path.is_file():
            missing.append(
                {
                    "seed": seed,
                    "reference_exists": reference_path.is_file(),
                    "ablation_exists": ablation_path.is_file(),
                    "reference_path": str(reference_path),
                    "ablation_path": str(ablation_path),
                }
            )
            continue

        reference = load_json(reference_path)
        ablation = load_json(ablation_path)

        ref_motion = reference.get("motion_summary", {})
        abl_motion = ablation.get("motion_summary", {})
        fields = sorted(set(DEFAULT_FIELDS) | set(ref_motion) | set(abl_motion))

        motion_exact = True
        motion_close = True
        seed_max_abs = 0.0
        seed_max_rel = 0.0

        for field in fields:
            exact, close, abs_delta, rel_delta = scalar_delta(
                ref_motion.get(field),
                abl_motion.get(field),
                rtol=args.rtol,
                atol=args.atol,
            )
            motion_exact = motion_exact and exact
            motion_close = motion_close and close
            if math.isfinite(abs_delta):
                seed_max_abs = max(seed_max_abs, abs_delta)
            else:
                seed_max_abs = math.inf
            if math.isfinite(rel_delta):
                seed_max_rel = max(seed_max_rel, rel_delta)
            else:
                seed_max_rel = math.inf

            rows.append(
                {
                    "seed": seed,
                    "field": field,
                    "reference": ref_motion.get(field),
                    "ablation": abl_motion.get(field),
                    "absolute_delta": abs_delta,
                    "relative_delta": rel_delta,
                    "exact_match": exact,
                    "tolerance_match": close,
                }
            )

        config_hash_equal = (
            reference.get("config_hash") == ablation.get("config_hash")
        )
        condition_equal = exact_json_equal(
            reference.get("condition"), ablation.get("condition")
        )
        config_record_equal = exact_json_equal(
            reference.get("config"), ablation.get("config")
        )
        topology_exact = exact_json_equal(
            reference.get("topology_summary"),
            ablation.get("topology_summary"),
        )
        topology_close, topology_max_abs, topology_max_rel = compare_topology(
            reference.get("topology_summary", {}),
            ablation.get("topology_summary", {}),
            rtol=args.rtol,
            atol=args.atol,
        )

        seed_summary.append(
            {
                "seed": seed,
                "config_hash_equal": config_hash_equal,
                "condition_equal": condition_equal,
                "config_record_equal": config_record_equal,
                "motion_exact": motion_exact,
                "motion_tolerance_match": motion_close,
                "motion_max_absolute_delta": seed_max_abs,
                "motion_max_relative_delta": seed_max_rel,
                "topology_exact": topology_exact,
                "topology_tolerance_match": topology_close,
                "topology_max_absolute_delta": topology_max_abs,
                "topology_max_relative_delta": topology_max_rel,
            }
        )

    field_csv = output / "motion_field_deltas.csv"
    with field_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        columns = [
            "seed",
            "field",
            "reference",
            "ablation",
            "absolute_delta",
            "relative_delta",
            "exact_match",
            "tolerance_match",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    seed_csv = output / "seed_match_summary.csv"
    with seed_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        columns = [
            "seed",
            "config_hash_equal",
            "condition_equal",
            "config_record_equal",
            "motion_exact",
            "motion_tolerance_match",
            "motion_max_absolute_delta",
            "motion_max_relative_delta",
            "topology_exact",
            "topology_tolerance_match",
            "topology_max_absolute_delta",
            "topology_max_relative_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(seed_summary)

    checked = len(seed_summary)
    counts = {
        "requested": len(seeds),
        "checked": checked,
        "missing": len(missing),
        "config_hash_equal": sum(bool(x["config_hash_equal"]) for x in seed_summary),
        "condition_equal": sum(bool(x["condition_equal"]) for x in seed_summary),
        "config_record_equal": sum(bool(x["config_record_equal"]) for x in seed_summary),
        "motion_exact": sum(bool(x["motion_exact"]) for x in seed_summary),
        "motion_tolerance_match": sum(
            bool(x["motion_tolerance_match"]) for x in seed_summary
        ),
        "topology_exact": sum(bool(x["topology_exact"]) for x in seed_summary),
        "topology_tolerance_match": sum(
            bool(x["topology_tolerance_match"]) for x in seed_summary
        ),
    }

    max_motion_abs = max(
        (float(x["motion_max_absolute_delta"]) for x in seed_summary),
        default=math.nan,
    )
    max_motion_rel = max(
        (float(x["motion_max_relative_delta"]) for x in seed_summary),
        default=math.nan,
    )

    if checked == 0:
        classification = "NO_COMPARABLE_FILES"
        recommendation = (
            "Check the two root paths and condition ID. No simulations need "
            "to be rerun until the paths are corrected."
        )
        exit_code = 2
    elif counts["motion_exact"] == checked and counts["topology_exact"] == checked:
        classification = "EXACT_MATCH"
        recommendation = (
            "The ablation and formal C008 references match exactly. The prior "
            "failure should not occur with these files."
        )
        exit_code = 0
    elif (
        counts["motion_tolerance_match"] == checked
        and counts["topology_tolerance_match"] == checked
        and counts["config_hash_equal"] == checked
    ):
        classification = "NUMERICAL_ONLY_DIFFERENCE"
        recommendation = (
            "Differences are below the selected tolerances and are consistent "
            "with platform/library floating-point variation. Keep the completed "
            "ablation data; use tolerance-based reference checking."
        )
        exit_code = 0
    else:
        classification = "MATERIAL_REFERENCE_MISMATCH"
        recommendation = (
            "At least one seed differs materially. Do not combine the old C008 "
            "four-method aggregate with this ablation as a strictly paired "
            "six-method Figure 1. Keep the completed ablation, and rerun only "
            "C008 for centralized, dual_layer, ci_trace and ci_det using the "
            "current source; the full 33-condition matrix does not need rerunning."
        )
        exit_code = 1

    report = {
        "classification": classification,
        "recommendation": recommendation,
        "reference_root": str(reference_root),
        "ablation_root": str(ablation_root),
        "condition": args.condition,
        "seeds": seeds,
        "rtol": args.rtol,
        "atol": args.atol,
        "counts": counts,
        "max_motion_absolute_delta": max_motion_abs,
        "max_motion_relative_delta": max_motion_rel,
        "missing": missing,
        "per_seed": seed_summary,
    }

    report_json = output / "reference_audit_report.json"
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"[audit] classification={classification}")
    print(
        "[audit] checked={checked}, config_hash={config}/{checked}, "
        "motion_exact={motion_exact}/{checked}, "
        "motion_tolerance={motion_tol}/{checked}, "
        "topology_tolerance={topology_tol}/{checked}".format(
            checked=checked,
            config=counts["config_hash_equal"],
            motion_exact=counts["motion_exact"],
            motion_tol=counts["motion_tolerance_match"],
            topology_tol=counts["topology_tolerance_match"],
        )
    )
    print(f"[audit] max motion abs delta={max_motion_abs:.12g}")
    print(f"[audit] max motion rel delta={max_motion_rel:.12g}")
    print(f"[audit] {recommendation}")
    print(f"[audit] report -> {report_json}")
    print(f"[audit] seed CSV -> {seed_csv}")
    print(f"[audit] field CSV -> {field_csv}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
