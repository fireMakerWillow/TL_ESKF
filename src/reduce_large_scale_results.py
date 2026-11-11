#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduce large analysis CSV files into smaller uploadable files.

The default input directory is the analysis_results folder under:
C:\\Users\\Administrator\\Documents\\Python\\NEES\\results_large_scale_120s

Dependency: Python standard library only.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT = Path(
    r"C:\Users\Administrator\Documents\Python\NEES"
    r"\results_large_scale_120s\analysis_results"
)

GROUP_ALL_MARKER = "metrics.groups.all."
CLASS_MARKERS = (
    "metrics.groups.class_A.",
    "metrics.groups.class_B.",
    "metrics.groups.class_C.",
)
RESOURCE_MARKER = "theoretical_resources."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce large-scale analysis CSVs for upload/review."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing aggregate_metrics.csv and paired_vs_centralized.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <input>/reduced_for_review",
    )
    parser.add_argument(
        "--top-anomalies",
        type=int,
        default=500,
        help="Number of largest anomaly rows to retain.",
    )
    return parser.parse_args()


def normalize_path(value: str) -> str:
    return (value or "").replace("\\", "/").strip()


def is_overall(metric_path: str) -> bool:
    return GROUP_ALL_MARKER in metric_path


def is_class(metric_path: str) -> bool:
    return any(marker in metric_path for marker in CLASS_MARKERS)


def is_resource(metric_path: str) -> bool:
    return RESOURCE_MARKER in metric_path


def safe_float(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def open_writer(path: Path, fieldnames: List[str]):
    handle = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def anomaly_score(row: Dict[str, str]) -> float:
    kind = (row.get("metric_kind") or "").lower()
    maximum = safe_float(row.get("max"))
    mean = safe_float(row.get("mean"))

    if kind in {"rmse", "rmte"}:
        return maximum if maximum is not None else (mean or 0.0)

    if kind == "tmnnees":
        values = [x for x in (maximum, mean) if x is not None]
        return max((abs(x - 1.0) for x in values), default=0.0)

    return -1.0


def reduce_aggregate(source: Path, output: Path, top_n: int) -> Dict[str, int]:
    counts = {"total": 0, "overall": 0, "classes": 0, "resources": 0}
    anomaly_heap: List[Tuple[float, int, Dict[str, str]]] = []

    with source.open("r", newline="", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise RuntimeError(f"No header found in {source}")
        fields = list(reader.fieldnames)

        handles = []
        try:
            h_all, w_all = open_writer(output / "aggregate_overall.csv", fields)
            h_cls, w_cls = open_writer(output / "aggregate_classes.csv", fields)
            h_res, w_res = open_writer(output / "aggregate_resources.csv", fields)
            handles.extend([h_all, h_cls, h_res])

            for index, row in enumerate(reader):
                counts["total"] += 1
                path = normalize_path(row.get("metric_path", ""))

                if is_overall(path):
                    w_all.writerow(row)
                    counts["overall"] += 1

                if is_class(path):
                    w_cls.writerow(row)
                    counts["classes"] += 1

                if is_resource(path):
                    w_res.writerow(row)
                    counts["resources"] += 1

                kind = (row.get("metric_kind") or "").lower()
                if kind in {"rmse", "rmte", "tmnnees"}:
                    score = anomaly_score(row)
                    item = (score, index, dict(row))
                    if len(anomaly_heap) < top_n:
                        heapq.heappush(anomaly_heap, item)
                    elif score > anomaly_heap[0][0]:
                        heapq.heapreplace(anomaly_heap, item)
        finally:
            for handle in handles:
                handle.close()

    anomaly_rows = [
        row for _, _, row in sorted(anomaly_heap, key=lambda item: item[0], reverse=True)
    ]
    anomaly_fields = ["anomaly_score"] + fields
    with (output / "aggregate_top_anomalies.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=anomaly_fields, extrasaction="ignore")
        writer.writeheader()
        for row in anomaly_rows:
            out = dict(row)
            out["anomaly_score"] = anomaly_score(row)
            writer.writerow(out)

    return counts


def reduce_paired(source: Path, output: Path) -> Dict[str, int]:
    counts = {"total": 0, "overall": 0, "classes": 0, "resources": 0}

    with source.open("r", newline="", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise RuntimeError(f"No header found in {source}")
        fields = list(reader.fieldnames)

        handles = []
        try:
            h_all, w_all = open_writer(output / "paired_overall.csv", fields)
            h_cls, w_cls = open_writer(output / "paired_classes.csv", fields)
            h_res, w_res = open_writer(output / "paired_resources.csv", fields)
            handles.extend([h_all, h_cls, h_res])

            for row in reader:
                counts["total"] += 1
                path = normalize_path(row.get("metric_path", ""))

                if is_overall(path):
                    w_all.writerow(row)
                    counts["overall"] += 1

                if is_class(path):
                    w_cls.writerow(row)
                    counts["classes"] += 1

                if is_resource(path):
                    w_res.writerow(row)
                    counts["resources"] += 1
        finally:
            for handle in handles:
                handle.close()

    return counts


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else input_dir / "reduced_for_review"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = input_dir / "aggregate_metrics.csv"
    paired = input_dir / "paired_vs_centralized.csv"

    missing = [str(path) for path in (aggregate, paired) if not path.exists()]
    if missing:
        print("[ERROR] Missing required file(s):")
        for path in missing:
            print("  ", path)
        return 2

    print(f"[reduce] input:  {input_dir}")
    print(f"[reduce] output: {output_dir}")

    aggregate_counts = reduce_aggregate(
        aggregate, output_dir, max(1, args.top_anomalies)
    )
    paired_counts = reduce_paired(paired, output_dir)

    summary = [
        "Large-scale result reduction summary",
        "",
        f"Input: {input_dir}",
        "",
        "aggregate_metrics.csv:",
        f"  total rows:     {aggregate_counts['total']}",
        f"  overall rows:   {aggregate_counts['overall']}",
        f"  class rows:     {aggregate_counts['classes']}",
        f"  resource rows:  {aggregate_counts['resources']}",
        "",
        "paired_vs_centralized.csv:",
        f"  total rows:     {paired_counts['total']}",
        f"  overall rows:   {paired_counts['overall']}",
        f"  class rows:     {paired_counts['classes']}",
        f"  resource rows:  {paired_counts['resources']}",
    ]
    (output_dir / "reduction_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    zip_path = output_dir / "reduced_for_review.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path != zip_path:
                archive.write(path, arcname=path.name)

    print("")
    print("[reduce] complete")
    print(f"[reduce] ZIP: {zip_path}")
    print(f"[reduce] ZIP size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
