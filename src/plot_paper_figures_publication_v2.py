"""Generate the four paper figures from the schema-2.0 analysis tables.

This replaces ``plot_reduced_paper_figures.py``, which walked the raw per-seed
JSON tree.  The collector (``collect_results.py``) now emits tidy per-seed
tables, so this script only reads CSV and never parses a nested payload.

Input
-----
A directory holding the collector output::

    conditions.csv  metrics.csv[.gz]  resources.csv[.gz]  motion.csv[.gz]
    topology.csv[.gz]  runs.csv  diagnostics.csv[.gz]

Figures
-------
fig2_nominal_performance
    Nominal scenario, six methods, position RMSE and ANEES for All / A / B / C.
fig3_scenario_robustness
    Robustness to the UWB ranging interval, maneuver intensity and target
    speed.  The curves are shown directly without OLS callouts because the
    trends are visually clear and the extra annotations consume valuable space.
    Here ``T`` denotes the time between successive complete UWB ranging rounds;
    it is not a communication-clock synchronization period.
fig4_network_scaling
    Performance and resources against K, plus the cost-consistency plane.
fig5_composition_topology
    Sensitivity to class composition and to graph degree.

Two conventions that the figures depend on
------------------------------------------
1.  **Divergent seeds are dropped from RMSE statistics only.**  ANEES is always
    reported over every seed.  The decentralized baseline is overconfident by
    construction, and a NEES-threshold filter would delete precisely the
    evidence the paper is presenting.  Failure counts are annotated as ``n/N``.
2.  **Blow-ups never set the axis range.**  Several conditions push a baseline
    to a NEES of 10^4-10^6.  Plotting those on a shared linear axis would
    flatten the 1.0-1.7 band that carries the paper's actual claim, so the
    consistency panels use a log axis clipped to a readable window, and points
    above the window are drawn on the frame with their true value printed.

Circular-2 is excluded from every figure: with no anchor visible and only two
inter-entity ranges per epoch, class-C position is instantaneously rank
deficient and all architectures, including the centralized reference, diverge.
It is a scenario-observability failure, not an algorithmic one.

All figures are exported as vector PDF files with embedded TrueType fonts.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Publication constants
# ---------------------------------------------------------------------------
ARCH_METHODS = ("centralized", "dual_layer", "ci_trace", "local_only")
ABLATION_METHODS = ("uwb_only", "imu_only")
ALL_METHODS = ARCH_METHODS + ABLATION_METHODS

METHOD_LABEL = {
    "centralized": "Centralized",
    "dual_layer": "Dual-layer",
    "ci_trace": "CI",
    "local_only": "DCL",
    "uwb_only": "UWB-only",
    "imu_only": "IMU-only",
}
# Okabe-Ito, chosen so that no two architectures share a hue family.  CI and
# DCL were previously orange and vermillion, which are hard to separate in the
# categorical panels where their curves run close together.
METHOD_COLOR = {
    "centralized": "#333333",
    "dual_layer": "#0072B2",
    "ci_trace": "#E69F00",
    "local_only": "#CC79A7",
    "uwb_only": "#009E73",
    "imu_only": "#999999",
}
METHOD_MARKER = {
    "centralized": "o",
    "dual_layer": "s",
    "ci_trace": "^",
    "local_only": "v",
    "uwb_only": "P",
    "imu_only": "X",
}

GROUPS = ("all", "class_A", "class_B", "class_C")
GROUP_LABEL = {"all": "All", "class_A": "A", "class_B": "B", "class_C": "C"}

# Degree of the circular graph below which the scenario is not observable.
MIN_ADMISSIBLE_DEGREE = 3

# Standard full-width figure size for a two-column paper.  Creating each figure
# close to its final print width avoids shrinking the labels after LaTeX scales
# the graphic to \textwidth.
PAPER_FULL_WIDTH_IN = 7.16

# T is the time between successive complete UWB ranging rounds.  Calling it a
# "sync interval" can be mistaken for clock or communication synchronization.
UWB_INTERVAL_LABEL = r"UWB ranging interval $T$ (s)"


@dataclass(frozen=True)
class FailurePolicy:
    """Seed-level divergence rule, applied to RMSE statistics only.

    ``paired_rmse_ratio``
        A run this many times worse than the centralized filter *on the same
        seed and scenario* has diverged.  Being paired makes the rule
        scale-free, so it works equally at 0.08 m and at 20 m.
    ``anees_bound``
        Position ANEES has a known ideal of 1.  A run above this bound is a
        filter breakdown rather than poor tuning.
    """

    paired_rmse_ratio: float = 20.0
    anees_bound: float = 50.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read(root: Path, stem: str, required: bool = True) -> pd.DataFrame:
    for name in (f"{stem}.csv.gz", f"{stem}.csv"):
        path = root / name
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig")
    if required:
        raise FileNotFoundError(f"{stem}.csv[.gz] not found under {root}")
    return pd.DataFrame()


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    tables = {name: _read(root, name, required=name in {"conditions", "metrics"})
              for name in ("conditions", "metrics", "resources", "motion",
                           "topology", "runs", "diagnostics")}

    conditions = tables["conditions"]
    # `topology` is a name in conditions.csv; keep the human label separate.
    conditions["degree"] = pd.to_numeric(conditions["topology_k"], errors="coerce")
    conditions["topology_label"] = [
        "Full" if str(t) == "full" else f"Circular-{int(d)}" if np.isfinite(d) else "unknown"
        for t, d in zip(conditions["topology"], conditions["degree"])
    ]
    conditions["module"] = conditions["experiment_module"].fillna("").astype(str)
    tables["conditions"] = conditions
    return tables


CONDITION_COLS = ["condition_id", "module", "K", "class_sizes", "num_A", "num_B",
                  "num_C", "target_speed", "maneuver_accel_std", "sync_interval",
                  "topology", "degree", "topology_label", "edges_per_epoch"]


def build_metric_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (condition, seed, method, group, block) with rmse and anees."""
    metrics = tables["metrics"]
    metrics = metrics[metrics["group"].isin(GROUPS) & (metrics["component"] == "block")]
    wide = metrics.pivot_table(
        index=["condition_id", "seed", "method", "group", "block"],
        columns="metric", values="value", aggfunc="first").reset_index()
    wide = wide.rename(columns={"tmnnees": "anees"})
    if "anees" not in wide:
        raise RuntimeError("no TMNNEES/ANEES rows in metrics table")
    return wide.merge(tables["conditions"][CONDITION_COLS], on="condition_id", how="left")


# ---------------------------------------------------------------------------
# Failure audit
# ---------------------------------------------------------------------------
def classify_failures(frame: pd.DataFrame, policy: FailurePolicy) -> pd.DataFrame:
    """Mark diverged (condition, seed, method) triples using the paired rule."""
    data = frame.copy()
    position = data[(data["group"] == "all") & (data["block"] == "position")]
    reference = (position[position["method"] == "centralized"]
                 .set_index(["condition_id", "seed"])["rmse"])

    keys = position[["condition_id", "seed", "method", "rmse", "anees"]].copy()
    base = keys.set_index(["condition_id", "seed"]).index.map(reference)
    base = pd.Series(np.asarray(base, dtype=float), index=keys.index)

    # The divergence rule compares architectures with each other.  The sensor
    # ablations are expected to be far worse -- unaided inertial drift reaches
    # 10^3 m by construction -- so applying an architecture threshold to them
    # would delete the very baseline that motivates the fusion.
    architecture = keys["method"].isin(ARCH_METHODS)
    ratio_bad = architecture & (keys["method"] != "centralized") & np.isfinite(base) \
        & (base > 0) & (keys["rmse"] / base > policy.paired_rmse_ratio)
    anees_bad = architecture & (keys["anees"] > policy.anees_bound)
    nonfinite = ~np.isfinite(keys["rmse"]) | ~np.isfinite(keys["anees"])
    keys["diverged"] = (ratio_bad | anees_bad | nonfinite).to_numpy()

    flags = keys.set_index(["condition_id", "seed", "method"])["diverged"]
    idx = pd.MultiIndex.from_frame(data[["condition_id", "seed", "method"]])
    data["diverged"] = flags.reindex(idx).fillna(False).to_numpy()
    return data


def failure_counts(frame: pd.DataFrame) -> pd.DataFrame:
    position = frame[(frame["group"] == "all") & (frame["block"] == "position")]
    out = position.groupby(["condition_id", "method"], dropna=False).agg(
        n_seeds=("seed", "nunique"), n_diverged=("diverged", "sum")).reset_index()
    out["failure_rate"] = out["n_diverged"] / out["n_seeds"].clip(lower=1)
    return out


# ---------------------------------------------------------------------------
# Condition selectors
# ---------------------------------------------------------------------------
def _conditions(tables: dict[str, pd.DataFrame], token: str) -> pd.DataFrame:
    conditions = tables["conditions"]
    return conditions[conditions["module"].str.contains(token, na=False)].copy()


def nominal_id(tables: dict[str, pd.DataFrame]) -> str:
    conditions = tables["conditions"]
    hit = conditions[(conditions["topology"] == "full") & (conditions["K"] == 10)
                     & (conditions["num_C"] == 2) & (conditions["num_A"] == 2)
                     & (conditions["maneuver_accel_std"] == 1.0)
                     & (conditions["target_speed"] == 5.0)
                     & (conditions["sync_interval"] == 0.10)]
    if hit.empty:
        raise RuntimeError("nominal condition (K=10, 2/6/2, full, sigma=1, v=5, T=0.1) not found")
    return str(hit["condition_id"].iloc[0])


def speed_sweep(tables: dict[str, pd.DataFrame]) -> tuple[set[str], float]:
    """Conditions that differ from the nominal one only in speed (and interval).

    Selecting by parameter rather than by module name keeps the panel working
    whichever module the speed control ended up in, and lets it return an empty
    set -- rather than a misleading one-point axis -- if the campaign never
    varied the speed.  Where several sync intervals carry a speed sweep the one
    with the most speeds is used, so the panel never mixes intervals that are
    not comparable; the chosen interval is returned for the panel title.
    """
    conditions = tables["conditions"]
    nominal = conditions[conditions["condition_id"] == nominal_id(tables)].iloc[0]
    mask = np.ones(len(conditions), dtype=bool)
    for column in ("K", "num_A", "num_B", "num_C", "topology", "maneuver_accel_std"):
        mask &= (conditions[column] == nominal[column]).to_numpy()
    same = conditions[mask]
    counts = same.groupby("sync_interval")["target_speed"].nunique()
    counts = counts[counts >= 2]
    if counts.empty:
        return set(), float("nan")
    best = float(counts.idxmax()) if counts.max() > counts.get(
        nominal["sync_interval"], 0) else float(nominal["sync_interval"])
    chosen = same[same["sync_interval"] == best]
    return set(chosen["condition_id"].astype(str)), best


def drop_unobservable(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove degrees below the trilateration rank threshold (Circular-2)."""
    degree = pd.to_numeric(frame["degree"], errors="coerce")
    return frame[~(degree < MIN_ADMISSIBLE_DEGREE)].copy()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _sem(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def summarise(frame: pd.DataFrame, value: str, by: Sequence[str],
              drop_diverged: bool) -> pd.DataFrame:
    sub = frame[np.isfinite(frame[value])]
    if drop_diverged:
        sub = sub[~sub["diverged"]]
    rows = []
    for key, group in sub.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        x = group[value].to_numpy(dtype=float)
        row = dict(zip(by, key))
        row.update({"quantity": value, "n": int(x.size), "mean": float(np.mean(x)),
                    "sem": _sem(x), "median": float(np.median(x)),
                    "p10": float(np.quantile(x, 0.10)),
                    "p90": float(np.quantile(x, 0.90)), "max": float(np.max(x))})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_position(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-seed wide table of position RMSE and ANEES, one column per method."""
    position = frame[(frame["group"] == "all") & (frame["block"] == "position")]
    index = ["condition_id", "seed"] + [c for c in CONDITION_COLS if c != "condition_id"]
    rmse = position.pivot_table(index=index, columns="method", values="rmse")
    anees = position.pivot_table(index=index, columns="method", values="anees")
    rmse.columns = [f"rmse_{c}" for c in rmse.columns]
    anees.columns = [f"anees_{c}" for c in anees.columns]
    return rmse.join(anees).reset_index()


# ---------------------------------------------------------------------------
# Styling and drawing helpers
# ---------------------------------------------------------------------------
def apply_style() -> None:
    """Apply a balanced, journal-ready Matplotlib style.

    Dense six-panel figures use two columns and three rows, giving each panel
    enough horizontal room at final paper width. Text is kept moderate rather
    than enlarged globally.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 8.2,
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.1,
        "axes.titleweight": "bold",
        "axes.titlepad": 7.0,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.1,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 0.85,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })


def _clean(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, linewidth=0.55, alpha=0.22)
        ax.set_axisbelow(True)


def _panel(ax: plt.Axes, label: str, dx: float = 0.02) -> None:
    """Place a panel identifier inside the upper-left corner."""
    ax.text(
        dx, 0.98, label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=20,
        bbox=dict(
            boxstyle="round,pad=0.10",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
    )


def _reference_line(ax: plt.Axes, value: float = 1.0, label: str = "ideal = 1") -> None:
    ax.axhline(
        value,
        color="#555555",
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    if label:
        ax.text(
            0.98, value, label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#555555",
            clip_on=True,
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.08",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
        )


def _compact(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1e4:
        return f"{value:.0e}".replace("e+0", "e").replace("e+", "e")
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _ceiling(values: np.ndarray, hard: float, headroom: float = 1.5) -> float:
    """Pick a clipping ceiling that keeps the readable band as tall as possible.

    ``hard`` is the largest value still considered informative.  Anything above
    it is a blow-up whose exact height carries no information the reader needs
    from the axis, so the ceiling is placed just above the tallest *informative*
    point instead of at the hard bound; otherwise a sweep whose real range is
    1-4 would be drawn in the bottom eighth of the panel.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0) & (finite <= hard)]
    if finite.size == 0:
        return hard
    return float(min(hard, float(np.max(finite)) * headroom))


def plot_clipped(ax: plt.Axes, x: np.ndarray, y: np.ndarray, ceiling: float,
                 color: str, marker: str, label: str | None = None,
                 linewidth: float = 1.7, annotate: bool = True) -> None:
    """Draw a series on a clipped axis, printing true values for the clipped points.

    Without this, a single condition at NEES 10^5 forces a range in which the
    1.0-1.7 band that carries the paper's claim is a single pixel.  Clipped
    points are drawn as open markers on the ceiling and labelled, so no
    information is lost and the readable band is preserved.
    """
    y = np.asarray(y, dtype=float)
    over = y > ceiling
    drawn = np.where(over, ceiling, y)
    ax.plot(x, drawn, marker=marker, color=color, linewidth=linewidth,
            markersize=5.2, label=label, zorder=3)
    if not over.any():
        return
    ax.plot(np.asarray(x)[over], drawn[over], marker=marker, color=color,
            markersize=8.0, markerfacecolor="white", markeredgewidth=1.6,
            linestyle="none", zorder=4)
    if annotate:
        for xx, yy in zip(np.asarray(x)[over], y[over]):
            ax.annotate(f"{_compact(yy)}\u2191", xy=(xx, ceiling), xytext=(0, -9),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=7.2, color=color, fontweight="bold",
                        zorder=6, clip_on=False,
                        bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                                  edgecolor="none", alpha=0.8))


def _annotate_failures(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    counts: Sequence[tuple[int, int]],
    ceiling: float | None = None,
    offset: tuple[float, float] = (0.0, 7.0),
) -> None:
    """Annotate diverged-seed counts with a caller-selected offset."""
    for xx, yy, (bad, total) in zip(np.asarray(x), np.asarray(y), counts):
        if bad <= 0 or not np.isfinite(yy):
            continue
        yy = min(yy, ceiling) if ceiling is not None else yy
        ax.annotate(
            f"{bad}/{total}",
            xy=(xx, yy),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#8B0000",
            fontweight="bold",
            zorder=9,
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.07",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
        )


def _readable_log_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Label a narrow log axis at 2/3/5 as well as at the decades.

    Several consistency panels span well under one decade.  Matplotlib's
    default then prints a single tick at 10^0, which makes a curve sitting
    between 1 and 4 impossible to read off.
    """
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    target.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0), numticks=12))
    # ScalarFormatter rounds 0.5 to "0" on a log axis, which silently mislabels
    # the conservative baselines, so the ticks are formatted explicitly.
    label = FuncFormatter(lambda value, _pos: f"{value:g}")
    target.set_major_formatter(label)
    target.set_minor_formatter(label)
    ax.tick_params(axis=axis, which="minor", labelsize=7.5)


def _log_ticks(ax: plt.Axes, values: Sequence[float], axis: str = "x") -> None:
    """Label a log axis with the actual grid values rather than decades."""
    labels = [f"{v:g}" for v in values]
    if axis == "x":
        ax.set_xticks(list(values)); ax.set_xticklabels(labels)
        ax.set_xticks([], minor=True)
    else:
        ax.set_yticks(list(values)); ax.set_yticklabels(labels)
        ax.set_yticks([], minor=True)


def _save(fig: plt.Figure, outdir: Path, stem: str) -> list[str]:
    """Save one paper figure as a vector PDF with embedded fonts."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.pdf"
    metadata = {
        "Title": stem.replace("_", " "),
        "Author": "",
        "Subject": "Paper figure",
        "Creator": "Matplotlib",
    }
    try:
        fig.savefig(
            path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            metadata=metadata,
        )
    except PermissionError:
        path = path.with_name(f"{path.stem}_new{path.suffix}")
        warnings.warn(f"{stem}.pdf is locked; wrote {path.name} instead",
                      RuntimeWarning)
        fig.savefig(
            path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
            metadata=metadata,
        )
    plt.close(fig)
    return [str(path)]


# ---------------------------------------------------------------------------
# Figure 2: nominal scenario
# ---------------------------------------------------------------------------
def plot_nominal(frame: pd.DataFrame, counts: pd.DataFrame, outdir: Path,
                 cid: str) -> tuple[list[str], pd.DataFrame]:
    sub = frame[(frame["condition_id"] == cid) & (frame["block"] == "position")]
    methods = [m for m in ALL_METHODS if m in set(sub["method"])]

    rmse = summarise(sub, "rmse", ["method", "group"], drop_diverged=True)
    anees = summarise(sub, "anees", ["method", "group"], drop_diverged=False)
    source = pd.concat([rmse, anees], ignore_index=True)
    source["figure"] = "fig2_nominal_performance"
    source["condition_id"] = cid

    # The ceiling is set by the four architectures; IMU-only is three orders of
    # magnitude above them and would otherwise make the panel unreadable.
    visible = rmse[rmse["method"] != "imu_only"]["mean"].to_numpy(dtype=float)
    visible = visible[np.isfinite(visible) & (visible > 0)]
    bottom, ceiling = float(visible.min()) / 2.0, float(visible.max()) * 3.2

    fig, axes = plt.subplots(2, 2, figsize=(PAPER_FULL_WIDTH_IN, 5.15))
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.09,
        top=0.82,
        wspace=0.34,
        hspace=0.42,
    )
    panels = [
        (
            axes[0, 0], "rmse", ("all",), "Position RMSE (m)",
            "Overall position RMSE", "(a)",
        ),
        (
            axes[0, 1], "rmse", ("class_A", "class_B", "class_C"),
            "Position RMSE (m)", "Class-wise position RMSE", "(b)",
        ),
        (
            axes[1, 0], "anees", ("all",), "Position ANEES",
            "Overall position ANEES", "(c)",
        ),
        (
            axes[1, 1], "anees", ("class_A", "class_B", "class_C"),
            "Position ANEES", "Class-wise position ANEES", "(d)",
        ),
    ]

    for ax, quantity, groups, ylabel, title, tag in panels:
        table = source[(source["quantity"] == quantity) & source["group"].isin(groups)]
        x = np.arange(len(groups), dtype=float)
        total = 0.84
        width = total / max(len(methods), 1)
        for mi, method in enumerate(methods):
            values = np.asarray([
                float(table.loc[(table["method"] == method) & (table["group"] == g),
                                "mean"].iloc[0])
                if not table.loc[(table["method"] == method)
                                 & (table["group"] == g)].empty else np.nan
                for g in groups])
            positions = x - total / 2 + width / 2 + mi * width
            if quantity == "rmse":
                shown = np.where(values > ceiling, ceiling, values)
            else:
                shown = values
            # On a log axis a bar has to be drawn from the axis floor, so the
            # height is the span above that floor rather than the value itself.
            floor = bottom if quantity == "rmse" else 0.0
            heights = np.nan_to_num(shown, nan=floor) - floor
            bars = ax.bar(positions, heights, width=width * 0.9, bottom=floor,
                          color=METHOD_COLOR[method], edgecolor="white", linewidth=0.45,
                          hatch="//" if method in ABLATION_METHODS else None,
                          label=METHOD_LABEL[method], zorder=3)
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    bar.set_alpha(0.12)
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "n/a",
                            transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=6.5, rotation=90)
                elif quantity == "rmse" and value > ceiling:
                    ax.annotate(f"actual\n{_compact(value)}",
                                xy=(bar.get_x() + bar.get_width() / 2, ceiling),
                                xytext=(0, -3), textcoords="offset points",
                                ha="center", va="top", fontsize=7.4,
                                fontweight="bold", linespacing=1.0, zorder=6,
                                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                                          edgecolor="none", alpha=0.9))
        if quantity == "rmse":
            ax.set_yscale("log")
            ax.set_ylim(bottom, ceiling)
        else:
            ax.set_yscale("log")
            _reference_line(ax, 1.0, "ideal = 1")
            top = float(np.nanmax(table["mean"].to_numpy(dtype=float))) * 3.0
            ax.set_ylim(0.08, max(top, 3.0))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABEL[g] for g in groups])
        _clean(ax, "y")
        _panel(ax, tag)

    handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor="white",
                     hatch="//" if m in ABLATION_METHODS else None,
                     label=METHOD_LABEL[m]) for m in methods]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=7.3,
    )
    return _save(fig, outdir, "fig2_nominal_performance"), source


# ---------------------------------------------------------------------------
# Figure 3: scenario robustness (sync interval, maneuver intensity, speed)
# ---------------------------------------------------------------------------
def plot_maneuver_sync(frame: pd.DataFrame, tables: dict[str, pd.DataFrame],
                       counts: pd.DataFrame, outdir: Path) -> tuple[list[str], pd.DataFrame]:
    """The scenario-robustness study.

    Layout.  The top row is the paper's claim -- a penalty of a few millimetres
    that does not move with the sync interval, the maneuver intensity or the
    target speed -- and is drawn at the scale of the effect itself.  The bottom row puts that number in context by
    showing the same two quantities for the other architectures; those are one
    to three orders of magnitude larger, so they need a logarithmic axis and
    cannot share a panel with the top row without flattening it into a line.
    """
    ids = set(_conditions(tables, "E1_maneuver")["condition_id"])
    count_map = counts.set_index(["condition_id", "method"])[["n_diverged", "n_seeds"]]
    grid = frame[frame["condition_id"].isin(ids)]
    paired = paired_position(grid).dropna(subset=["rmse_centralized", "rmse_dual_layer"])
    paired["gap_mm"] = (paired["rmse_dual_layer"] - paired["rmse_centralized"]) * 1e3
    paired["anees_excess"] = paired["anees_dual_layer"] - paired["anees_centralized"]

    sigmas = sorted(paired["maneuver_accel_std"].unique())
    intervals = sorted(paired["sync_interval"].unique())
    sigma_colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(sigmas)))
    interval_colors = plt.cm.plasma(np.linspace(0.05, 0.8, len(intervals)))

    def cell(column: str, how: str = "mean") -> pd.DataFrame:
        return paired.pivot_table(index="maneuver_accel_std", columns="sync_interval",
                                  values=column, aggfunc=how)

    sem = lambda series: _sem(series.to_numpy(dtype=float))
    gap, gap_sem = cell("gap_mm"), cell("gap_mm", sem)
    excess = cell("anees_excess")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(PAPER_FULL_WIDTH_IN, 8.35),
        constrained_layout=True,
    )

    gap_top = max(6.0, float(np.nanmax(gap.to_numpy())) * 1.5)

    # (a) Accuracy penalty over the UWB ranging interval.  Maneuver intensity
    # is represented by the curve family; no OLS annotation is added because
    # the nearly flat, overlapping curves already communicate the result.
    ax = axes[0, 0]
    for i, sigma in enumerate(sigmas):
        ax.errorbar(intervals, gap.loc[sigma], yerr=gap_sem.loc[sigma], fmt="o-",
                    color=sigma_colors[i], capsize=2.4, linewidth=1.6, markersize=4.6,
                    label=rf"$\sigma_{{\mathrm{{man}}}}$ = {sigma:g}")
    ax.set_xscale("log"); _log_ticks(ax, intervals)
    ax.set_ylim(0, gap_top)
    ax.set_xlabel(UWB_INTERVAL_LABEL)
    ax.set_ylabel("Position RMSE gap (mm)")
    ax.set_title("Excess position RMSE")
    ax.legend(loc="lower left", frameon=False, ncol=3, fontsize=6.9)
    _clean(ax); _panel(ax, "(a)")

    # (b) the speed axis.  Absolute RMSE for every architecture rather than the
    # paired gap: the question here is whether the operating speed changes the
    # accuracy at all, not whether it changes the cost of decentralising, and a
    # reader who sees all four curves flat cannot suspect that the effect was
    # hidden by the normalisation.  The axis starts below the lowest curve and
    # is logarithmic, so a real speed dependence of even ten per cent would be
    # plainly visible.
    ax = axes[0, 1]
    speed_ids, speed_interval = speed_sweep(tables)
    speed_stats = pd.DataFrame()
    if speed_ids:
        # Architectures only: the sensor ablations are three orders of magnitude
        # above them and were not rerun at the other speeds anyway, so including
        # them would set the axis range and flatten the four curves that matter.
        speed_position = frame[frame["condition_id"].isin(speed_ids)
                               & (frame["group"] == "all")
                               & (frame["block"] == "position")
                               & frame["method"].isin(ARCH_METHODS)]
        speed_stats = summarise(speed_position, "rmse",
                                ["method", "target_speed", "condition_id"],
                                drop_diverged=True)
        speeds = sorted(speed_stats["target_speed"].dropna().unique())
        for method in ARCH_METHODS:
            sub = speed_stats[speed_stats["method"] == method].sort_values("target_speed")
            if sub.empty:
                continue
            ax.errorbar(sub["target_speed"], sub["mean"], yerr=sub["sem"],
                        fmt="-", marker=METHOD_MARKER[method], color=METHOD_COLOR[method],
                        capsize=2.4, linewidth=1.7, markersize=5.0,
                        label=METHOD_LABEL[method])
        values = speed_stats["mean"].to_numpy(dtype=float)
        ax.set_xscale("log"); _log_ticks(ax, speeds)
        ax.set_yscale("log")
        ax.set_ylim(float(np.nanmin(values)) * 0.45, float(np.nanmax(values)) * 1.9)
        _readable_log_axis(ax)
        ax.set_xlabel("Target speed $v$ (m/s)")
        ax.set_ylabel("Position RMSE (m)")
        ax.set_title(rf"Position RMSE vs speed ($T={speed_interval:g}$ s)")
        ax.legend(loc="upper center", frameon=False, ncol=2, fontsize=7.0)
    else:
        # No speed control in the campaign.  Rather than draw a one-point axis
        # that would imply an experiment which was never run, the panel falls
        # back to the maneuver axis of the penalty and the console says why.
        warnings.warn("no speed sweep found in conditions.csv; Figure 3(b) falls "
                      "back to the maneuver axis of the structural penalty",
                      RuntimeWarning)
        for j, interval in enumerate(intervals):
            ax.errorbar(sigmas, gap[interval], yerr=gap_sem[interval], fmt="s-",
                        color=interval_colors[j], capsize=2.4, linewidth=1.6,
                        markersize=4.6, label=rf"$T={interval:g}$ s")
        ax.set_xscale("log"); _log_ticks(ax, sigmas)
        ax.set_ylim(0, gap_top)
        ax.set_xlabel(r"Maneuver intensity $\sigma_{\mathrm{man}}$ (m/s$^2$)")
        ax.set_ylabel("Position RMSE gap (mm)")
        ax.set_title("Excess position RMSE")
        ax.legend(loc="upper center", frameon=False, ncol=3, fontsize=6.9)
    _clean(ax); _panel(ax, "(b)")

    # (c) consistency penalty, same normalisation as (a).
    ax = axes[1, 0]
    for i, sigma in enumerate(sigmas):
        ax.plot(intervals, excess.loc[sigma], "^-", color=sigma_colors[i],
                linewidth=1.6, markersize=4.6)
    ax.set_xscale("log"); _log_ticks(ax, intervals)
    ax.set_xlabel(UWB_INTERVAL_LABEL)
    ax.set_ylabel("Position ANEES gap")
    ax.set_title("Excess position ANEES")
    _clean(ax); _panel(ax, "(c)")

    reference_sigma = 1.0 if 1.0 in sigmas else sigmas[len(sigmas) // 2]
    label = rf"$\sigma_{{\mathrm{{man}}}}={reference_sigma:g}$"

    # (d) accuracy of every architecture, as a paired ratio so the panel is
    # comparable with Figure 5 and independent of the absolute error level.
    ax = axes[1, 1]
    ratios = _paired_ratio_table(grid)
    ratios = ratios[ratios["maneuver_accel_std"] == reference_sigma]
    for method in ARCH_METHODS:
        sub = ratios[ratios["method"] == method].sort_values("sync_interval")
        if sub.empty:
            continue
        ax.plot(sub["sync_interval"], sub["mean"], marker=METHOD_MARKER[method],
                color=METHOD_COLOR[method], linewidth=1.7, markersize=5.0,
                label=METHOD_LABEL[method])
    _reference_line(ax, 1.0, "Centralized = 1")
    ax.set_xscale("log"); ax.set_yscale("log"); _log_ticks(ax, intervals)
    _readable_log_axis(ax)
    ax.set_xlabel(UWB_INTERVAL_LABEL)
    ax.set_ylabel("Position RMSE / Centralized")
    ax.set_title(f"Relative position RMSE ({label})")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.0)
    _clean(ax); _panel(ax, "(d)")

    # (e) consistency of every architecture.
    ax = axes[2, 0]
    anees = summarise(grid[(grid["group"] == "all") & (grid["block"] == "position")],
                      "anees", ["method", "maneuver_accel_std", "sync_interval",
                                "condition_id"], drop_diverged=False)
    ceiling = _ceiling(
        anees.loc[anees["maneuver_accel_std"] == reference_sigma, "median"]
        .to_numpy(dtype=float), 40.0)
    for method in ARCH_METHODS:
        sub = anees[(anees["method"] == method)
                    & (anees["maneuver_accel_std"] == reference_sigma)].sort_values("sync_interval")
        if sub.empty:
            continue
        x = sub["sync_interval"].to_numpy(dtype=float)
        y = sub["median"].to_numpy(dtype=float)
        plot_clipped(ax, x, y, ceiling, METHOD_COLOR[method], METHOD_MARKER[method],
                     METHOD_LABEL[method])
        # The median deliberately hides the tail, so the diverged-seed count is
        # printed next to it; otherwise CI at long T reads as healthy.
        pairs = [tuple(count_map.loc[(cid, method)])
                 if (cid, method) in count_map.index else (0, 0)
                 for cid in sub["condition_id"]]
        failure_offsets = {
            "centralized": (-9, 5),
            "dual_layer": (9, 5),
            "ci_trace": (-9, 12),
            "local_only": (9, 12),
        }
        _annotate_failures(
            ax,
            x,
            np.minimum(y, ceiling),
            pairs,
            ceiling,
            offset=failure_offsets.get(method, (0, 7)),
        )
    _reference_line(ax, 1.0, "ideal = 1")
    ax.set_xscale("log"); ax.set_yscale("log"); _log_ticks(ax, intervals)
    ax.set_ylim(0.25, ceiling * 1.50)
    _readable_log_axis(ax)
    ax.set_xlabel(UWB_INTERVAL_LABEL)
    ax.set_ylabel("Median position ANEES")
    ax.set_title(f"Position ANEES ({label})")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.0)
    _clean(ax); _panel(ax, "(e)")

    # (f) why the maneuver axis stays flat: the coupling coefficient grows with
    # the specific force, but the attitude it acts on gets much better.
    ax = axes[2, 1]
    attitude = summarise(grid[(grid["group"] == "all") & (grid["block"] == "attitude")
                              & (grid["method"] == "centralized")],
                         "rmse", ["maneuver_accel_std", "sync_interval"],
                         drop_diverged=True)
    for j, interval in enumerate(intervals):
        sub = attitude[attitude["sync_interval"] == interval].sort_values("maneuver_accel_std")
        ax.plot(sub["maneuver_accel_std"], sub["mean"], "o-", markersize=4.4,
                linewidth=1.5, color=interval_colors[j], label=rf"$T={interval:g}$ s")
    ax.set_xscale("log"); _log_ticks(ax, sigmas)
    ax.set_xlabel(r"Maneuver intensity $\sigma_{\mathrm{man}}$ (m/s$^2$)")
    ax.set_ylabel("Attitude RMSE (deg)")
    ax.set_title("Attitude RMSE vs maneuver intensity")
    ax.legend(
        ncol=2,
        loc="upper right",
        frameon=False,
        fontsize=6.9,
    )
    motion = tables["motion"]
    if not motion.empty:
        force = (motion[motion["key"] == "specific_force_mean"]
                 .merge(tables["conditions"][["condition_id", "maneuver_accel_std", "module"]],
                        on="condition_id")
                 .query("module.str.contains('E1_maneuver')", engine="python")
                 .groupby("maneuver_accel_std")["value"].mean() / 9.80665)
        twin = ax.twinx()
        twin.plot(force.index, force.to_numpy(), "k--", linewidth=1.2)
        twin.set_ylabel(r"Mean specific force $|f^b|/g$", fontsize=8.6)
        twin.spines["top"].set_visible(False)
    _clean(ax); _panel(ax, "(f)")


    long = gap.stack().rename("gap_mm").reset_index()
    long["gap_sem_mm"] = gap_sem.stack().to_numpy()
    long["anees_excess"] = excess.stack().to_numpy()
    long["panel"] = "penalty_grid"
    if not speed_stats.empty:
        speed_long = speed_stats.copy()
        speed_long["panel"] = "speed_sweep"
        speed_long["sync_interval"] = speed_interval
        long = pd.concat([long, speed_long], ignore_index=True, sort=False)
    long["figure"] = "fig3_scenario_robustness"
    return _save(fig, outdir, "fig3_scenario_robustness"), long


def _paired_ratio_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-condition mean of the per-seed RMSE ratio to the centralized filter."""
    position = frame[(frame["group"] == "all") & (frame["block"] == "position")].copy()
    reference = (position[position["method"] == "centralized"]
                 .set_index(["condition_id", "seed"])["rmse"])
    base = pd.MultiIndex.from_frame(position[["condition_id", "seed"]]).map(reference)
    position["ratio"] = position["rmse"] / np.asarray(base, dtype=float)
    return summarise(position, "ratio",
                     ["method", "maneuver_accel_std", "sync_interval", "condition_id"],
                     drop_diverged=True)


# ---------------------------------------------------------------------------
# Figure 4: network scaling
# ---------------------------------------------------------------------------
RESOURCE_PANELS = (
    ("C_total_bytes", "Communication (MB)", 1024.0 ** 2),
    ("storage_bytes", "Covariance storage (MB)", 1024.0 ** 2),
    ("flops", "Compute (GFLOPs)", 1e9),
)


def plot_network_scaling(frame: pd.DataFrame, tables: dict[str, pd.DataFrame],
                         counts: pd.DataFrame, outdir: Path) -> tuple[list[str], pd.DataFrame]:
    ids = set(_conditions(tables, "E2_network_size")["condition_id"])
    scaling = drop_unobservable(frame[frame["condition_id"].isin(ids)])
    position = scaling[(scaling["group"] == "all") & (scaling["block"] == "position")]

    rmse = summarise(position, "rmse", ["method", "K", "condition_id"], drop_diverged=True)
    anees = summarise(position, "anees", ["method", "K", "condition_id"], drop_diverged=False)
    count_map = counts.set_index(["condition_id", "method"])[["n_diverged", "n_seeds"]]

    resources = tables["resources"].merge(
        tables["conditions"][["condition_id", "K", "module"]], on="condition_id")
    resources = resources[resources["condition_id"].isin(ids)]

    methods = [m for m in ARCH_METHODS if m in set(position["method"])]
    Ks = sorted(position["K"].dropna().unique())

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(PAPER_FULL_WIDTH_IN, 8.05),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    for method in methods:
        sub = rmse[rmse["method"] == method].sort_values("K")
        ax.plot(sub["K"], sub["mean"], marker=METHOD_MARKER[method],
                color=METHOD_COLOR[method], linewidth=1.7, markersize=5.0,
                label=METHOD_LABEL[method])
    ax.set_yscale("log")
    _readable_log_axis(ax)
    ax.set_xscale("log"); _log_ticks(ax, [int(k) for k in Ks])
    ax.set_xlabel("Number of entities $K$"); ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Accuracy")
    ax.legend(fontsize=7.0, frameon=False, ncol=2, loc="upper left")
    _clean(ax); _panel(ax, "(a)")

    # (b) Consistency.  Median with a p90 band, on a clipped log axis: the
    # decentralized baseline reaches 10^4 at K=40 and would otherwise compress
    # the 1.0-1.7 band that the paper's claim lives in.
    # Median only.  A p90 band appeared here and nowhere else, which invited the
    # question of why one panel is drawn differently from every other
    # consistency panel; the diverged-seed count carries the tail instead.
    ax = axes[0, 1]
    ceiling = _ceiling(anees["median"].to_numpy(dtype=float), 50.0)
    for method in methods:
        sub = anees[anees["method"] == method].sort_values("K")
        x = sub["K"].to_numpy(dtype=float)
        y = sub["median"].to_numpy(dtype=float)
        plot_clipped(ax, x, y, ceiling, METHOD_COLOR[method],
                     METHOD_MARKER[method], METHOD_LABEL[method])
        pairs = [tuple(count_map.loc[(cid, method)]) if (cid, method) in count_map.index
                 else (0, 0) for cid in sub["condition_id"]]
        failure_offsets = {
            "centralized": (-9, 5),
            "dual_layer": (9, 5),
            "ci_trace": (-9, 12),
            "local_only": (9, 12),
        }
        _annotate_failures(
            ax,
            x,
            np.minimum(y, ceiling),
            pairs,
            ceiling,
            offset=failure_offsets.get(method, (0, 7)),
        )
    _reference_line(ax, 1.0, "ideal = 1")
    ax.set_yscale("log"); ax.set_ylim(0.3, ceiling * 1.48)
    _readable_log_axis(ax)
    ax.set_xscale("log"); _log_ticks(ax, [int(k) for k in Ks])
    ax.set_xlabel("Number of entities $K$")
    ax.set_ylabel("Position ANEES (median)")
    ax.set_title("Consistency")
    _clean(ax); _panel(ax, "(b)")

    resource_axes = (axes[1, 0], axes[1, 1], axes[2, 0])
    for ax, (key, ylabel, scale), tag in zip(resource_axes, RESOURCE_PANELS, "cde"):
        sub = resources[resources["key"] == key]
        drawn: list[np.ndarray] = []
        for method in methods:
            series = (sub[sub["method"] == method].groupby("K")["value"].mean() / scale)
            values = series.to_numpy(dtype=float)
            # CI and DCL hold the same per-entity blocks, so their storage is
            # identical and their FLOP counts agree to within a percent.  Rather
            # than break the line style used everywhere else, a curve that lands
            # on one already drawn gets a hollow, larger marker: the reader sees
            # two symbols at one position, which is what the data says.
            coincident = any(np.allclose(values, other, rtol=0.02)
                             for other in drawn)
            if coincident:
                # Drawing a second solid line here would simply hide the first.
                # Rings around the existing markers say "same value" without
                # changing the line style used in every other panel.
                ax.plot(series.index, values, marker=METHOD_MARKER[method],
                        color=METHOD_COLOR[method], linestyle="none",
                        markersize=9.0, markerfacecolor="none",
                        markeredgewidth=1.7, label=METHOD_LABEL[method])
            else:
                ax.plot(series.index, values, marker=METHOD_MARKER[method],
                        color=METHOD_COLOR[method], linewidth=1.7, markersize=5.0,
                        label=METHOD_LABEL[method])
            drawn.append(values)
        ax.set_yscale("log")
        ax.set_xscale("log"); _log_ticks(ax, [int(k) for k in Ks])
        ax.set_xlabel("Number of entities $K$"); ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" (")[0])
        _clean(ax); _panel(ax, f"({tag})")

    # (f) How the resource advantage of the proposed filter scales.  An earlier
    # draft put a cost-consistency scatter here, but that panel had to pick one
    # accuracy metric and one cost metric out of several, and neither choice was
    # defensible on its own.  Showing all three cost axes at once removes the
    # arbitrary pick and makes the point the paper actually needs: the saving
    # relative to the centralized filter widens with network size.
    ax = axes[2, 1]
    reference = resources[resources["method"] == "centralized"]
    styles = {"C_total_bytes": ("Communication", "o", "#0072B2"),
              "storage_bytes": ("Covariance storage", "s", "#009E73"),
              "flops": ("Compute", "^", "#CC79A7")}
    for key, (label, marker, color) in styles.items():
        numerator = (resources[(resources["method"] == "dual_layer")
                               & (resources["key"] == key)].groupby("K")["value"].mean())
        denominator = (reference[reference["key"] == key].groupby("K")["value"].mean())
        ratio = (numerator / denominator).dropna()
        ax.plot(ratio.index, ratio.to_numpy(), marker=marker, color=color,
                linewidth=1.7, markersize=5.0, label=label)
    ax.set_yscale("log")
    _readable_log_axis(ax)
    ax.set_xscale("log"); _log_ticks(ax, [int(k) for k in Ks])
    ax.set_xlabel("Number of entities $K$")
    ax.set_ylabel("Dual-layer cost / Centralized")
    ax.set_title("Resource saving vs network size")
    ax.legend(fontsize=7.0, frameon=False, loc="upper right")
    _clean(ax); _panel(ax, "(f)")


    source = pd.concat([rmse, anees], ignore_index=True)
    source["figure"] = "fig4_network_scaling"
    return _save(fig, outdir, "fig4_network_scaling"), source


# ---------------------------------------------------------------------------
# Figure 5: composition and topology
# ---------------------------------------------------------------------------
def _categorical(frame: pd.DataFrame, ids: set[str], label_col: str,
                 counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired RMSE ratio and ANEES for one categorical sweep.

    The ratio is summarised over the seeds that converged, per the stated
    policy.  Where a method diverged on *every* seed the filtered mean does not
    exist, so the all-seed value is carried instead and flagged: dropping the
    point entirely would leave a gap that reads as missing data rather than as
    the total failure it actually is.
    """
    sub = drop_unobservable(frame[frame["condition_id"].isin(ids)])
    position = sub[(sub["group"] == "all") & (sub["block"] == "position")].copy()
    reference = (position[position["method"] == "centralized"]
                 .set_index(["condition_id", "seed"])["rmse"])
    base = pd.MultiIndex.from_frame(position[["condition_id", "seed"]]).map(reference)
    position["ratio"] = position["rmse"] / np.asarray(base, dtype=float)

    keys = ["condition_id", label_col, "method"]
    survived = summarise(position, "ratio", keys, drop_diverged=True)
    everything = summarise(position, "ratio", keys, drop_diverged=False)
    ratios = everything.merge(survived[keys + ["mean", "n"]], on=keys, how="left",
                              suffixes=("_all", "_survived"))
    ratios["all_diverged"] = ratios["n_survived"].isna()
    ratios["mean"] = ratios["mean_survived"].fillna(ratios["mean_all"])
    ratios["n"] = ratios["n_survived"].fillna(ratios["n_all"])

    anees = summarise(position, "anees", keys, drop_diverged=False)
    return ratios, anees


def plot_composition_topology(frame: pd.DataFrame, tables: dict[str, pd.DataFrame],
                              counts: pd.DataFrame, outdir: Path,
                              nominal: str) -> tuple[list[str], pd.DataFrame]:
    conditions = tables["conditions"]
    composition_ids = set(_conditions(tables, "E3_class_composition")["condition_id"])
    topology_ids = set(_conditions(tables, "E4_topology")["condition_id"])

    order = conditions[conditions["condition_id"].isin(composition_ids)].copy()
    order = order.sort_values(["num_C", "num_A"], ascending=[True, False])
    composition_order = list(dict.fromkeys(order["class_sizes"].astype(str)))

    topology_meta = conditions[conditions["condition_id"].isin(topology_ids)].copy()
    topology_meta = topology_meta[topology_meta["degree"] >= MIN_ADMISSIBLE_DEGREE]
    topology_meta = topology_meta.sort_values("degree", ascending=False)
    topology_order = list(dict.fromkeys(topology_meta["topology_label"].astype(str)))

    comp_ratio, comp_anees = _categorical(frame, composition_ids, "class_sizes", counts)
    top_ratio, top_anees = _categorical(frame, topology_ids, "topology_label", counts)
    count_map = counts.set_index(["condition_id", "method"])[["n_diverged", "n_seeds"]]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(PAPER_FULL_WIDTH_IN, 5.55),
        constrained_layout=True,
    )

    def draw(ax: plt.Axes, table: pd.DataFrame, label_col: str,
             categories: Sequence[str], value: str, hard: float,
             ylabel: str, title: str, tag: str, reference: float) -> None:
        mapping = {c: i for i, c in enumerate(categories)}
        ceiling = _ceiling(table[value].to_numpy(dtype=float), hard)
        for method in ARCH_METHODS:
            sub = table[table["method"] == method].copy()
            sub["x"] = sub[label_col].astype(str).map(mapping)
            sub = sub.dropna(subset=["x"]).sort_values("x")
            if sub.empty:
                continue
            x = sub["x"].to_numpy(dtype=float)
            y = sub[value].to_numpy(dtype=float)
            plot_clipped(ax, x, y, ceiling, METHOD_COLOR[method],
                         METHOD_MARKER[method], METHOD_LABEL[method])
            pairs = [tuple(count_map.loc[(cid, method)])
                     if (cid, method) in count_map.index else (0, 0)
                     for cid in sub["condition_id"]]
            failure_offsets = {
                "centralized": (-10, 5),
                "dual_layer": (10, 5),
                "ci_trace": (-10, 13),
                "local_only": (10, 13),
            }
            _annotate_failures(
                ax,
                x,
                np.minimum(y, ceiling),
                pairs,
                ceiling,
                offset=failure_offsets.get(method, (0, 7)),
            )
        _reference_line(ax, reference,
                        "ideal = 1" if value != "mean_ratio" else "Centralized = 1")
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories, rotation=18 if len(categories) > 4 else 0,
                           ha="right" if len(categories) > 4 else "center")
        ax.set_yscale("log")
        ax.set_ylim(min(0.2, reference * 0.4), ceiling * 1.65)
        _readable_log_axis(ax)
        ax.set_ylabel(ylabel); ax.set_title(title)
        _clean(ax); _panel(ax, tag)

    draw(axes[0, 0], comp_ratio, "class_sizes", composition_order, "mean", 30.0,
         "Position RMSE / Centralized", "Class composition (A/B/C)", "(a)", 1.0)
    draw(axes[0, 1], comp_anees, "class_sizes", composition_order, "median", 50.0,
         "Position ANEES (median)", "Class composition (A/B/C)", "(b)", 1.0)
    draw(axes[1, 0], top_ratio, "topology_label", topology_order, "mean", 30.0,
         "Position RMSE / Centralized", "Topology density", "(c)", 1.0)
    draw(axes[1, 1], top_anees, "topology_label", topology_order, "median", 50.0,
         "Position ANEES (median)", "Topology density", "(d)", 1.0)
    axes[0, 0].legend(fontsize=7.0, frameon=False, ncol=2, loc="upper left")


    comp_ratio["panel"], comp_anees["panel"] = "composition_ratio", "composition_anees"
    top_ratio["panel"], top_anees["panel"] = "topology_ratio", "topology_anees"
    source = pd.concat([comp_ratio, comp_anees, top_ratio, top_anees],
                       ignore_index=True, sort=False)
    source["figure"] = "fig5_composition_topology"
    return _save(fig, outdir, "fig5_composition_topology"), source


# ---------------------------------------------------------------------------
# Text-only quantities the paper quotes
# ---------------------------------------------------------------------------
def eta_scaling(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fit the T exponent of each structural residual metric.

    Only eta_S reaches the paper (III-B predicts O(T^2) for the innovation
    variance error); the others are exported for the record.
    """
    diagnostics = tables["diagnostics"]
    if diagnostics.empty:
        return pd.DataFrame()
    merged = diagnostics.merge(
        tables["conditions"][["condition_id", "module", "maneuver_accel_std",
                              "sync_interval"]], on="condition_id")
    merged = merged[merged["module"].str.contains("E1_maneuver", na=False)
                    & (merged["scope"] == "all")]
    rows = []
    for key, sample_point in (("eta_S_mean", "at_innovation"),
                              ("eta_dyn_mean", "pre_uwb"),
                              ("eta_struct_mean", "pre_uwb"),
                              ("eta_abs_mean", "pre_uwb")):
        sub = merged[(merged["key"] == key) & (merged["sample_point"] == sample_point)]
        if sub.empty:
            continue
        for sigma, group in sub.groupby("maneuver_accel_std"):
            cell = group.groupby("sync_interval")["value"].mean()
            positive = cell[cell > 0]
            if len(positive) < 3:
                continue
            slope, intercept = np.polyfit(np.log(positive.index.to_numpy(dtype=float)),
                                          np.log(positive.to_numpy()), 1)
            rows.append({"key": key, "sample_point": sample_point,
                         "maneuver_accel_std": float(sigma),
                         "T_exponent": float(slope),
                         "value_at_T_0.1": float(np.exp(intercept) * 0.1 ** slope)})
    return pd.DataFrame(rows)


def nominal_summary(frame: pd.DataFrame, cid: str) -> pd.DataFrame:
    """Per-class, per-block table backing Table I."""
    sub = frame[frame["condition_id"] == cid]
    rmse = summarise(sub, "rmse", ["method", "group", "block"], drop_diverged=True)
    anees = summarise(sub, "anees", ["method", "group", "block"], drop_diverged=False)
    merged = rmse.merge(anees, on=["method", "group", "block"], suffixes=("_rmse", "_anees"))
    return merged[["method", "group", "block", "mean_rmse", "sem_rmse",
                   "mean_anees", "median_anees", "n_rmse"]]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables",
        default="analysis_tables"
                r"\server_return\analysis_tables",
        help="directory holding conditions.csv, metrics.csv.gz, ...")
    parser.add_argument("--output", default=None,
                        help="default: <tables>/paper_figures")
    parser.add_argument("--paired-rmse-ratio", type=float, default=20.0)
    parser.add_argument("--anees-bound", type=float, default=50.0)
    args = parser.parse_args(argv)

    root = Path(args.tables).expanduser().resolve()
    outdir = Path(args.output).expanduser().resolve() if args.output else root / "paper_figures"
    outdir.mkdir(parents=True, exist_ok=True)
    apply_style()
    policy = FailurePolicy(paired_rmse_ratio=args.paired_rmse_ratio,
                           anees_bound=args.anees_bound)

    print(f"Tables : {root}")
    print(f"Output : {outdir}")

    tables = load_tables(root)
    frame = classify_failures(build_metric_frame(tables), policy)
    counts = failure_counts(frame)
    nominal = nominal_id(tables)
    speed_ids, speed_interval = speed_sweep(tables)

    excluded = tables["conditions"]
    excluded = excluded[excluded["degree"] < MIN_ADMISSIBLE_DEGREE]["condition_id"].tolist()
    print(f"Nominal condition : {nominal}")
    print(f"Excluded (degree < {MIN_ADMISSIBLE_DEGREE}) : {excluded or 'none'}")
    print(f"Methods           : {sorted(frame['method'].unique())}")
    if speed_ids:
        speeds = sorted(tables["conditions"].loc[
            tables["conditions"]["condition_id"].isin(speed_ids),
            "target_speed"].dropna().unique())
        print(f"Speed sweep       : {sorted(speed_ids)} at UWB interval T = {speed_interval:g} s, "
              f"v = {[float(v) for v in speeds]}")
    else:
        print("Speed sweep       : none found; Figure 3(b) falls back to the "
              "maneuver axis of the structural penalty")

    generated: list[str] = []
    sources: list[pd.DataFrame] = []
    for files, source in (
        plot_nominal(frame, counts, outdir, nominal),
        plot_maneuver_sync(frame, tables, counts, outdir),
        plot_network_scaling(frame, tables, counts, outdir),
        plot_composition_topology(frame, tables, counts, outdir, nominal),
    ):
        generated.extend(files)
        sources.append(source)

    _write(pd.concat(sources, ignore_index=True, sort=False),
           outdir / "figure_source_values.csv")
    _write(counts.sort_values(["condition_id", "method"]),
           outdir / "failure_counts.csv")
    _write(nominal_summary(frame, nominal), outdir / "table1_nominal_by_class.csv")
    eta = eta_scaling(tables)
    if not eta.empty:
        _write(eta, outdir / "eta_T_scaling.csv")

    diverged = frame[(frame["group"] == "all") & (frame["block"] == "position")
                     & frame["diverged"]]
    manifest = {
        "tables_root": str(root),
        "output_directory": str(outdir),
        "nominal_condition": nominal,
        "speed_sweep_conditions": sorted(speed_ids),
        "speed_sweep_sync_interval": None if not speed_ids else float(speed_interval),
        "excluded_conditions": excluded,
        "min_admissible_degree": MIN_ADMISSIBLE_DEGREE,
        "failure_policy": asdict(policy),
        "diverged_runs": int(len(diverged)),
        "generated_files": generated,
        "notes": [
            "Divergent seeds are excluded from RMSE statistics only; ANEES is "
            "always reported over every seed.",
            "Consistency panels use a clipped log axis so that a baseline "
            "blow-up cannot compress the band around 1.",
            "Circular-2 is excluded: class-C position is rank deficient with "
            "two ranges, so every architecture including the reference diverges.",
            "In Figure 3, T denotes the time between successive complete UWB "
            "ranging rounds; it is not a clock- or communication-synchronization "
            "period.",
            "All paper figures are exported as vector PDF files with embedded "
            "TrueType fonts.",
        ],
    }
    (outdir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Diverged runs     : {len(diverged)}")
    if len(diverged):
        table = (diverged.groupby(["condition_id", "method"]).size()
                 .rename("n").reset_index().sort_values("n", ascending=False))
        for _, row in table.iterrows():
            print(f"   {row['condition_id']}  {row['method']:<12} {row['n']}")
    print("Saved:")
    for path in generated:
        print("  ", path)
    for name in ("figure_source_values.csv", "failure_counts.csv",
                 "table1_nominal_by_class.csv", "eta_T_scaling.csv",
                 "figure_manifest.json"):
        if (outdir / name).exists():
            print("  ", outdir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
