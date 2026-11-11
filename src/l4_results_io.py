"""M23 ResultsIO (L4 application/experiment).

Persist JSON/CSV results, configuration snapshots, and comparison/failure-
boundary plots.  Plot labels are kept in English for consistent publication
output.  Plotting degrades gracefully when matplotlib is unavailable.
"""

import csv
import json
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


_METHOD_DISPLAY = {"local_only": "DCL"}


def _display_method(name):
    return _METHOD_DISPLAY.get(name, name)


def _ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def save_json(obj, path):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_default)


def save_csv_rows(rows, path):
    """Save a list of flat-ish dictionaries; containers are JSON-encoded."""
    rows = list(rows)
    _ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        with open(path, "w", encoding="utf-8", newline=""):
            return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    def encode(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, default=_default)
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: encode(row.get(key, "")) for key in fields})


def save_config_snapshot(config, out_dir, fname="config_snapshot.json"):
    _ensure_dir(out_dir)
    snapshot = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in config.items()
    }
    save_json(snapshot, os.path.join(out_dir, fname))


def plot_comparison(results, config, out_dir, architecture_methods=None,
                    ablation_methods=None):
    """Save M21 figures and return a list of paths.

    Architectures and sensor ablations are deliberately kept in separate
    figures because the latter are diagnostic references, not entries in the
    covariance-architecture ranking.  Each quantity gets its own figure so the
    axes remain readable in publication output.
    """
    if not (_HAVE_MPL and config["save_plots"]):
        return []
    _ensure_dir(out_dir)
    architecture = list(architecture_methods or [
        n for n, row in results.items() if row.get("role") == "architecture"
    ])
    ablations = list(ablation_methods or [
        n for n, row in results.items() if row.get("role") == "sensor_ablation"
    ])
    paths = []

    def bar(names, values, ylabel, title, filename, ideal=None, band=None):
        if not names:
            return
        fig, axis = plt.subplots(figsize=(8.2, 4.8))
        axis.bar([_display_method(name) for name in names], values)
        if ideal is not None:
            axis.axhline(ideal, linestyle="--", linewidth=1, label="Ideal")
        if band is not None and np.all(np.isfinite(band)):
            axis.axhspan(band[0], band[1], alpha=0.15,
                        label=f"{int(config.get('nees_bootstrap_confidence', 0.95) * 100)}% bootstrap CI")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        if ideal is not None or band is not None:
            axis.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{filename}.{config['plot_format']}")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    bar(architecture, [results[n]["pos_rmse"] for n in architecture],
        "Position RMSE (m)", "Architecture accuracy",
        "comparison_architecture_rmse")
    bar(architecture, [results[n]["rmte"] for n in architecture],
        "Position RMTE (m)", "Architecture reported uncertainty",
        "comparison_architecture_rmte")
    if architecture:
        first = results[architecture[0]]
        bar(
            architecture,
            [results[n]["time_mean_nnees"] for n in architecture],
            "Time-mean normalized position NEES",
            "Architecture consistency across seeds",
            "comparison_architecture_time_mean_nnees",
            ideal=1.0,
            band=first["time_mean_nnees_bootstrap_ci"],
        )
    bar(architecture, [results[n]["comm_estimation_kb"] for n in architecture],
        "Estimation communication (KB)", "Architecture communication",
        "comparison_architecture_estimation_comm")
    bar(architecture, [results[n]["runtime_total_s"] for n in architecture],
        "Estimator runtime per trial (s)", "Measured architecture runtime",
        "comparison_architecture_runtime")
    bar(architecture, [results[n]["memory_array_kb"] for n in architecture],
        "Measured covariance memory (KB)", "Architecture covariance memory",
        "comparison_architecture_memory")

    for metric, ylabel in (
        ("pos_rmse", "Position RMSE (m)"),
        ("rmte", "Position RMTE (m)"),
        ("time_mean_nnees", "Time-mean normalized position NEES"),
    ):
        for label in ("A", "B", "C"):
            names = architecture
            values = [results[n]["class_metrics"][label][metric] for n in names]
            bar(names, values, ylabel, f"Class {label}: {metric}",
                f"comparison_class_{label}_{metric}")

    bar(ablations, [results[n]["pos_rmse"] for n in ablations],
        "Position RMSE (m)", "Sensor ablation accuracy",
        "comparison_sensor_ablation_rmse")
    bar(ablations, [results[n]["rmte"] for n in ablations],
        "Position RMTE (m)", "Sensor ablation reported uncertainty",
        "comparison_sensor_ablation_rmte")
    if ablations:
        first = results[ablations[0]]
        bar(
            ablations,
            [results[n]["time_mean_nnees"] for n in ablations],
            "Time-mean normalized position NEES",
            "Sensor ablation consistency across seeds",
            "comparison_sensor_ablation_time_mean_nnees",
            ideal=1.0,
            band=first["time_mean_nnees_bootstrap_ci"],
        )
    return paths


def plot_failure_boundary(sweep_axis, values, agg, config, out_dir):
    if not (_HAVE_MPL and config["save_plots"]):
        return None
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.7))

    axes[0].plot(values, agg["D3"], "o-", label="D3 overall error")
    axes[0].plot(values, agg["D4"], "s-", label="D4 non-position ratio")
    axes[0].plot(values, agg["D1_rel"], "^-", label="D1 position-layer error")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Structural approximation")
    axes[0].legend(fontsize=8)

    axes[1].plot(values, agg["nees_gap"], "o-", label="Dual - centralized")
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("ANEES gap")
    axes[1].set_title("Consistency degradation")
    axes[1].legend(fontsize=8)

    axes[2].plot(values, agg["D5_kappa"], "o-", label="Measured kappa")
    theory = np.asarray(agg.get("D5_theory", []), dtype=float)
    if theory.size and np.any(np.isfinite(theory)):
        axes[2].plot(values, theory, "x--", label="0.5 |a| T^2")
    axes[2].set_ylabel("Attitude coupling strength")
    axes[2].set_title("D5 measured vs theory")
    axes[2].legend(fontsize=8)

    rho = np.asarray(agg.get("D6_rho", []), dtype=float)
    if rho.size and np.any(np.isfinite(rho)):
        axes[3].plot(values, rho, "o-", label="D6 lever-arm margin")
    axes[3].axhline(1.0, linestyle="--", linewidth=1, label="rho = 1")
    axes[3].set_ylabel("Lever-arm margin")
    axes[3].set_title("D6 lever-arm criterion")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.set_xlabel(sweep_axis)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(
        out_dir, f"failure_boundary_{sweep_axis}.{config['plot_format']}"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)

    x = np.asarray(agg.get("kappa_scatter", []), dtype=float)
    y = np.asarray(agg.get("D1_scatter", []), dtype=float)
    if x.size and y.size:
        fig2, axis = plt.subplots(figsize=(5.7, 4.7))
        axis.scatter(x, y, s=18, alpha=0.55)
        if x.size > 2 and np.ptp(x) > 0:
            coefficients = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            axis.plot(
                xs, np.polyval(coefficients, xs), linewidth=1,
                label=f"Linear slope={coefficients[0]:.2e}",
            )
            axis.legend(fontsize=8)
        axis.set_xlabel("D5 kappa")
        axis.set_ylabel("D1 position-layer deviation")
        axis.set_title("Coupling strength vs position-layer error")
        axis.grid(alpha=0.3)
        fig2.tight_layout()
        scatter_path = os.path.join(
            out_dir, f"kappa_scatter_{sweep_axis}.{config['plot_format']}"
        )
        fig2.savefig(scatter_path, dpi=140)
        plt.close(fig2)
    return path


if __name__ == "__main__":
    from config import default_config

    cfg = default_config()
    cfg["output_dir"] = "results_smoke"
    classes = {
        label: {"pos_rmse": 0.06, "rmte": 0.07, "anees": 4.0,
                "time_mean_nnees": 1.0}
        for label in ("A", "B", "C")
    }
    sample = {
        "centralized": {
            "role": "architecture", "pos_rmse": 0.06, "rmte": 0.07,
            "anees": 12.2, "anees_bounds": (10, 14), "dof": 12,
            "time_mean_nnees": 1.02,
            "time_mean_nnees_bootstrap_ci": (0.9, 1.1),
            "comm_estimation_kb": 51, "runtime_total_s": 1.0,
            "memory_array_kb": 120, "class_metrics": classes,
        },
        "dual_layer": {
            "role": "architecture", "pos_rmse": 0.063, "rmte": 0.075,
            "anees": 12.5, "anees_bounds": (10, 14), "dof": 12,
            "time_mean_nnees": 1.04,
            "time_mean_nnees_bootstrap_ci": (0.9, 1.1),
            "comm_estimation_kb": 21, "runtime_total_s": 0.3,
            "memory_array_kb": 45, "class_metrics": classes,
        },
    }
    p1 = plot_comparison(sample, cfg, cfg["output_dir"],
                         architecture_methods=list(sample), ablation_methods=[])
    agg = {
        "D3": [0.01, 0.05, 0.2],
        "D4": [0.1, 0.3, 0.9],
        "D1_rel": [0.01, 0.1, 0.4],
        "nees_gap": [0.1, 0.8, 3.0],
        "D5_kappa": [0.1, 0.4, 1.6],
        "D5_theory": [0.12, 0.45, 1.7],
        "D6_rho": [0.1, 0.5, 1.4],
        "kappa_scatter": [0.1, 0.4, 1.6],
        "D1_scatter": [0.01, 0.1, 0.4],
    }
    p2 = plot_failure_boundary(
        "imu_per_uwb", [1, 5, 10], agg, cfg, cfg["output_dir"]
    )
    save_csv_rows([{"method": "centralized", **sample["centralized"]}],
                  os.path.join(cfg["output_dir"], "sample.csv"))
    print("[M23 ResultsIO] OK ->", p1, p2)
