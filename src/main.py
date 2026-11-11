"""M24 command-line entry point.

Load and validate the unified configuration, then dispatch either the M21
comparison experiment or the M22 failure-boundary experiment.

Usage:
    python main.py comparison
    python main.py failure
"""

import argparse

from config import default_config, validate_config
import l4_comparison
import l4_failure_boundary


def build_config(args):
    cfg = default_config()
    if args.runs is not None:
        cfg["mc_runs"] = args.runs
    if args.duration is not None:
        cfg["sim_duration"] = args.duration
    if args.entities is not None:
        from config import apply_entity_count
        apply_entity_count(cfg, args.entities)
        # An explicit --entities request means one comparison scenario rather
        # than the default K=8/K=10 suite.
        cfg["comparison_entity_counts"] = [int(args.entities)]
        cfg["comparison_class_sizes"] = {
            int(args.entities): tuple(cfg["class_sizes"]),
        }
        cfg["comparison_run_extension"] = False
    if args.output is not None:
        cfg["output_dir"] = args.output
    validate_config(cfg)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dual-layer covariance cooperative localization"
    )
    parser.add_argument(
        "experiment", choices=["comparison", "failure"],
        help="experiment to run",
    )
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--entities", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    cfg = build_config(args)
    if args.experiment == "comparison":
        return l4_comparison.run(cfg)
    return l4_failure_boundary.run(cfg)


if __name__ == "__main__":
    main()
