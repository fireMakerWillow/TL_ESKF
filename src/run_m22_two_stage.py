"""Run M22 fixed-duration and fixed-angular-travel scans automatically.

Examples (PowerShell / VS Code terminal):

    python -u .\run_m22_two_stage.py
    python -u .\run_m22_two_stage.py --runs 5 --duration 60 --angle-deg 90
"""

from __future__ import annotations

import argparse
import math

from config import apply_entity_count, default_config, validate_config
from l4_failure_boundary import run_two_stage


def _float_list(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one number")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="M22 two-stage fixed-duration + fixed-angle sweep")
    parser.add_argument("--runs", type=int, default=3,
                        help="Monte-Carlo trials per grid point (default: 3)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="fixed-duration stage length in seconds")
    parser.add_argument("--angle-deg", type=float, default=90.0,
                        help="fixed-angle stage target travel in degrees")
    parser.add_argument("--speeds", type=_float_list, default=[2.0, 5.0, 10.0],
                        help="comma-separated speeds, e.g. 2,5,10")
    parser.add_argument("--intervals", type=_float_list,
                        default=[0.01, 0.05, 0.10],
                        help="comma-separated T values, e.g. 0.01,0.05,0.1")
    parser.add_argument("--entities", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--burn-in", type=float, default=0.5)
    parser.add_argument("--output", default="results_m22_two_stage")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    cfg = default_config()
    if args.entities == 8:
        apply_entity_count(cfg, 8, (2, 4, 2))
    elif args.entities == 10:
        apply_entity_count(cfg, 10, (2, 6, 2))
    else:
        apply_entity_count(cfg, args.entities)

    cfg["sweep_v_values"] = list(args.speeds)
    cfg["sweep_T_values"] = list(args.intervals)
    cfg["mc_runs"] = int(args.runs)
    cfg["seed_base"] = int(args.seed_base)
    cfg["sim_duration"] = float(args.duration)
    cfg["m22_fixed_duration_s"] = float(args.duration)
    cfg["m22_target_angle_rad"] = math.radians(float(args.angle_deg))
    cfg["m22_burn_in_frac"] = float(args.burn_in)
    cfg["output_dir"] = str(args.output)
    cfg["save_plots"] = not bool(args.no_plots)
    cfg["run_preflight"] = True
    cfg["separation_check"] = True
    validate_config(cfg)

    print("[M22] Automatic two-stage scan", flush=True)
    print(
        f"[M22] Grid: {len(cfg['sweep_v_values'])} speeds x "
        f"{len(cfg['sweep_T_values'])} intervals x {cfg['mc_runs']} runs",
        flush=True)
    print(
        f"[M22] Stage 1 fixed duration: {args.duration:g} s; "
        f"Stage 2 fixed angle: {args.angle_deg:g} deg",
        flush=True)
    run_two_stage(
        cfg, fixed_duration_s=args.duration,
        target_angle_rad=cfg["m22_target_angle_rad"], verbose=True)
    print(f"[M22] Completed -> {cfg['output_dir']}", flush=True)


if __name__ == "__main__":
    main()
