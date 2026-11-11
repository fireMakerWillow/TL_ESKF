"""Truth-only validation of the five maneuver-intensity grid points.

Runs the random-acceleration trajectory without filters and reports the motion
quantities required by the experiment design after burn-in.  This is cheap
compared with the full Monte Carlo campaign and should be run before it.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from config import default_config, validate_config
from l2_truth import TruthSimulator

SIGMA_A_VALUES = (0.5, 1.0, 2.0, 4.0, 8.0)


def parse_seeds(text: str):
    if ":" in text:
        a, b = text.split(":", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def validate_one(sigma_a: float, seeds, duration_s: float, burn_in_s: float):
    accel, force, speed = [], [], []
    outside = total_positions = 0

    for seed in seeds:
        cfg = default_config()
        cfg["traj_type"] = "random_accel"
        cfg["speed"] = 5.0
        cfg["target_speed"] = 5.0
        cfg["maneuver_accel_std"] = float(sigma_a)
        cfg["sim_duration"] = float(duration_s)
        cfg["separation_check"] = False
        validate_config(cfg)

        truth = TruthSimulator(cfg)
        truth.reset(int(seed))
        steps = int(round(duration_s / cfg["dt_imu"]))
        gravity = np.asarray(cfg["gravity"], dtype=float)
        for step in range(steps):
            record = truth.step()
            time_s = (step + 1) * float(cfg["dt_imu"])
            if time_s <= burn_in_s:
                continue
            f_b = np.asarray(record.f_b, dtype=float)
            force.append(np.linalg.norm(f_b, axis=1))
            a_n = np.stack([
                record.states[i].R @ f_b[i] + gravity
                for i in range(cfg["num_entities"])
            ])
            accel.append(np.linalg.norm(a_n, axis=1))
            speed.append(np.asarray([
                np.linalg.norm(state.v) for state in record.states
            ], dtype=float))
            xy = np.asarray([state.p[:2] for state in record.states], dtype=float)
            field = float(cfg["field_size"])
            outside += int(np.count_nonzero(
                (xy[:, 0] < 0.0) | (xy[:, 0] > field)
                | (xy[:, 1] < 0.0) | (xy[:, 1] > field)))
            total_positions += int(xy.shape[0])

    a = np.concatenate(accel) if accel else np.empty(0)
    f = np.concatenate(force) if force else np.empty(0)
    v = np.concatenate(speed) if speed else np.empty(0)
    return {
        "maneuver_accel_std": float(sigma_a),
        "accel_norm_mean": float(np.mean(a)),
        "accel_norm_p95": float(np.quantile(a, 0.95)),
        "specific_force_mean": float(np.mean(f)),
        "specific_force_over_g": float(np.mean(f) / 9.81),
        "speed_mean": float(np.mean(v)),
        "xy_out_of_bounds_fraction": (
            float(outside) / total_positions if total_positions else None),
        "samples": int(a.size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1010")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--burn-in", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("maneuver_validation.json"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--strict", action="store_true",
                        help="fail if any XY sample is outside the field or p95 reaches the cap")
    args = parser.parse_args()
    if args.duration <= 0 or not (0 <= args.burn_in < args.duration):
        parser.error("require duration > 0 and 0 <= burn-in < duration")

    seeds = parse_seeds(args.seeds)
    workers = max(1, min(int(args.workers), len(SIGMA_A_VALUES)))
    if workers == 1:
        rows = [validate_one(sigma, seeds, args.duration, args.burn_in)
                for sigma in SIGMA_A_VALUES]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(validate_one, sigma, seeds,
                                   args.duration, args.burn_in)
                       for sigma in SIGMA_A_VALUES]
            rows = [future.result() for future in futures]
    payload = {
        "seeds": seeds,
        "duration_s": args.duration,
        "burn_in_s": args.burn_in,
        "max_accel": 30.0,
        "xy_boundary_margin": 30.0,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("sigma_a  mean|a|  p95|a|  mean|f_b|  |f|/g  mean|v|  out-of-bounds")
    for row in rows:
        print(f"{row['maneuver_accel_std']:>7.1f}  "
              f"{row['accel_norm_mean']:>7.3f}  "
              f"{row['accel_norm_p95']:>7.3f}  "
              f"{row['specific_force_mean']:>9.3f}  "
              f"{row['specific_force_over_g']:>6.3f}  "
              f"{row['speed_mean']:>7.3f}  "
              f"{row['xy_out_of_bounds_fraction']:.6f}")
    print(f"wrote {args.output}")

    if args.strict:
        failures = []
        for row in rows:
            if row["xy_out_of_bounds_fraction"] != 0.0:
                failures.append(
                    f"sigma={row['maneuver_accel_std']}: out-of-bounds fraction "
                    f"{row['xy_out_of_bounds_fraction']}")
            if row["accel_norm_p95"] >= 0.95 * payload["max_accel"]:
                failures.append(
                    f"sigma={row['maneuver_accel_std']}: p95 acceleration too close to cap")
        if failures:
            raise SystemExit("validation failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
