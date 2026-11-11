"""M22 first-stage applicability-domain prescan."""

from config import apply_entity_count, default_config, validate_config
from l4_failure_boundary import run


def main() -> None:
    print("[M22] Starting prescan...", flush=True)

    cfg = default_config()
    apply_entity_count(cfg, 10, (2, 6, 2))

    # ------------------------------------------------------------------
    # M22 motion model: bounded 3-D random motion
    # ------------------------------------------------------------------
    # Do not inherit default_config()'s circle_formation trajectory.
    cfg["traj_type"] = "random_accel"

    # Random maneuver intensity.
    cfg["max_accel"] = 4.0
    cfg["max_turn_rate"] = 0.8
    cfg["random_hold_time"] = 0.50
    cfg["random_vertical_scale"] = 0.20

    # Sustained full 3-D target speed.  l4_failure_boundary overwrites
    # target_speed with each current sweep_v value (2/5/10 m/s).
    cfg["target_speed"] = None
    cfg["target_speed_gain"] = 0.80

    # Smooth x-y boundary recovery.
    cfg["xy_boundary_margin"] = 20.0
    cfg["xy_boundary_k"] = 0.20
    cfg["xy_boundary_damping"] = 0.80
    cfg["random_velocity_damping"] = 0.03

    # Diverse per-entity height layers.  z_reference is only the centre
    # of the band; entities must NOT all converge to z=25 m.
    cfg["z_reference"] = 25.0
    cfg["z_reference_spread"] = 30.0
    cfg["z_reference_jitter"] = 1.0
    cfg["z_initial_spread"] = 2.0
    cfg["z_min"] = 5.0
    cfg["z_max"] = 45.0
    cfg["z_restore_k"] = 0.035
    cfg["z_velocity_damping"] = 0.25
    cfg["z_boundary_margin"] = 3.0
    cfg["z_boundary_k"] = 0.60

    # fixed_angle is meaningful for circular motion, not random motion.
    # For random_accel the M22 grid is therefore evaluated over the same
    # simulated duration at every (v, T) point.
    cfg["m22_horizon_mode"] = "fixed_duration"

    cfg["sweep_v_values"] = [2.0, 5.0, 10.0]
    cfg["sweep_T_values"] = [0.01, 0.05, 0.10]

    cfg["mc_runs"] = 3

    # v=2 m/s 时至少约 54 秒才能通过 T20。
    cfg["sim_duration"] = 60.0

    cfg["seed_base"] = 1000
    cfg["output_dir"] = "results_m22_prescan"
    cfg["save_plots"] = False

    cfg["run_preflight"] = True
    cfg["separation_check"] = True

    validate_config(cfg)

    print(
        f"[M22] Motion: {cfg['traj_type']} | "
        f"horizon={cfg['m22_horizon_mode']} | "
        f"duration={cfg['sim_duration']:.1f}s | "
        f"target-speed gain={cfg['target_speed_gain']:.2f}",
        flush=True,
    )
    print(
        f"[M22] Grid: "
        f"{len(cfg['sweep_v_values'])} speeds × "
        f"{len(cfg['sweep_T_values'])} intervals × "
        f"{cfg['mc_runs']} runs",
        flush=True,
    )

    run(cfg, verbose=True)

    print("[M22] Prescan completed.", flush=True)


if __name__ == "__main__":
    main()
