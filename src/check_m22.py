"""Focused acceptance gate for the revised M22 two-horizon experiment."""

import math
import os
import shutil
import tempfile

from config import apply_entity_count, default_config, validate_config
from l4_failure_boundary import _horizon_config, run_two_stage


def main():
    cfg = default_config()
    apply_entity_count(cfg, 3, (1, 1, 1))
    cfg["sweep_v_values"] = [2.0, 4.0]
    cfg["sweep_T_values"] = [0.05]
    cfg["mc_runs"] = 1
    cfg["sim_duration"] = 0.1
    cfg["m22_fixed_duration_s"] = 0.1
    cfg["m22_target_angle_rad"] = 0.01
    cfg["m22_burn_in_frac"] = 0.0
    cfg["run_preflight"] = False
    cfg["separation_check"] = False
    cfg["save_plots"] = False
    validate_config(cfg)

    _, d2 = _horizon_config(cfg, 2.0, 0.05, "fixed_duration", 0.1, None)
    _, d4 = _horizon_config(cfg, 4.0, 0.05, "fixed_duration", 0.1, None)
    assert d2["actual_duration_s"] == d4["actual_duration_s"]
    assert d4["actual_angle_rad"] > d2["actual_angle_rad"]

    _, a2 = _horizon_config(cfg, 2.0, 0.05, "fixed_angle", None, 0.1)
    _, a4 = _horizon_config(cfg, 4.0, 0.05, "fixed_angle", None, 0.1)
    assert a2["actual_angle_rad"] >= 0.1 and a4["actual_angle_rad"] >= 0.1
    assert a2["actual_duration_s"] > a4["actual_duration_s"]

    out = tempfile.mkdtemp(prefix="m22_gate_")
    try:
        cfg["output_dir"] = out
        payload = run_two_stage(cfg, 0.1, 0.01, verbose=False)
        assert len(payload["fixed_duration"]["rows"]) == 2
        assert len(payload["fixed_angle"]["rows"]) == 2
        required = {
            "rmse_rel_gap", "rmse_C_rel_gap", "dual_layer_anees",
            "nees_band_excess", "eta_dyn_per_cycle", "eta_S_mean",
            "pos_layer_dev_rel_final", "zone",
        }
        for stage in ("fixed_duration", "fixed_angle"):
            for row in payload[stage]["rows"]:
                assert required.issubset(row)
        files = {
            "applicability_two_stage.json",
            os.path.join("fixed_duration", "applicability_fixed_duration.csv"),
            os.path.join("fixed_angle", "applicability_fixed_angle.csv"),
        }
        for rel in files:
            assert os.path.exists(os.path.join(out, rel)), rel
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print("[M22 two-stage gate] PASS")


if __name__ == "__main__":
    main()
