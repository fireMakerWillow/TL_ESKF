"""Estimator registry for the seven supported v3 methods.

All estimator name resolution lives here.  The supported set is exactly the
five covariance architectures used by RQ1 plus the two sensor ablations.  M18
is intentionally vacant: the obsolete estimator previously assigned to that
number was physically removed, so no import path, factory entry, configuration
name, resource row, or test hook remains.
"""

from config import (ESTIMATOR_NAMES, MAIN_LINE_ESTIMATORS,
                    SENSOR_ABLATION_ESTIMATORS)
from l2_centralized_eskf import CentralizedESKF
from l2_ci_eskf import CIDetESKF, CITraceESKF
from l2_dual_layer import DualLayerESKF
from l2_imu_only import ImuOnlyESKF
from l2_local_only import LocalOnlyESKF
from l2_uwb_only import UwbOnlyKF

ESTIMATOR_CLASSES = {
    "centralized": CentralizedESKF,      # M15
    "dual_layer": DualLayerESKF,         # M19
    "local_only": LocalOnlyESKF,         # M16 DCL-compatible
    "ci_trace": CITraceESKF,             # M17 trace objective
    "ci_det": CIDetESKF,                 # M17 determinant objective
    "imu_only": ImuOnlyESKF,             # M25
    "uwb_only": UwbOnlyKF,               # M26
}

_missing = sorted(set(ESTIMATOR_NAMES) - set(ESTIMATOR_CLASSES))
_extra = sorted(set(ESTIMATOR_CLASSES) - set(ESTIMATOR_NAMES))
if _missing or _extra:
    raise ImportError(
        "estimator registry disagrees with config.ESTIMATOR_NAMES: "
        f"missing {_missing}, unexpected {_extra}")
for _key, _cls in ESTIMATOR_CLASSES.items():
    if getattr(_cls, "name", None) != _key:
        raise ImportError(
            f"{_cls.__name__}.name is {getattr(_cls, 'name', None)!r} but it is "
            f"registered as {_key!r}")


def estimator_class(name):
    try:
        return ESTIMATOR_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"unknown estimator {name!r}; known names are "
            f"{sorted(ESTIMATOR_CLASSES)}") from None


def build(name, config):
    return estimator_class(name)(config)


def build_all(names, config):
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate estimator names: {names}")
    return {name: build(name, config) for name in names}


def main_line():
    return list(MAIN_LINE_ESTIMATORS)


def sensor_ablations():
    return list(SENSOR_ABLATION_ESTIMATORS)


def default_set():
    return main_line() + sensor_ablations()


def is_main_line(name):
    return name in MAIN_LINE_ESTIMATORS


def provides_full_cov(name):
    return bool(getattr(estimator_class(name), "provides_full_cov", True))


def evaluated_channels(name):
    return tuple(estimator_class(name).evaluated_channels)


if __name__ == "__main__":
    import numpy as np
    from config import default_config, validate_config
    from l1_resource import METHODS, resource_summary

    cfg = default_config(); validate_config(cfg)
    assert len(ESTIMATOR_CLASSES) == 7
    assert sorted(default_set()) == sorted(ESTIMATOR_NAMES)
    assert sorted(METHODS) == sorted(ESTIMATOR_NAMES)
    built = build_all(ESTIMATOR_NAMES, cfg)
    for name, est in built.items():
        assert est.name == name
        row = resource_summary(name, cfg)
        assert np.isfinite(float(row["storage_floats"]))
        for method in ("reset", "predict", "update", "get_eval_cov",
                       "get_pos_estimate", "get_comm_summary"):
            assert callable(getattr(est, method)), (name, method)
    assert built["ci_trace"].objective == "trace"
    assert built["ci_det"].objective == "det"
    assert not provides_full_cov("uwb_only")
    assert evaluated_channels("uwb_only") == ("pos",)
    print("[l2_estimators registry] OK  (7 supported methods; M18 vacant)")
