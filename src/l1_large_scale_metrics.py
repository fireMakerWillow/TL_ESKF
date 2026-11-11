"""Streaming metrics for the 120 s large-scale experiment.

This module keeps only running sums.  It deliberately does not retain state,
error, or covariance histories, which makes many independent server processes
practical.  RMSE/RMTE are reported for every 3-D state block and component;
full-state output is TMNNEES only, as required by the experiment protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from config import (
    ENTITY_CLASSES,
    IDX_BA,
    IDX_BG,
    IDX_P,
    IDX_TH,
    IDX_V,
    POS_DIM,
    STATE_DIM,
)
from l1_linalg import safe_inverse, symmetrize


BLOCKS = {
    "position": IDX_P,
    "velocity": IDX_V,
    "attitude": IDX_TH,
    "gyro_bias": IDX_BG,
    "accel_bias": IDX_BA,
}
COMPONENTS = ("x", "y", "z")
UNIT_SCALE = {
    "position": 1.0,
    "velocity": 1.0,
    "attitude": 180.0 / np.pi,
    "gyro_bias": (180.0 / np.pi) * 3600.0,
    "accel_bias": 1.0e6 / 9.80665,
}
UNITS = {
    "position": "m",
    "velocity": "m/s",
    "attitude": "deg",
    "gyro_bias": "deg/h",
    "accel_bias": "ug",
}


def _entity_state_indices(entities: Iterable[int], sl: slice) -> np.ndarray:
    parts = [
        np.arange(i * STATE_DIM + sl.start, i * STATE_DIM + sl.stop, dtype=int)
        for i in entities
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def _entity_component_indices(entities: Iterable[int], sl: slice, axis: int) -> np.ndarray:
    return np.asarray(
        [i * STATE_DIM + sl.start + int(axis) for i in entities], dtype=int
    )


def _position_indices(entities: Iterable[int], axis: int | None = None) -> np.ndarray:
    if axis is None:
        parts = [
            np.arange(i * POS_DIM, (i + 1) * POS_DIM, dtype=int)
            for i in entities
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=int)
    return np.asarray([i * POS_DIM + int(axis) for i in entities], dtype=int)


def _quadratic_form(e: np.ndarray, P: np.ndarray, eps: float) -> float:
    """Return e' P^-1 e using a Cholesky solve with a stable fallback."""
    e = np.asarray(e, dtype=float).ravel()
    if e.size == 0:
        return float("nan")
    P = symmetrize(np.asarray(P, dtype=float))
    if P.shape != (e.size, e.size) or not np.all(np.isfinite(P)):
        return float("nan")
    diag_scale = max(1.0, float(np.max(np.abs(np.diag(P)))))
    jitter = max(float(eps), np.finfo(float).eps) * diag_scale
    try:
        L = np.linalg.cholesky(P + jitter * np.eye(e.size))
        y = np.linalg.solve(L, e)
        return float(y @ y)
    except np.linalg.LinAlgError:
        try:
            return float(e @ safe_inverse(P, eps) @ e)
        except Exception:
            return float("nan")


@dataclass
class _ScalarMean:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        if np.isfinite(value):
            self.total += float(value)
            self.count += 1

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


class StreamingMetricAccumulator:
    """Accumulate the complete large-scale metric protocol without histories."""

    def __init__(self, config, channels, burn_in_s: float = 20.0):
        self.config = config
        self.class_map = list(config["entity_class_map"])
        self.K = int(config["num_entities"])
        self.channels = set(channels)
        self.burn_in_s = float(burn_in_s)
        self.eps = float(config.get("reg_epsilon", 1e-9))
        self.epochs_seen = 0
        self.epochs_used = 0

        self.groups: Dict[str, List[int]] = {
            "all": list(range(self.K)),
            **{
                f"class_{label}": [
                    i for i, current in enumerate(self.class_map) if current == label
                ]
                for label in ENTITY_CLASSES
            },
            **{f"vehicle_{i:03d}": [i] for i in range(self.K)},
        }

        self.error_sq = {}
        self.cov_sum = {}
        self.nnees = {}
        self.full_nnees = {group: _ScalarMean() for group in self.groups}
        for group in self.groups:
            self.error_sq[group] = {}
            self.cov_sum[group] = {}
            self.nnees[group] = {}
            for block in BLOCKS:
                self.error_sq[group][block] = {
                    "components": np.zeros(3, dtype=float),
                    "block": 0.0,
                    "scalar_count": 0,
                }
                self.cov_sum[group][block] = {
                    "components": np.zeros(3, dtype=float),
                    "block": 0.0,
                    "scalar_count": 0,
                }
                self.nnees[group][block] = {
                    "components": [_ScalarMean() for _ in range(3)],
                    "block": _ScalarMean(),
                }

    def update(self, time_s: float, E: np.ndarray, eval_cov: np.ndarray,
               full_cov: np.ndarray | None) -> None:
        self.epochs_seen += 1
        if float(time_s) <= self.burn_in_s:
            return
        E = np.asarray(E, dtype=float)
        if E.shape != (self.K, STATE_DIM):
            raise ValueError(f"expected error shape {(self.K, STATE_DIM)}, got {E.shape}")
        eval_cov = np.asarray(eval_cov, dtype=float)
        P_full = None if full_cov is None else np.asarray(full_cov, dtype=float)
        self.epochs_used += 1

        for group, entities in self.groups.items():
            n = len(entities)
            if n == 0:
                continue
            for block, sl in BLOCKS.items():
                if self._channel_name(block) not in self.channels:
                    continue
                block_error = E[np.asarray(entities), sl]
                sq = np.sum(block_error ** 2, axis=0)
                self.error_sq[group][block]["components"] += sq
                self.error_sq[group][block]["block"] += float(np.sum(sq))
                self.error_sq[group][block]["scalar_count"] += n

                P_source, idx_block = self._cov_and_indices(
                    block, entities, eval_cov, P_full)
                if P_source is None:
                    continue
                P_block = P_source[np.ix_(idx_block, idx_block)]
                diag = np.diag(P_block).reshape(n, 3)
                self.cov_sum[group][block]["components"] += np.sum(diag, axis=0)
                self.cov_sum[group][block]["block"] += float(np.sum(diag))
                self.cov_sum[group][block]["scalar_count"] += n

                q = _quadratic_form(block_error.ravel(), P_block, self.eps)
                self.nnees[group][block]["block"].add(q / (3.0 * n))
                for axis in range(3):
                    idx_axis = self._component_indices(
                        block, entities, axis, eval_cov, P_full)
                    P_axis = P_source[np.ix_(idx_axis, idx_axis)]
                    q_axis = _quadratic_form(block_error[:, axis], P_axis, self.eps)
                    self.nnees[group][block]["components"][axis].add(q_axis / n)

            if P_full is not None:
                idx_full = _entity_state_indices(entities, slice(0, STATE_DIM))
                q_full = _quadratic_form(
                    E[np.asarray(entities), :].ravel(),
                    P_full[np.ix_(idx_full, idx_full)],
                    self.eps,
                )
                self.full_nnees[group].add(q_full / (STATE_DIM * n))

    @staticmethod
    def _channel_name(block: str) -> str:
        return {
            "position": "pos",
            "velocity": "vel",
            "attitude": "att",
            "gyro_bias": "gyro_bias",
            "accel_bias": "acc_bias",
        }[block]

    def _cov_and_indices(self, block, entities, eval_cov, P_full):
        if block == "position":
            return eval_cov, _position_indices(entities)
        if P_full is None:
            return None, np.empty(0, dtype=int)
        return P_full, _entity_state_indices(entities, BLOCKS[block])

    def _component_indices(self, block, entities, axis, eval_cov, P_full):
        if block == "position":
            return _position_indices(entities, axis)
        return _entity_component_indices(entities, BLOCKS[block], axis)

    def finalize(self) -> dict:
        out = {
            "burn_in_s": self.burn_in_s,
            "epochs_seen": int(self.epochs_seen),
            "epochs_used": int(self.epochs_used),
            "groups": {},
        }
        for group, entities in self.groups.items():
            group_out = {
                "entities": list(entities),
                "states": {},
                "full_state": {
                    "tmnnees": self.full_nnees[group].mean(),
                    "unit": "1",
                },
            }
            for block in BLOCKS:
                scale = float(UNIT_SCALE[block])
                err = self.error_sq[group][block]
                cov = self.cov_sum[group][block]
                denom_err = int(err["scalar_count"])
                denom_cov = int(cov["scalar_count"])
                components = {}
                for axis, name in enumerate(COMPONENTS):
                    rmse = (
                        np.sqrt(err["components"][axis] / denom_err) * scale
                        if denom_err else float("nan")
                    )
                    rmte = (
                        np.sqrt(cov["components"][axis] / denom_cov) * scale
                        if denom_cov else float("nan")
                    )
                    components[name] = {
                        "rmse": float(rmse),
                        "rmte": float(rmte),
                        "tmnnees": self.nnees[group][block]["components"][axis].mean(),
                    }
                block_rmse = (
                    np.sqrt(err["block"] / denom_err) * scale
                    if denom_err else float("nan")
                )
                block_rmte = (
                    np.sqrt(cov["block"] / denom_cov) * scale
                    if denom_cov else float("nan")
                )
                group_out["states"][block] = {
                    "unit": UNITS[block],
                    "components": components,
                    "block": {
                        "rmse": float(block_rmse),
                        "rmte": float(block_rmte),
                        "tmnnees": self.nnees[group][block]["block"].mean(),
                    },
                }
            out["groups"][group] = group_out
        return out
