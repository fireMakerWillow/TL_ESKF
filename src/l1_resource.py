"""M11 ResourceModel (L1 evaluation kernel) -- analytic cost accounting.

Three resources are tracked: communication, arithmetic work and covariance
storage.  Communication remains split into raw measurement traffic and
estimator-state traffic::

    C_total = C_measurement + C_estimation

Most methods consume every generated range.  The DCL-compatible ``local_only``
method is the exception: its persistent measurement stride processes only every
``dcl_measurement_stride``-th inter-entity range.  The skipped measurements are
not fused and are not charged as communication.  Each accepted undirected pair
exchanges one local position/covariance snapshot in each direction,
``2 * (3 + 6) = 18`` scalars.  Anchor ranges remain local absolute references
and are always processed.

Per entity and per UWB period (before DCL stride averaging):

    Centralized   15 state + 15x15 symmetric covariance (120) + correction (15)
    Dual-layer    P_G^ii (6) + A_i (9) + delta p_i (3)
    DCL/local     per accepted neighbour and direction: p (3) + P_pp (6)
    CI-trace/det  one post-anchor p/P_pp broadcast (9)
    IMU-only      no communication
    UWB-only      position aggregation (3K) + dense joint 3K covariance

Storage (floats, covariance objects only), with d = 15 and r = 3:

    Centralized  (dK)^2                   Dual-layer  (rK)^2 + K d^2
    DCL/local    K d^2                    CI          K d^2
    IMU-only     K d^2                    UWB-only    (rK)^2
"""

import numpy as np

from config import POS_DIM, STATE_DIM, class_indices

_EXP_LOG_FLOPS = 60          # nominal cost of one Exp or Log

METHODS = ("centralized", "dual_layer", "local_only", "ci_trace", "ci_det",
           "imu_only", "uwb_only")


# ----------------------------------------------------------------------
# FLOPs
# ----------------------------------------------------------------------
def flops_matmul(m, n, p):
    """FLOPs of an (m x n) @ (n x p) product."""
    return 2 * int(m) * int(n) * int(p)


def flops_inverse(d):
    """FLOPs of a dense d x d factorisation / inverse."""
    return 2 * int(d) ** 3


def flops_cholesky(d):
    """FLOPs of a d x d Cholesky factorisation."""
    return int(d) ** 3 // 3


def flops_manifold(count=1):
    """FLOPs of ``count`` Exp/Log evaluations."""
    return _EXP_LOG_FLOPS * int(count)


def flops_covariance_propagation(d, blocks=1):
    """Phi P Phi^T + Q for ``blocks`` d x d blocks."""
    return blocks * (2 * flops_matmul(d, d, d) + d * d)


def flops_scalar_update(d):
    """One sequential scalar Kalman update on a d-dim state."""
    return 2 * d * d + 4 * d


# ----------------------------------------------------------------------
# communication
# ----------------------------------------------------------------------
def symmetric_scalars(d):
    """Number of independent entries of a symmetric d x d matrix."""
    d = int(d)
    return d * (d + 1) // 2


def measurement_scalars(config, method=None):
    """Range scalars per UWB epoch.

    With ``method=None`` this is the raw sensor-network count.  For
    ``method='local_only'`` the returned ``total`` is the number processed in
    the first epoch of a fresh persistent DCL stride counter; ``raw_total``
    keeps the unthinned count visible.
    """
    K = int(config["num_entities"])
    groups = class_indices(config)
    table = config["anchor_visible_by_class"]
    topology = config["entity_topology"]
    if topology == "full":
        inter = K * (K - 1) // 2
    elif topology == "knn":
        # Fixed circular k-regular graph: exact degree k and K*k/2 edges.
        k = int(config.get("knn_k", 4))
        inter = K * k // 2
    else:
        inter = K * (K - 1) // 2
    anchor = sum(len(table[label]) * len(groups[label]) for label in groups)
    raw_total = int(inter + anchor)
    if method != "local_only":
        return {"inter": int(inter), "anchor": int(anchor),
                "total": raw_total, "raw_total": raw_total,
                "processed_inter": int(inter), "skipped_inter": 0}
    counts = dcl_stride_counts(config, epochs=1)
    return {"inter": int(inter), "anchor": int(anchor),
            "total": int(anchor + counts["processed_inter"]),
            "raw_total": raw_total,
            "processed_inter": counts["processed_inter"],
            "skipped_inter": counts["skipped_inter"]}


def dcl_stride_counts(config, epochs=1, start_counter=0):
    """Exact static-topology DCL stride counts over ``epochs`` periods.

    The estimator accepts counter values s, 2s, ... using a persistent 1-based
    counter.  The fixed circular topology has an exact static edge count, so
    this helper is exact for ``entity_topology='knn'`` as well.
    """
    epochs = int(epochs)
    start_counter = int(start_counter)
    if epochs < 0 or start_counter < 0:
        raise ValueError("epochs and start_counter must be non-negative")
    stride = int(config.get("dcl_measurement_stride", 1))
    if stride < 1:
        raise ValueError("dcl_measurement_stride must be positive")
    inter_per_epoch = measurement_scalars(config)["inter"]
    arrivals = int(inter_per_epoch) * epochs
    processed = ((start_counter + arrivals) // stride
                 - start_counter // stride)
    return {
        "stride": stride,
        "inter_per_epoch": int(inter_per_epoch),
        "arrivals": arrivals,
        "processed_inter": int(processed),
        "skipped_inter": int(arrivals - processed),
        "end_counter": int(start_counter + arrivals),
    }


def estimation_scalars(method, config):
    """Estimation scalars per UWB epoch: per entity and network total.

    Returns ``{'per_entity': x, 'total': y, 'detail': str}``.  ``per_entity`` is
    the number a single node sends in one period; ``total`` is the network sum,
    which is what the byte counter accumulates.
    """
    K = int(config["num_entities"])
    d, r = STATE_DIM, POS_DIM
    if method == "centralized":
        up = d + symmetric_scalars(d)                 # 15 + 120 = 135
        down = d                                      # global correction
        per = up + down
        detail = f"uplink {up} (state {d} + cov {symmetric_scalars(d)}), downlink {down}"
    elif method == "dual_layer":
        up = symmetric_scalars(r) + r * r             # 6 + 9 = 15
        down = r                                      # delta p_i
        per = up + down
        detail = f"uplink {up} (P_G^ii {symmetric_scalars(r)} + A_i {r * r}), downlink {down}"
    elif method == "local_only":
        stride = int(config.get("dcl_measurement_stride", 1))
        inter = measurement_scalars(config)["inter"]
        # Long-run average.  Runtime/resource_summary use exact integer counts.
        total = (2.0 * (r + symmetric_scalars(r)) * inter / stride)
        per = total / K if K else 0.0
        detail = (f"DCL stride {stride}: long-run average 2 directions x "
                  f"(p {r} + P_pp {symmetric_scalars(r)}) x "
                  f"{inter}/{stride} accepted pairs")
    elif method in ("ci_trace", "ci_det"):
        per = r + symmetric_scalars(r)                 # one epoch broadcast
        detail = (f"one post-anchor broadcast (p {r} + P_pp "
                  f"{symmetric_scalars(r)}); omega solved locally")
    elif method == "imu_only":
        per = 0
        detail = "no communication"
    elif method == "uwb_only":
        per = r * K + symmetric_scalars(r * K)
        detail = f"position aggregation {r * K} + joint cov {symmetric_scalars(r * K)}"
    else:
        raise ValueError(f"unknown method {method!r}")
    if method != "local_only":
        total = per * (1 if method == "uwb_only" else K)
    return {"per_entity": per, "total": total, "detail": detail}


def comm_bytes(num_scalars, float_bytes):
    """Bytes needed to transmit ``num_scalars`` floats."""
    return int(num_scalars) * int(float_bytes)


class CommunicationLedger:
    """Accumulates the two communication categories separately.

    Keeping one counter per category preserves the distinction between raw
    measurements and estimator-state exchange.  DCL's stride makes both
    categories method-dependent, so the split is also needed to show where its
    communication reduction comes from.
    """

    def __init__(self, config, method):
        self.cfg = config
        self.method = method
        self.float_bytes = int(config["float_bytes"])
        self.measurement_scalars = 0
        self.estimation_scalars = 0
        self.epochs = 0
        self._dcl_relative_counter = 0

    def add_epoch(self, measurement=None, estimation=None):
        """Charge one UWB epoch; defaults come from the analytic model.

        DCL's defaults advance the same persistent stride phase as the runtime
        estimator.  The estimator itself supplies realised exact counts, which
        also supports dynamic topologies.
        """
        if self.method == "local_only" and measurement is None and estimation is None:
            counts = dcl_stride_counts(
                self.cfg, epochs=1, start_counter=self._dcl_relative_counter)
            self._dcl_relative_counter = counts["end_counter"]
            anchor = measurement_scalars(self.cfg)["anchor"]
            processed = counts["processed_inter"]
            measurement = anchor + processed
            estimation = 2 * (POS_DIM + symmetric_scalars(POS_DIM)) * processed
        else:
            if measurement is None:
                measurement = measurement_scalars(self.cfg)["total"]
            if estimation is None:
                estimation = estimation_scalars(self.method, self.cfg)["total"]
        self.measurement_scalars += int(measurement)
        self.estimation_scalars += int(estimation)
        self.epochs += 1
        return self

    def add_estimation(self, scalars):
        """Charge extra estimation scalars actually sent by the estimator."""
        self.estimation_scalars += int(scalars)
        return self

    def summary(self):
        fb = self.float_bytes
        c_m = comm_bytes(self.measurement_scalars, fb)
        c_e = comm_bytes(self.estimation_scalars, fb)
        return {
            "epochs": self.epochs,
            "C_measurement_scalars": self.measurement_scalars,
            "C_estimation_scalars": self.estimation_scalars,
            "C_measurement_bytes": c_m,
            "C_estimation_bytes": c_e,
            "C_total_bytes": c_m + c_e,
            "C_estimation_bytes_per_epoch": (c_e / self.epochs
                                             if self.epochs else float("nan")),
        }


# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------
def storage_floats(method, K, state_dim=STATE_DIM, pos_dim=POS_DIM):
    """Peak covariance storage in floats for each representation."""
    K, d, r = int(K), int(state_dim), int(pos_dim)
    if method == "centralized":
        return (d * K) ** 2
    if method == "dual_layer":
        return (r * K) ** 2 + K * d ** 2
    if method in ("local_only", "ci_trace", "ci_det", "imu_only"):
        return K * d ** 2
    if method == "uwb_only":
        return (r * K) ** 2
    raise ValueError(f"unknown method {method!r}")


def storage_bytes(method, K, config):
    return storage_floats(method, K, config["state_dim"],
                          config["pos_dim"]) * int(config["float_bytes"])


def storage_scaling(method):
    """Symbolic complexity string used in the paper's resource table."""
    return {
        "centralized": "O(d^2 K^2)",
        "dual_layer": "O(r^2 K^2) + O(d^2 K)",
        "local_only": "O(d^2 K)",
        "ci_trace": "O(d^2 K)",
        "ci_det": "O(d^2 K)",
        "imu_only": "O(d^2 K)",
        "uwb_only": "O(r^2 K^2)",
    }[method]


def compute_scaling(method):
    """Symbolic per-UWB-period computational complexity."""
    return {
        "centralized": "O(d^3 K^3)",
        "dual_layer": "O(K d^3) + O(K^2 r^3)",
        "local_only": "O(K d^3 + E d^2 / s)",
        "ci_trace": "O(K d^3)",
        "ci_det": "O(K d^3)",
        "imu_only": "O(K d^3)",
        "uwb_only": "O(K^2 r^2)",
    }[method]


def resource_summary(method, config, epochs=None):
    """One row of the analytic resource table."""
    K = int(config["num_entities"])
    if epochs is None:
        epochs = int(round(float(config["sim_duration"])
                           / float(config["sync_interval"])))
    ledger = CommunicationLedger(config, method)
    for _ in range(int(epochs)):
        ledger.add_epoch()
    out = ledger.summary()
    out.update({
        "method": method,
        "num_entities": K,
        "estimation_scalars_per_entity_per_epoch":
            (out["C_estimation_scalars"] / (K * int(epochs))
             if K and int(epochs) else float("nan")),
        "measurement_scalars_per_epoch":
            (out["C_measurement_scalars"] / int(epochs)
             if int(epochs) else float("nan")),
        "raw_measurement_scalars_per_epoch": measurement_scalars(config)["total"],
        "dcl_measurement_stride":
            (int(config.get("dcl_measurement_stride", 1))
             if method == "local_only" else 1),
        "storage_floats": storage_floats(method, K, config["state_dim"],
                                         config["pos_dim"]),
        "storage_bytes": storage_bytes(method, K, config),
        "storage_scaling": storage_scaling(method),
        "compute_scaling": compute_scaling(method),
    })
    return out


def comparison_table(config, methods=METHODS, epochs=None):
    """Analytic resource table for a list of methods."""
    return {m: resource_summary(m, config, epochs) for m in methods}


if __name__ == "__main__":
    from config import default_config, validate_config, apply_entity_count

    cfg = default_config()
    assert cfg["dcl_measurement_stride"] == 1
    # Exercise the optional thinning path explicitly; the experiment main line
    # keeps stride=1 so every architecture consumes identical measurements.
    cfg["dcl_measurement_stride"] = 3
    validate_config(cfg)
    K = cfg["num_entities"]

    # ---- 1) FLOP primitives ----
    assert flops_matmul(2, 3, 4) == 48
    assert flops_inverse(3) == 54
    assert flops_manifold(2) == 2 * _EXP_LOG_FLOPS
    assert flops_cholesky(6) == 72
    assert comm_bytes(10, 8) == 80
    assert symmetric_scalars(15) == 120 and symmetric_scalars(3) == 6

    # ---- 2) raw topology and DCL thinning ----
    raw = measurement_scalars(cfg)
    assert raw["inter"] == K * (K - 1) // 2
    assert raw["anchor"] == (4 * len(class_indices(cfg)["A"])
                              + 2 * len(class_indices(cfg)["B"]))
    dc = dcl_stride_counts(cfg, epochs=10)
    assert dc["processed_inter"] == (10 * raw["inter"]
                                      // cfg["dcl_measurement_stride"])
    dcl_first = measurement_scalars(cfg, "local_only")
    assert dcl_first["total"] < raw["total"]

    # ---- 3) estimation traffic ----
    assert estimation_scalars("centralized", cfg)["per_entity"] == 150
    assert estimation_scalars("dual_layer", cfg)["per_entity"] == 18
    assert estimation_scalars("imu_only", cfg)["per_entity"] == 0
    expected_dcl_avg = 18.0 * raw["inter"] / cfg["dcl_measurement_stride"]
    assert abs(estimation_scalars("local_only", cfg)["total"]
               - expected_dcl_avg) < 1e-12
    assert estimation_scalars("ci_trace", cfg)["per_entity"] == 9

    # DCL still grows with K, but the stride reduces its slope.
    big = default_config(); apply_entity_count(big, 20, (4, 12, 4)); validate_config(big)
    assert (estimation_scalars("local_only", big)["per_entity"]
            > estimation_scalars("local_only", cfg)["per_entity"])

    # ---- 4) exact persistent ledger phase ----
    led = CommunicationLedger(cfg, "local_only")
    for _ in range(10):
        led.add_epoch()
    s = led.summary()
    assert s["C_measurement_scalars"] == 10 * raw["anchor"] + dc["processed_inter"]
    assert s["C_estimation_scalars"] == 18 * dc["processed_inter"]
    led.add_estimation(7)
    assert led.summary()["C_estimation_scalars"] == 18 * dc["processed_inter"] + 7

    # Non-DCL methods retain the raw measurement count.
    dual = CommunicationLedger(cfg, "dual_layer")
    for _ in range(10):
        dual.add_epoch()
    ds = dual.summary()
    assert ds["C_measurement_scalars"] == 10 * raw["total"]

    # ---- 5) storage is unchanged by stride ----
    s_cent = storage_floats("centralized", K)
    s_dual = storage_floats("dual_layer", K)
    s_loc = storage_floats("local_only", K)
    assert s_cent == (15 * K) ** 2 and s_loc == K * 225
    assert s_loc < s_dual < s_cent

    # ---- 6) assembled table exposes raw and processed traffic ----
    table = comparison_table(cfg, ("centralized", "dual_layer", "local_only",
                                   "ci_trace", "ci_det", "imu_only", "uwb_only"))
    assert table["local_only"]["measurement_scalars_per_epoch"]         < table["local_only"]["raw_measurement_scalars_per_epoch"]
    assert table["centralized"]["measurement_scalars_per_epoch"]         == table["centralized"]["raw_measurement_scalars_per_epoch"]
    assert table["local_only"]["storage_floats"] == K * STATE_DIM ** 2

    print("[M11 ResourceModel] OK  (DCL stride=%d; raw=%d, processed=%.2f "
          "meas/epoch; C_est=%.2f scalars/epoch; storage unchanged)"
          % (cfg["dcl_measurement_stride"], raw["total"],
             table["local_only"]["measurement_scalars_per_epoch"],
             table["local_only"]["C_estimation_scalars"]
             / table["local_only"]["epochs"]))
