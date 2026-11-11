"""M8 RangeModel (L1 basic capability) -- the UWB observation.

A range is a function of POSITIONS ONLY:

    h_a(x) = ||p_i - a||                     (entity-to-anchor)
    h_e(x) = ||p_i - p_j||                   (entity-to-entity)

so its Jacobian in the 15-dim error state is supported entirely on the position
blocks:

    dh/dp_i = u^T,     dh/dp_j = -u^T,       u = (p_i - p_j)/||p_i - p_j||.

This is the structural fact the whole dual layer rests on: with H the position
selector, ``H E H^T = 0`` for the M12 residual, and the position layer sees the
measurement exactly.  Consequently the full-state design matrix always
factorises as

    H_state = H_position @ S,      S = blkdiag(H, ..., H),

which :func:`build_H_state` computes that way rather than re-deriving it -- a
single definition, so the two can never drift apart.

Noise: ``R = sigma_range^2 I``, independent across edges.  NLOS (``los_prob``,
``nlos_bias_mean``) is demoted on the v3 main line and defaults to off; the code
path stays so section 12.2 can be revisited.
"""

import numpy as np

from config import POS_DIM, STATE_DIM, IDX_P
from l1_geometry import edge_endpoints, los_unit


def position_selector_stack(K):
    """S = blkdiag(H, ..., H) mapping a 15K error state to its 3K positions."""
    S = np.zeros((POS_DIM * K, STATE_DIM * K))
    for i in range(K):
        S[i * POS_DIM:(i + 1) * POS_DIM,
          i * STATE_DIM + IDX_P.start:i * STATE_DIM + IDX_P.stop] = np.eye(POS_DIM)
    return S


def predict_range(edge, positions, anchors):
    """Return the predicted range and the LOS unit vector of one edge."""
    p_i, p_j = edge_endpoints(edge, positions, anchors)
    u, rho = los_unit(p_i, p_j)
    return rho, u


def range_row_position(edge, positions, anchors, K):
    """Row of dh/d(position stack), shape (3K,)."""
    row = np.zeros(POS_DIM * K)
    _, u = predict_range(edge, positions, anchors)
    row[edge.i * POS_DIM:(edge.i + 1) * POS_DIM] = u
    if not edge.is_anchor:
        row[edge.j * POS_DIM:(edge.j + 1) * POS_DIM] = -u
    return row


def build_H_position(edges, positions, anchors, K):
    """Design matrix in the position layer, shape (m, 3K)."""
    if not edges:
        return np.zeros((0, POS_DIM * K))
    return np.stack([range_row_position(e, positions, anchors, K) for e in edges])


def build_H_state(edges, positions, anchors, K):
    """Design matrix in the full error state, shape (m, 15K).

    Defined as ``H_position @ S`` so that the two representations are the same
    object by construction (the identity the M12 diagnostic asserts).
    """
    return build_H_position(edges, positions, anchors, K) @ position_selector_stack(K)


def predict_all(edges, positions, anchors):
    """Predicted ranges of an edge list, shape (m,)."""
    if not edges:
        return np.zeros(0)
    return np.array([predict_range(e, positions, anchors)[0] for e in edges])


def measurement_noise(num_edges, sigma_range):
    """R = sigma_range^2 I."""
    return float(sigma_range) ** 2 * np.eye(int(num_edges))


def simulate_ranges(edges, true_positions, anchors, sigma_range, rng,
                    los_prob=1.0, nlos_bias_mean=0.0):
    """Generate noisy ranges from the TRUE positions.

    On the main line ``los_prob = 1`` and the return is simply
    ``h(p_true) + N(0, sigma^2)``.  With ``los_prob < 1`` an exponential positive
    bias is added on the NLOS edges (ablation path only).
    """
    m = len(edges)
    if m == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    z = predict_all(edges, true_positions, anchors)
    z = z + float(sigma_range) * rng.standard_normal(m)
    nlos = np.zeros(m, dtype=bool)
    if los_prob < 1.0 and nlos_bias_mean > 0.0:
        nlos = rng.random(m) > float(los_prob)
        if np.any(nlos):
            z[nlos] += rng.exponential(float(nlos_bias_mean), int(nlos.sum()))
    return z, nlos


def innovation(z, edges, positions, anchors):
    """Measurement residual ``z - h(p_hat)``."""
    return np.asarray(z, dtype=float).ravel() - predict_all(edges, positions, anchors)


def innovation_variance(H_row, P, R_scalar):
    """S = h P h^T + R for a single edge row."""
    H_row = np.asarray(H_row, dtype=float).ravel()
    return float(H_row @ np.asarray(P, dtype=float) @ H_row) + float(R_scalar)


if __name__ == "__main__":
    from config import default_config, validate_config
    from l1_geometry import build_edges, Edge

    cfg = default_config(); validate_config(cfg)
    K = cfg["num_entities"]
    anchors = cfg["anchor_positions"]
    rng = np.random.default_rng(5)
    positions = [np.array([20.0 + 15.0 * i, 30.0 + 7.0 * (i % 3), 2.0 + 0.5 * i])
                 for i in range(K)]
    edges = build_edges(cfg)

    # ---- 1) prediction agrees with the plain norm ----
    e = edges[0]
    rho, u = predict_range(e, positions, anchors)
    assert abs(rho - np.linalg.norm(positions[e.i] - anchors[e.anchor])) < 1e-12
    assert abs(np.linalg.norm(u) - 1.0) < 1e-12

    # ---- 2) Jacobian rows have the +u / -u structure and nothing else ----
    ee = next(x for x in edges if not x.is_anchor)
    row = range_row_position(ee, positions, anchors, K)
    _, u = predict_range(ee, positions, anchors)
    assert np.allclose(row[ee.i * 3:(ee.i + 1) * 3], u)
    assert np.allclose(row[ee.j * 3:(ee.j + 1) * 3], -u)
    assert abs(np.sum(np.abs(row)) - 2 * np.sum(np.abs(u))) < 1e-12
    row_a = range_row_position(e, positions, anchors, K)
    assert int(np.count_nonzero(np.abs(row_a) > 1e-15)) <= 3

    # ---- 3) numerical Jacobian check on both edge kinds ----
    h = 1e-7
    for probe in (e, ee):
        analytic = range_row_position(probe, positions, anchors, K)
        numeric = np.zeros_like(analytic)
        for k in range(POS_DIM * K):
            pert = [p.copy() for p in positions]
            pert[k // 3][k % 3] += h
            numeric[k] = (predict_range(probe, pert, anchors)[0]
                          - predict_range(probe, positions, anchors)[0]) / h
        assert np.max(np.abs(analytic - numeric)) < 1e-5, probe

    # ---- 4) H_state = H_position @ S, and its position support ----
    Hp = build_H_position(edges, positions, anchors, K)
    Hs = build_H_state(edges, positions, anchors, K)
    S = position_selector_stack(K)
    assert Hp.shape == (len(edges), 3 * K) and Hs.shape == (len(edges), 15 * K)
    assert np.allclose(Hs, Hp @ S)
    # every non-position column is exactly zero -> H E H^T = 0 is structural
    mask = np.ones(15 * K, dtype=bool)
    for i in range(K):
        mask[i * 15:i * 15 + 3] = False
    assert np.all(Hs[:, mask] == 0.0)

    # ---- 5) noise matrix ----
    R = measurement_noise(len(edges), cfg["sigma_range"])
    assert R.shape == (len(edges),) * 2
    assert abs(R[0, 0] - cfg["sigma_range"] ** 2) < 1e-18

    # ---- 6) the simulator is unbiased with the configured sigma ----
    devs = []
    for _ in range(300):
        z, nlos = simulate_ranges(edges, positions, anchors, cfg["sigma_range"],
                                  rng, cfg["los_prob"], cfg["nlos_bias_mean"])
        assert not nlos.any()
        devs.append(z - predict_all(edges, positions, anchors))
    devs = np.concatenate(devs)
    assert abs(float(np.mean(devs))) < 4.0 * cfg["sigma_range"] / np.sqrt(devs.size)
    assert abs(float(np.std(devs)) - cfg["sigma_range"]) < 0.05 * cfg["sigma_range"]

    # ---- 7) NLOS path stays available but is off by default ----
    z_n, nlos = simulate_ranges(edges, positions, anchors, cfg["sigma_range"],
                                np.random.default_rng(2), 0.5, 1.0)
    assert nlos.any() and float(np.mean(z_n - predict_all(edges, positions, anchors))) > 0

    # ---- 8) innovation is zero-mean at the truth ----
    z, _ = simulate_ranges(edges, positions, anchors, 0.0, rng)
    assert np.max(np.abs(innovation(z, edges, positions, anchors))) < 1e-12

    # ---- 9) innovation variance of one edge ----
    P = np.eye(3 * K) * 0.25
    s = innovation_variance(range_row_position(ee, positions, anchors, K), P,
                            cfg["sigma_range"] ** 2)
    assert abs(s - (0.25 * 2 + cfg["sigma_range"] ** 2)) < 1e-12

    # ---- 10) a degenerate (coincident) pair yields a zero row, not a NaN ----
    same = [positions[0].copy() for _ in range(K)]
    r0 = range_row_position(Edge(0, 1), same, anchors, K)
    assert np.all(np.isfinite(r0)) and np.allclose(r0, 0.0)

    print("[M8 RangeModel] OK  (%d edges, H_state = H_pos @ S exactly, "
          "sigma_hat=%.4f vs %.4f)"
          % (len(edges), float(np.std(devs)), cfg["sigma_range"]))
