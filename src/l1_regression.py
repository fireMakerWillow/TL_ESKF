"""M9 RegressionLift (L1 basic capability) -- the position -> full-state lift.

For one entity, partition the local covariance in the M0 state order

    P_L,i = [[P_pp, P_pz],
             [P_zp, P_zz]]          p = position (3), z = the other 12 states

The regression (conditional-mean) lift is

    L_i = [ I_3 ; B_i ],            B_i = P_zp P_pp^{-1}                (15 x 3)

so that ``L_i dp`` is the best linear estimate of the WHOLE local error given a
position correction ``dp``.  Two consequences are used everywhere downstream:

    H L_i = I_3                     (the first three rows of L are the identity)
    E_pp = 0  and  H E H^T = 0      for the M12 structural residual

The Schur complement ``D_i = P_zz - B_i P_pp B_i^T`` is the part of the local
covariance that the position layer CANNOT represent; it is what stays in the
local filter after the push-down.

Dual-L contract (v3, plan sections 3.2 / 3.6 / 3.8).  One UWB period needs TWO
lifts computed at TWO different instants; they are NOT interchangeable:

    L_i^rst  -- from P_L,i AFTER the previous reset.  Used to assemble
                A_i = H Phi_bar_i L_i^rst for this period's cross-block
                propagation P_G^ij <- A_i P_G^ij A_j^T.
    L_i^-    -- from P_L,i AFTER this period's IMU propagation and BEFORE the
                UWB update.  Used to push down c_i = L_i^- delta p_i and
                P_L,i^+ = P_L,i^- - L_i^- Delta_i L_i^-^T.

The module therefore exposes a single pure function ``compute_L(P_L, eps)``
(alias ``lift_matrix``); M19 calls it at both instants and caches the two
results separately.  ``lifts_distinguishable`` is the runtime assertion that the
two caches were not accidentally aliased.

The reset changes P_zp,i <- G_z,i P_zp,i; a cached stale L_i injects a bias
proportional to the correction, so ``recompute_L_after_reset=False`` is an
ablation only.
"""

import numpy as np

from config import POS_DIM, STATE_DIM, IDX_P
from l1_linalg import safe_inverse, symmetrize


def lift_matrix(P_local, eps=1e-9):
    """Return the (15, 3) regression lift ``L = [I; P_zp P_pp^{-1}]``."""
    P_local = np.asarray(P_local, dtype=float)
    if P_local.shape != (STATE_DIM, STATE_DIM):
        raise ValueError(
            f"local covariance must be {(STATE_DIM, STATE_DIM)}, got {P_local.shape}")
    P_pp = symmetrize(P_local[IDX_P, IDX_P])
    P_zp = P_local[POS_DIM:, IDX_P]
    scale = float(np.max(np.abs(np.diag(P_pp)))) if P_pp.size else 1.0
    scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    # Implementation-only numerical safeguard: the manuscript mapping uses
    # the unregularized P_zp P_pp^{-1}; this scale-normalized perturbation is
    # added here solely to keep a nearly singular numerical inverse finite.
    P_pp_reg = P_pp + float(eps) * scale * np.eye(POS_DIM)
    L = np.zeros((STATE_DIM, POS_DIM))
    L[IDX_P, :] = np.eye(POS_DIM)
    L[POS_DIM:, :] = P_zp @ safe_inverse(P_pp_reg)
    return L


def compute_L(P_local, eps=1e-9):
    """Alias of :func:`lift_matrix` naming the dual-L contract explicitly."""
    return lift_matrix(P_local, eps)


def local_residual(P_local, L=None, eps=1e-9):
    """Schur complement ``D = P_zz - B P_pp B^T`` embedded back in 15x15.

    This is the part of the local uncertainty that survives a perfectly known
    position; the dual layer keeps it in the local filter untouched.
    """
    P_local = np.asarray(P_local, dtype=float)
    L = lift_matrix(P_local, eps) if L is None else np.asarray(L, dtype=float)
    P_pp = symmetrize(P_local[IDX_P, IDX_P])
    D = np.zeros((STATE_DIM, STATE_DIM))
    B = L[POS_DIM:, :]
    D[POS_DIM:, POS_DIM:] = symmetrize(P_local[POS_DIM:, POS_DIM:]
                                       - B @ P_pp @ B.T)
    return D


def reconstruct(P_local, P_pos_block, eps=1e-9):
    """``L P_pos L^T + D`` -- the two-layer reconstruction of a local block.

    When ``P_pos_block`` equals the local position block, this returns exactly
    ``P_local``: the representation is lossless on the diagonal and only loses
    information across entities.
    """
    L = lift_matrix(P_local, eps)
    return symmetrize(L @ np.asarray(P_pos_block, dtype=float) @ L.T
                      + local_residual(P_local, L, eps))


def lifts_distinguishable(L_minus, L_reset, tol=0.0):
    """True when the two dual-L caches are not bit-identical objects/values."""
    L_minus = np.asarray(L_minus, dtype=float)
    L_reset = np.asarray(L_reset, dtype=float)
    if L_minus is L_reset:
        return False
    return float(np.max(np.abs(L_minus - L_reset))) > tol


def apply_reset_to_lift(P_local, G, eps=1e-9):
    """Lift recomputed from the RESET covariance ``G P_L G^T`` (recompute path)."""
    G = np.asarray(G, dtype=float)
    return lift_matrix(symmetrize(G @ np.asarray(P_local, dtype=float) @ G.T), eps)


if __name__ == "__main__":
    from config import default_config, validate_config
    from l1_linalg import get_block

    cfg = default_config(); validate_config(cfg)
    eps = cfg["reg_epsilon"]
    rng = np.random.default_rng(6)

    M = rng.standard_normal((STATE_DIM, STATE_DIM))
    P = symmetrize(M @ M.T + np.eye(STATE_DIM))
    L = lift_matrix(P, eps)

    # ---- 1) shape and the H L = I structural identity ----
    assert L.shape == (STATE_DIM, POS_DIM)
    assert np.allclose(L[IDX_P, :], np.eye(POS_DIM))
    H = np.zeros((POS_DIM, STATE_DIM)); H[:, IDX_P] = np.eye(POS_DIM)
    assert np.allclose(H @ L, np.eye(POS_DIM))

    # ---- 2) B is the exact regression coefficient ----
    B_exact = P[POS_DIM:, IDX_P] @ np.linalg.inv(P[IDX_P, IDX_P])
    assert np.allclose(L[POS_DIM:, :], B_exact, atol=1e-5)

    # ---- 3) L dp is the conditional mean E[z | p = dp] ----
    dp = rng.standard_normal(POS_DIM)
    cond = P[POS_DIM:, IDX_P] @ np.linalg.solve(P[IDX_P, IDX_P], dp)
    assert np.allclose((L @ dp)[POS_DIM:], cond, atol=1e-5)

    # ---- 4) the Schur residual is PSD and kills the position block ----
    D = local_residual(P, L, eps)
    assert np.allclose(D[IDX_P, :], 0.0) and np.allclose(D[:, IDX_P], 0.0)
    assert np.min(np.linalg.eigvalsh(D[POS_DIM:, POS_DIM:])) > -1e-9

    # ---- 5) the reconstruction is LOSSLESS on the diagonal block ----
    rebuilt = reconstruct(P, P[IDX_P, IDX_P], eps)
    assert np.max(np.abs(rebuilt - P)) < 1e-6 * np.max(np.abs(P)), \
        float(np.max(np.abs(rebuilt - P)))

    # ---- 6) a position-only covariance gives B = 0 ----
    P_diag = symmetrize(np.diag(np.abs(rng.standard_normal(STATE_DIM)) + 1.0))
    assert np.allclose(lift_matrix(P_diag, eps)[POS_DIM:, :], 0.0)

    # ---- 7) singular position block does not blow up ----
    P_sing = P.copy(); P_sing[IDX_P, IDX_P] = 0.0
    assert np.all(np.isfinite(lift_matrix(P_sing, eps)))

    # ---- 8) dual-L contract: two covariances give distinguishable lifts ----
    P2 = P.copy()
    P2[POS_DIM:STATE_DIM, IDX_P] *= 1.05
    P2 = symmetrize(P2)
    assert lifts_distinguishable(lift_matrix(P, eps), lift_matrix(P2, eps))
    assert not lifts_distinguishable(L, L)
    assert compute_L(P, eps) is not L and np.allclose(compute_L(P, eps), L)

    # ---- 9) a reset genuinely moves the lift, so caching it is a bug ----
    G = np.eye(STATE_DIM)
    G[6:9, 6:9] = np.eye(3) - 0.5 * np.array([[0.0, -0.02, 0.01],
                                              [0.02, 0.0, -0.03],
                                              [-0.01, 0.03, 0.0]])
    L_rst = apply_reset_to_lift(P, G, eps)
    assert lifts_distinguishable(L_rst, L, tol=1e-12)
    stale_error = float(np.max(np.abs(L_rst - L)))

    # ---- 10) wrong input shape is rejected loudly ----
    try:
        lift_matrix(np.eye(3), eps)
        raise AssertionError("lift_matrix accepted a wrong shape")
    except ValueError:
        pass

    print("[M9 RegressionLift] OK  (H L = I, reconstruction lossless to %.1e, "
          "stale-lift error after reset %.2e)"
          % (float(np.max(np.abs(rebuilt - P))), stale_error))
