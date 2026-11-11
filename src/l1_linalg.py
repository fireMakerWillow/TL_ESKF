"""M1 LinAlgUtils (L1 basic capability) -- numerical primitives.

Every other module routes its linear algebra through this file, so that the
regularisation policy exists in exactly ONE place and can be audited.

Contract summary
----------------
``get_block`` / ``set_block``   : (i, j) block views of a joint covariance whose
                                  entities are stacked in index order.
``symmetrize``                  : the ONLY way symmetry is restored; never write
                                  ``0.5 * (A + A.T)`` by hand elsewhere.
``safe_inverse``                : SPD inverse via Cholesky with an eigenvalue
                                  fallback; used by NEES and by the gains.
``invsqrt_psd``                 : the UNIQUE whitening entry point (M12).  It
                                  returns a SYMMETRIC inverse square root, so
                                  ``W = W.T`` and ``W A W`` is a congruence.
``psd_project``                 : clip negative eigenvalues, used after Joseph
                                  updates that drift by round-off.
"""

import numpy as np

_JITTER_STEPS = 8


# ----------------------------------------------------------------------
# block access
# ----------------------------------------------------------------------
def get_block(P, i, j, dim):
    """Return the (i, j) ``dim x dim`` block of a stacked joint matrix."""
    return np.asarray(P)[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim]


def set_block(P, i, j, dim, value):
    """Write the (i, j) ``dim x dim`` block in place and return ``P``."""
    P[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = value
    return P


def block_indices(i, dim):
    """Return the index array of entity ``i`` inside a stacked joint vector."""
    return np.arange(i * dim, (i + 1) * dim)


def gather_indices(indices, dim):
    """Concatenated index array of several entities (used by marginal NEES)."""
    idx = list(indices)
    if not idx:
        return np.zeros(0, dtype=int)
    return np.concatenate([block_indices(i, dim) for i in idx])


# ----------------------------------------------------------------------
# symmetry / positive definiteness
# ----------------------------------------------------------------------
def symmetrize(A):
    """Return the symmetric part of ``A`` (the single sanctioned recipe)."""
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def is_psd(A, tol=0.0):
    """True when ``A`` is symmetric and has no eigenvalue below ``-tol``."""
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        return False
    if not np.allclose(A, A.T, atol=1e-10, rtol=0.0):
        return False
    return bool(np.min(np.linalg.eigvalsh(A)) >= -abs(tol))


def psd_project(A, floor=0.0):
    """Symmetrise ``A`` and clip its eigenvalues from below at ``floor``."""
    A = symmetrize(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, floor, None)
    return symmetrize((V * w) @ V.T)


def chol_jitter(A, eps=1e-12):
    """Cholesky factor of ``A``, adding a growing jitter until it succeeds."""
    A = symmetrize(A)
    scale = float(np.max(np.abs(np.diag(A)))) if A.size else 1.0
    scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    jitter = float(eps) * scale
    for _ in range(_JITTER_STEPS):
        try:
            return np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.cholesky(psd_project(A, floor=jitter))


def safe_inverse(A, eps=1e-12):
    """Inverse of an SPD matrix via Cholesky, with an eigenvalue fallback."""
    A = symmetrize(A)
    n = A.shape[0]
    if n == 0:
        return A.copy()
    try:
        L = np.linalg.cholesky(A)
        Linv = np.linalg.solve(L, np.eye(n))
        return symmetrize(Linv.T @ Linv)
    except np.linalg.LinAlgError:
        pass
    w, V = np.linalg.eigh(A)
    scale = float(np.max(np.abs(w))) if w.size else 1.0
    floor = max(float(eps) * max(scale, 1e-300), 1e-300)
    w = np.clip(w, floor, None)
    return symmetrize((V / w) @ V.T)


def safe_solve(A, B, eps=1e-12):
    """Solve ``A X = B`` for SPD ``A`` without forming the explicit inverse."""
    A = symmetrize(A)
    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, B)
        return np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        return safe_inverse(A, eps) @ B


def invsqrt_psd(A, floor=1e-18):
    """Return the SYMMETRIC inverse square root ``A^{-1/2}`` of a PSD matrix.

    This is the unique whitening entry point of the project (M12 section 5.7).
    Because the result is symmetric, ``W A W`` is a genuine congruence, which is
    what makes the eta ratios exactly invariant under a change of units.

    Eigenvalues below ``floor`` (an ABSOLUTE floor, so the caller controls the
    scale) are clipped, which keeps the map total on singular inputs.
    """
    A = symmetrize(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, max(float(floor), 1e-300), None)
    return symmetrize((V / np.sqrt(w)) @ V.T)


def sqrtm_psd(A, floor=0.0):
    """Symmetric square root of a PSD matrix (used by sampling helpers)."""
    A = symmetrize(A)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, max(float(floor), 0.0), None)
    return symmetrize((V * np.sqrt(w)) @ V.T)


def log_det_psd(A, eps=1e-12):
    """Log-determinant of an SPD matrix (CI 'det' objective of M17)."""
    L = chol_jitter(A, eps)
    return float(2.0 * np.sum(np.log(np.diag(L))))


def block_diag(matrices):
    """Dense block-diagonal assembly of a list of square matrices."""
    mats = [np.asarray(M, dtype=float) for M in matrices]
    if not mats:
        return np.zeros((0, 0))
    n = sum(M.shape[0] for M in mats)
    out = np.zeros((n, n))
    k = 0
    for M in mats:
        d = M.shape[0]
        out[k:k + d, k:k + d] = M
        k += d
    return out


def joseph_update(P, h, r, gain=None):
    """Joseph-form covariance update for ONE scalar measurement.

    ``P`` is the prior covariance, ``h`` the (n,) measurement row and ``r`` the
    scalar measurement variance.  With the exact Kalman gain this is
    algebraically identical to the short form ``P - k h P``; the reason the
    design ([M1]) insists on keeping it is that the two differ NUMERICALLY --
    the Joseph form stays symmetric positive semi-definite when the gain is
    perturbed (round-off, a stale gain, a deliberately suboptimal one), while
    the short form can and does go indefinite.  ``joseph_form`` in config
    selects between them, so both must exist for that ablation to mean anything.

        P+ = (I - k h) P (I - k h)^T + k r k^T
    """
    P = np.asarray(P, dtype=float)
    h = np.asarray(h, dtype=float).ravel()
    r = float(r)
    S = float(h @ P @ h) + r
    if not np.isfinite(S) or S <= 0.0:
        raise np.linalg.LinAlgError(f"non-positive innovation variance S={S}")
    k = (P @ h) / S if gain is None else np.asarray(gain, dtype=float).ravel()
    A = np.eye(P.shape[0]) - np.outer(k, h)
    return symmetrize(A @ P @ A.T + r * np.outer(k, k))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- block accessors round-trip ----
    P = np.zeros((12, 12))
    B = rng.standard_normal((3, 3))
    set_block(P, 1, 2, 3, B)
    assert np.allclose(get_block(P, 1, 2, 3), B)
    assert np.array_equal(block_indices(2, 3), np.array([6, 7, 8]))
    assert np.array_equal(gather_indices([0, 2], 3), np.array([0, 1, 2, 6, 7, 8]))
    assert gather_indices([], 3).size == 0

    # ---- symmetrize / psd ----
    A = rng.standard_normal((5, 5))
    assert np.allclose(symmetrize(A), symmetrize(A).T)
    S = A @ A.T + 1e-3 * np.eye(5)
    assert is_psd(S)
    assert not is_psd(A)
    bad = S.copy(); bad[0, 0] = -1.0
    assert np.min(np.linalg.eigvalsh(psd_project(bad, 0.0))) >= -1e-12

    # ---- safe_inverse matches the exact inverse on a well-conditioned SPD ----
    Sinv = safe_inverse(S)
    assert np.allclose(Sinv @ S, np.eye(5), atol=1e-8)
    assert np.allclose(Sinv, Sinv.T)
    # ... and stays finite on a rank-deficient input
    R = rng.standard_normal((5, 2))
    singular = R @ R.T
    assert np.all(np.isfinite(safe_inverse(singular)))

    # ---- safe_solve agrees with safe_inverse ----
    rhs = rng.standard_normal((5, 3))
    assert np.allclose(safe_solve(S, rhs), Sinv @ rhs, atol=1e-8)

    # ---- invsqrt_psd: symmetric, and W S W = I ----
    W = invsqrt_psd(S)
    assert np.allclose(W, W.T, atol=1e-12)
    assert np.allclose(W @ S @ W, np.eye(5), atol=1e-8)
    assert np.allclose(W @ W, Sinv, atol=1e-8)
    assert np.all(np.isfinite(invsqrt_psd(singular)))

    # congruence property: for D diagonal, invsqrt(D S D) = D^-1 invsqrt(S) D^-1
    # only up to an orthogonal factor, but the whitened norm is invariant:
    D = np.diag(np.array([1.0, 10.0, 100.0, 0.1, 1.0]))
    M = rng.standard_normal((5, 5))
    W2 = invsqrt_psd(D @ S @ D)
    n1 = np.linalg.norm(W @ M @ W.T, "fro")
    n2 = np.linalg.norm(W2 @ (D @ M @ D) @ W2.T, "fro")
    assert abs(n1 - n2) < 1e-6 * max(1.0, n1), (n1, n2)

    # ---- sqrtm / logdet ----
    assert np.allclose(sqrtm_psd(S) @ sqrtm_psd(S), S, atol=1e-8)
    sign, logdet = np.linalg.slogdet(S)
    assert sign > 0 and abs(log_det_psd(S) - logdet) < 1e-8

    # ---- block_diag ----
    BD = block_diag([np.eye(2), 2.0 * np.eye(3)])
    assert BD.shape == (5, 5) and BD[2, 2] == 2.0 and BD[0, 3] == 0.0

    # ---- joseph_update: exact gain == short form, perturbed gain stays PSD ----
    n = 6
    M = rng.standard_normal((n, n))
    S6 = symmetrize(M @ M.T) + n * np.eye(n)
    h = rng.standard_normal(n)
    r = 0.04
    s = float((h @ S6 @ h.T).item()) + r
    k_opt = (S6 @ h) / s
    short = S6 - np.outer(k_opt, h @ S6)
    jos = joseph_update(S6, h, r)
    assert np.allclose(jos, short, atol=1e-10), float(np.max(np.abs(jos - short)))
    # a badly scaled gain: Joseph stays PSD, the short form does not
    k_bad = 2.5 * k_opt
    jos_bad = joseph_update(S6, h, r, gain=k_bad)
    short_bad = S6 - np.outer(k_bad, h @ S6)
    assert float(np.min(np.linalg.eigvalsh(jos_bad))) > -1e-12
    assert float(np.min(np.linalg.eigvalsh(symmetrize(short_bad)))) < -1e-9

    print("[M1 LinAlgUtils] OK  (invsqrt_psd symmetric, whitened norm invariant "
          "under congruence to %.1e)" % abs(n1 - n2))
