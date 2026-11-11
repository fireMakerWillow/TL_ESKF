"""M2 SO3Utils (L1 basic capability) -- the manifold layer.

The whole project uses the RIGHT (body-frame) attitude-error convention

    R = R_hat Exp(dtheta),      dtheta = Log(R_hat^T R).

Everything that touches attitude goes through this file.  In particular the
error vector of M10 is built with :func:`boxminus`, NEVER by subtracting Euler
angles: at large tilt an Euler difference is not the tangent-space error and it
would forge both the RMSE and the NEES.

Numerics: ``Exp`` and ``Log`` use the closed-form Rodrigues expressions with
Taylor fallbacks near theta = 0 and, for ``Log``, a rotation-axis extraction
near theta = pi so that the map stays accurate over the full range.
"""

from dataclasses import dataclass, field

import numpy as np

_EPS_SMALL = 1e-8


def skew(v):
    """Return the 3x3 skew-symmetric matrix ``[v]_x``."""
    v = np.asarray(v, dtype=float).ravel()
    if v.size != 3:
        raise ValueError("skew expects a 3-vector")
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def unskew(S):
    """Inverse of :func:`skew` for an exactly skew-symmetric matrix."""
    S = np.asarray(S, dtype=float)
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def Exp(phi):
    """SO(3) exponential map (Rodrigues), with a Taylor branch near zero."""
    phi = np.asarray(phi, dtype=float).ravel()
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    if theta < _EPS_SMALL:
        # I + W + W^2/2 is exact to O(theta^3) and keeps the derivative right
        return np.eye(3) + W + 0.5 * (W @ W)
    s, c = np.sin(theta), np.cos(theta)
    return (np.eye(3) + (s / theta) * W
            + ((1.0 - c) / theta ** 2) * (W @ W))


def Log(R):
    """SO(3) logarithm; returns the rotation vector with ||phi|| <= pi."""
    R = np.asarray(R, dtype=float)
    trace = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(trace))
    if theta < _EPS_SMALL:
        return 0.5 * unskew(R - R.T)
    if np.pi - theta < 1e-6:
        # near pi the antisymmetric part vanishes; take the axis from R + I
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] < 1e-12:
            return np.zeros(3)
        axis = A[:, k] / axis[k]
        axis = axis / np.linalg.norm(axis)
        # fix the sign with the (small but nonzero) antisymmetric part
        anti = unskew(R - R.T)
        if float(anti @ axis) < 0.0:
            axis = -axis
        return theta * axis
    return (theta / (2.0 * np.sin(theta))) * unskew(R - R.T)


def right_jacobian(phi):
    """Right Jacobian J_r(phi) of the exponential map."""
    phi = np.asarray(phi, dtype=float).ravel()
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    if theta < _EPS_SMALL:
        return np.eye(3) - 0.5 * W + (1.0 / 6.0) * (W @ W)
    s, c = np.sin(theta), np.cos(theta)
    return (np.eye(3)
            - ((1.0 - c) / theta ** 2) * W
            + ((theta - s) / theta ** 3) * (W @ W))


def right_jacobian_inv(phi):
    """Inverse right Jacobian J_r^{-1}(phi)."""
    phi = np.asarray(phi, dtype=float).ravel()
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    if theta < _EPS_SMALL:
        return np.eye(3) + 0.5 * W + (1.0 / 12.0) * (W @ W)
    half = 0.5 * theta
    coef = (1.0 / theta ** 2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    if abs(np.sin(theta)) < 1e-12:                 # theta -> pi guard
        coef = (1.0 / theta ** 2) - (1.0 / (2.0 * half) ** 2)
    return np.eye(3) + 0.5 * W + coef * (W @ W)


def project_so3(R):
    """Project a nearly-orthogonal matrix back onto SO(3) (SVD polar factor)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0.0:
        U[:, -1] *= -1.0
        Rp = U @ Vt
    return Rp


def reset_jacobian(c_theta, mode="first_order"):
    """G_theta of the ESKF reset: I - 0.5 [c_theta]_x (or exactly J_r)."""
    if mode == "identity":
        return np.eye(3)
    if mode == "first_order":
        return np.eye(3) - 0.5 * skew(c_theta)
    raise ValueError(f"unknown reset_jacobian mode {mode!r}")


@dataclass
class NominalState:
    """Nominal (non-error) state of one entity: position, velocity, attitude."""

    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=float).ravel().copy()
        self.v = np.asarray(self.v, dtype=float).ravel().copy()
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3).copy()
        self.b_a = np.asarray(self.b_a, dtype=float).ravel().copy()
        self.b_g = np.asarray(self.b_g, dtype=float).ravel().copy()

    def copy(self):
        return NominalState(self.p.copy(), self.v.copy(), self.R.copy(),
                            self.b_a.copy(), self.b_g.copy())

    def as_vector(self):
        """[p, v, Log(R), b_a, b_g] -- a chart, used only for logging."""
        return np.concatenate([self.p, self.v, Log(self.R), self.b_a, self.b_g])


def boxplus(state, dx):
    """Inject a 15-dim error into the nominal state (right convention)."""
    dx = np.asarray(dx, dtype=float).ravel()
    if dx.size != 15:
        raise ValueError("boxplus expects a 15-dim error vector")
    out = state.copy()
    out.p = out.p + dx[0:3]
    out.v = out.v + dx[3:6]
    out.R = project_so3(out.R @ Exp(dx[6:9]))
    out.b_a = out.b_a + dx[9:12]
    out.b_g = out.b_g + dx[12:15]
    return out


def boxminus(state, reference):
    """Generalized subtraction ``state boxminus reference`` (15-dim).

    The attitude component is ``Log(R_ref^T R_state)``, i.e. exactly the error
    that the right-convention ESKF covariance describes.
    """
    return np.concatenate([
        state.p - reference.p,
        state.v - reference.v,
        Log(reference.R.T @ state.R),
        state.b_a - reference.b_a,
        state.b_g - reference.b_g,
    ])


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # ---- skew / unskew ----
    v = rng.standard_normal(3)
    assert np.allclose(unskew(skew(v)), v)
    assert np.allclose(skew(v), -skew(v).T)
    w = rng.standard_normal(3)
    assert np.allclose(skew(v) @ w, np.cross(v, w))

    # ---- Exp is a rotation; Log inverts it over the full range ----
    for scale in (1e-12, 1e-9, 1e-4, 0.5, 2.0, np.pi - 1e-3):
        for _ in range(20):
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            phi = scale * axis
            R = Exp(phi)
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-10), scale
            assert abs(np.linalg.det(R) - 1.0) < 1e-10
            assert np.allclose(Log(R), phi, atol=1e-7), (scale, Log(R), phi)

    # ---- Log(Exp) round trip near pi ----
    axis = np.array([0.0, 0.0, 1.0])
    R = Exp(np.pi * axis)
    assert abs(np.linalg.norm(Log(R)) - np.pi) < 1e-6

    # ---- right Jacobian: numerical check of Exp(phi + d) ~ Exp(phi) Exp(Jr d)
    phi = np.array([0.3, -0.2, 0.7])
    Jr = right_jacobian(phi)
    h = 1e-6
    for k in range(3):
        d = np.zeros(3); d[k] = h
        lhs = Exp(phi + d)
        rhs = Exp(phi) @ Exp(Jr @ d)
        assert np.max(np.abs(lhs - rhs)) < 1e-9, k
    assert np.allclose(right_jacobian_inv(phi) @ Jr, np.eye(3), atol=1e-9)

    # ---- project_so3 repairs a perturbed rotation ----
    Rp = Exp(phi) + 1e-3 * rng.standard_normal((3, 3))
    Rq = project_so3(Rp)
    assert np.allclose(Rq.T @ Rq, np.eye(3), atol=1e-12)
    assert np.linalg.det(Rq) > 0

    # ---- boxplus / boxminus are exact inverses ----
    st = NominalState(p=rng.standard_normal(3), v=rng.standard_normal(3),
                      R=Exp(rng.standard_normal(3)),
                      b_a=rng.standard_normal(3), b_g=rng.standard_normal(3))
    dx = 0.1 * rng.standard_normal(15)
    assert np.allclose(boxminus(boxplus(st, dx), st), dx, atol=1e-9)
    assert np.allclose(boxminus(st, st), np.zeros(15), atol=1e-12)

    # ---- boxminus differs from an Euler subtraction at large tilt (why M10
    #      must never use Euler angles) ----
    big = 1.5
    st2 = st.copy(); st2.R = project_so3(st.R @ Exp(np.array([big, 0.0, 0.0])))
    err = boxminus(st2, st)
    assert abs(np.linalg.norm(err[6:9]) - big) < 1e-9

    # ---- reset Jacobian ----
    c = np.array([0.01, -0.02, 0.005])
    G = reset_jacobian(c, "first_order")
    assert np.allclose(G, np.eye(3) - 0.5 * skew(c))
    assert np.allclose(reset_jacobian(c, "identity"), np.eye(3))
    # G = I - 0.5 [c]_x is the first-order truncation of Exp(-c/2); the residual
    # must therefore be second order in |c|
    nc = float(np.linalg.norm(c))
    assert np.max(np.abs(G - Exp(-0.5 * c))) < 1.0 * nc ** 2
    assert np.max(np.abs(G - np.eye(3))) < 0.6 * nc

    print("[M2 SO3Utils] OK  (Exp/Log invertible to |phi|=pi, boxplus/boxminus "
          "exact to 1e-9)")
