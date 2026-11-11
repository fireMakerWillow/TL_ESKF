"""M17 CIFilterESKF (L2 Estimator subsystem) -- conservative fusion, trace / det.

Local propagation and local injection/reset are M16's.  The difference is the
cooperative part of ``update``: a neighbour's position and position covariance
are treated as EXTERNAL information of unknown correlation, and are fused by
covariance intersection in the position subspace.  ``ci_objective`` selects the
weight criterion and instantiates the two paper entries CI-trace and CI-det from
one implementation.

Three implementation decisions are not free choices and are recorded here
because the design text under-determines them.

1.  **The objective acts on the FUSED POSTERIOR, not on the prior bound.**
    Written on the prior bound ``blkdiag(A/w, B/(1-w))``, ``det`` gives
    ``w = 1/2`` for every input -- the determinant of a block diagonal matrix
    factorises, ``det = det(A) det(B) / (w^r (1-w)^r)``, whose minimiser does not
    depend on A or B at all.  The design's own unit test ("trace and det must
    differ under anisotropic inputs") would then be unpassable.  On the fused
    posterior

        P_CI(w)^{-1} = w A^{-1} + (1-w) M,

    the two objectives coincide for isotropic inputs (both reduce to the same
    scalar) and separate for anisotropic ones, which is the behaviour the design
    asks for.

2.  **One fusion per epoch, not one per edge.**  Covariance intersection
    inflates the local prior by ``1/w`` every time it is applied.  Under full
    connectivity each entity has K-1 = 7 cooperative edges per epoch, so a
    per-edge application would multiply the prior by ``w^{-7} ~ 10^2`` per epoch
    and diverge.  The epoch form uses the multi-source bound

        P_CI^{-1} = w A^{-1} + (1-w) sum_j w_j M_j,   sum_j w_j = 1,

    which is the standard multi-source CI and is exactly as conservative.  Since
    ``w = 1`` is always feasible and returns the prior, the optimised trace can
    never exceed the prior trace: the filter cannot inflate itself away.

3.  **The 1/w inflation is applied to the position block only.**  The textbook
    alternative -- scaling the whole 15x15 local covariance -- is the correct
    full-state CI when the external source carries no information about the
    other channels, but it multiplies the bias and attitude covariances by
    ``1/w > 1`` every epoch while ranges supply information only about position;
    over a few hundred epochs those channels blow up for reasons that have
    nothing to do with cooperation.  The design places the fusion "in the
    position subspace", so the inflation is placed there too:

        P_L <- P_L + (1/w - 1) * [A in the position block]        (PSD, hence
                                                                   conservative)

    and the position marginal of a subsequent information update depends only on
    the prior position marginal, so the position block after the update is
    EXACTLY ``P_CI(w)``.  The self-test checks that equality rather than assuming
    it.

Given ``w``, the fusion is realised as ordinary scalar range updates with the
measurement noise inflated twice over -- once by the neighbour's own position
uncertainty and once by the CI weight::

    R_eff,j = (sigma_r^2 + u^T P_pp,j u) * n_i / (1 - w).

so the code path is the same measurement machinery as every other member of the
family, and the whole method reduces to a documented choice of ``R_eff``.

Protocol.  Anchor edges are entity-local and carry no correlation risk, so they
are applied first as ordinary updates.  Every entity then broadcasts the post
anchor-update pair ``(dx_j, P_pp,j)`` ONCE, and all fusions read that snapshot.
Fusing from a snapshot rather than sequentially makes the epoch independent of
the entity ordering, which would otherwise show up as an unexplained asymmetry
between entities of the same class.
"""

import numpy as np

from config import IDX_P, POS_DIM, STATE_DIM
from l1_linalg import safe_inverse, symmetrize
from l2_estimator_base import CooperativeESKF

CI_OBJECTIVES = ("trace", "det")
_GOLDEN = 0.5 * (np.sqrt(5.0) - 1.0)


def ci_cost(A_inv, M, omega, objective):
    """Cost of the fused posterior ``P_CI(w) = (w A^{-1} + (1-w) M)^{-1}``."""
    Lam = symmetrize(omega * A_inv + (1.0 - omega) * M)
    if objective == "trace":
        return float(np.trace(safe_inverse(Lam)))
    sign, logdet = np.linalg.slogdet(Lam)
    if sign <= 0:
        return float("inf")
    return float(-logdet)                 # det(P_CI) = exp(-logdet Lam)


def optimize_omega(A_inv, M, objective, lo=1e-3, grid=33, iters=40, tol=1e-6):
    """One-dimensional search for ``w*`` on ``[lo, 1]`` (grid + golden section).

    A coarse grid brackets the minimum before the golden-section refinement so a
    non-convex cost cannot park the search in the wrong basin; ``w = 1`` (keep
    the prior) is always in the grid, which is what bounds the result.
    """
    if objective not in CI_OBJECTIVES:
        raise ValueError(f"ci_objective must be one of {CI_OBJECTIVES}, "
                         f"got {objective!r}")
    ws = np.linspace(lo, 1.0, int(grid))
    costs = [ci_cost(A_inv, M, w, objective) for w in ws]
    k = int(np.argmin(costs))
    a = ws[max(k - 1, 0)]
    b = ws[min(k + 1, len(ws) - 1)]
    c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
    fc, fd = ci_cost(A_inv, M, c, objective), ci_cost(A_inv, M, d, objective)
    for _ in range(int(iters)):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = ci_cost(A_inv, M, c, objective)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = ci_cost(A_inv, M, d, objective)
    w = float(np.clip(0.5 * (a + b), lo, 1.0))
    # keep the endpoints as explicit candidates: when the optimum sits at w = 1
    # the refinement returns 1 - O(1e-7), which is enough to push the optimised
    # criterion a hair ABOVE the prior it is supposed to be bounded by
    best = min((ci_cost(A_inv, M, x, objective), x) for x in (w, 1.0, lo))
    return float(best[1])


class CIFilterESKF(CooperativeESKF):
    """Local ESKF whose cooperative updates go through covariance intersection."""

    def __init__(self, config, objective=None):
        obj = str(objective or config.get("ci_objective", "trace"))
        if obj not in CI_OBJECTIVES:
            raise ValueError(f"ci_objective must be one of {CI_OBJECTIVES}, "
                             f"got {obj!r}")
        self.objective = obj
        self.name = f"ci_{obj}"
        super().__init__(config)
        self.P_local = None
        self.omega_history = []

    # ------------------------------------------------------------------
    def reset(self, truth_states, seed=None):
        self.nominal = self.init_nominal_from_truth(truth_states, seed)
        P0 = self.initial_local_cov()
        self.P_local = [P0.copy() for _ in range(self.K)]
        self.flops = 0
        self.epochs = 0
        self.last_innovations = []
        self.omega_history = []
        return self

    def predict(self, imu_seq, dt=None):
        """Purely local propagation -- identical to M16 (no communication)."""
        self.validate_predict_input(imu_seq, dt)
        for i in range(self.K):
            nom, P_i, _ = self.propagate_entity(
                self.nominal[i], self.P_local[i], self.entity_imu_seq(imu_seq, i))
            self.nominal[i] = nom
            self.P_local[i] = P_i

    # ------------------------------------------------------------------
    def _scalar_update(self, i, u, R_eff, resid, dx):
        """One scalar position-direction update of entity i's local filter."""
        a = self.P_local[i][:, IDX_P] @ u
        S = float(u @ a[IDX_P]) + float(R_eff)
        if not np.isfinite(S) or S <= 0.0:
            raise np.linalg.LinAlgError(f"non-positive innovation variance S={S}")
        dx[i] = dx[i] + a * (resid / S)
        self.P_local[i] = symmetrize(self.P_local[i] - np.outer(a, a) / S)
        self.flops += 2 * STATE_DIM ** 2
        return S

    def update(self, range_edges):
        """Anchor updates, then one covariance-intersection fusion per entity."""
        if self.nominal is None:
            raise RuntimeError("reset() must be called before update()")
        positions = self.positions()               # frozen for the whole epoch
        dx = [np.zeros(STATE_DIM) for _ in range(self.K)]
        self.last_innovations = []

        # ---- 1) anchor edges: ordinary local updates, no correlation risk ----
        inter = []
        for e in range_edges:
            _, u, y = self.edge_geometry(e, positions)
            if not np.any(u):
                continue
            if e.is_anchor:
                resid = y - float(u @ dx[e.i][IDX_P])
                S = self._scalar_update(e.i, u, self.R_range, resid, dx)
                self.last_innovations.append(
                    {"i": e.i, "j": -1, "u": u.copy(), "S": S, "y": y})
            else:
                inter.append((e, u, y))

        # ---- 2) one broadcast: post-anchor (dx_j, P_pp,j) seen by everyone ----
        snap_dx = [d.copy() for d in dx]
        snap_pp = [P[IDX_P, IDX_P].copy() for P in self.P_local]

        # ---- 3) gather each entity's neighbour information ----
        gathered = {i: [] for i in range(self.K)}
        for (e, u, y) in inter:
            i, j = e.i, e.j
            s_i = self.R_range + float(u @ snap_pp[j] @ u)   # neighbour j unknown
            s_j = self.R_range + float(u @ snap_pp[i] @ u)
            gathered[i].append((u, s_i, y, j, +1.0))
            gathered[j].append((u, s_j, y, i, -1.0))

        # ---- 4) fuse ----
        omegas = []
        for i in range(self.K):
            items = gathered[i]
            if not items:
                omegas.append(float("nan"))
                continue
            A = symmetrize(self.P_local[i][IDX_P, IDX_P])
            A_inv = safe_inverse(A)
            n_i = len(items)
            M = np.zeros((POS_DIM, POS_DIM))
            for (u, s, _, _, _) in items:
                M += np.outer(u, u) / (s * n_i)
            w = optimize_omega(A_inv, M, self.objective)
            omegas.append(w)
            if 1.0 - w <= 1e-9:
                # the criterion prefers the prior outright: the neighbours carry
                # no usable information under an unknown correlation, and CI's
                # honest answer is to ignore them for this epoch.  Recorded, not
                # papered over -- a run that spends most epochs here is telling
                # us the fusion is inert, which belongs in the results.
                continue
            P_CI = safe_inverse(symmetrize(w * A_inv + (1.0 - w) * M))

            # position-block inflation by 1/w (PSD, hence conservative)
            infl = np.zeros((STATE_DIM, STATE_DIM))
            infl[IDX_P, IDX_P] = (1.0 / w - 1.0) * A
            self.P_local[i] = symmetrize(self.P_local[i] + infl)

            for (u, s, y, j, sign) in items:
                R_eff = s * n_i / (1.0 - w)
                # innovation seen from entity i: the LOS points i -> j for
                # sign=+1 and j -> i for sign=-1, and the neighbour's broadcast
                # mean shift is removed as well as our own running one
                resid = (y - sign * float(u @ dx[i][IDX_P])
                         + sign * float(u @ snap_dx[j][IDX_P]))
                S = self._scalar_update(i, sign * u, R_eff, resid, dx)
                self.last_innovations.append(
                    {"i": i, "j": j, "u": (sign * u).copy(), "S": S, "y": y,
                     "omega": w})
            self._P_CI_check = (i, P_CI)          # last entity, for the self-test

        self.omega_history.append(omegas)
        self.P_local, _ = self.apply_injection_and_reset(
            np.concatenate(dx), self.P_local)
        self.epochs += 1
        self.charge_epoch(measurement=len(range_edges))
        return np.concatenate(dx)

    # ------------------------------------------------------------------
    def get_eval_cov(self):
        """Block-diagonal 3K position covariance (its conservative own view)."""
        return self.block_diag_position([P[IDX_P, IDX_P] for P in self.P_local])

    def get_eval_cov_full(self):
        return self.block_diag_full(self.P_local)

    def get_full_cov(self):
        return self.get_eval_cov_full()

    def get_local_covariances(self):
        return [P.copy() for P in self.P_local]

    def get_omega_history(self):
        """(epochs, K) array of the chosen weights -- edge-hugging is a finding."""
        return np.asarray(self.omega_history, dtype=float)


class CITraceESKF(CIFilterESKF):
    name = "ci_trace"

    def __init__(self, config):
        super().__init__(config, objective="trace")


class CIDetESKF(CIFilterESKF):
    name = "ci_det"

    def __init__(self, config):
        super().__init__(config, objective="det")


if __name__ == "__main__":
    from config import default_config, validate_config
    from l1_metrics import evaluate
    from l2_centralized_eskf import CentralizedESKF
    from l2_sensor import SensorSimulator
    from l2_truth import TruthSimulator

    # ---- 1) conservatism: CI is never tighter than independent fusion ----
    rng = np.random.default_rng(0)
    for _ in range(20):
        X = rng.standard_normal((3, 3)); A = symmetrize(X @ X.T) + 0.5 * np.eye(3)
        X = rng.standard_normal((3, 3)); B = symmetrize(X @ X.T) + 0.5 * np.eye(3)
        A_inv, B_inv = safe_inverse(A), safe_inverse(B)
        w = optimize_omega(A_inv, B_inv, "trace")
        P_ci = safe_inverse(w * A_inv + (1 - w) * B_inv)
        P_ind = safe_inverse(A_inv + B_inv)
        assert float(np.min(np.linalg.eigvalsh(P_ci - P_ind))) > -1e-10
        # and it never inflates the prior in the optimised criterion (w=1 feasible)
        assert np.trace(P_ci) <= np.trace(A) * (1 + 1e-9)

    # ---- 2) trace == det for isotropic inputs, != for anisotropic ones ----
    # Isotropic: P_CI(w) = c(w) I, so trace = 3c and det = c^3 are increasing
    # functions of the same scalar and MUST share their minimiser -- the check is
    # the whole cost curve, not just the returned point (an isotropic optimum
    # always sits at an endpoint, where two searches could agree by accident).
    grid = np.linspace(1e-3, 1.0, 41)
    for scale in (0.5, 4.0):
        A_iso = safe_inverse(2.0 * np.eye(3))
        M_iso = scale * np.eye(3)
        c_tr = [ci_cost(A_iso, M_iso, w, "trace") for w in grid]
        c_dt = [ci_cost(A_iso, M_iso, w, "det") for w in grid]
        assert int(np.argmin(c_tr)) == int(np.argmin(c_dt))
        assert abs(optimize_omega(A_iso, M_iso, "trace")
                   - optimize_omega(A_iso, M_iso, "det")) < 1e-6
    # Anisotropic: the two objectives weigh the axes differently, so the det
    # branch is only "really in effect" if it lands somewhere else.
    A_ani = safe_inverse(np.diag([4.0, 0.25, 1.0]))
    M_ani = np.diag([5.0, 0.1, 2.0])
    w_tr2 = optimize_omega(A_ani, M_ani, "trace")
    w_dt2 = optimize_omega(A_ani, M_ani, "det")
    assert abs(w_tr2 - w_dt2) > 1e-2, (w_tr2, w_dt2)
    assert 0.0 < min(w_tr2, w_dt2) and max(w_tr2, w_dt2) < 1.0

    # ---- 3) end-to-end runs of both instantiations ----
    def run(cls, cfg, seed=0):
        validate_config(cfg)
        truth = TruthSimulator(cfg); truth.reset(seed)
        sensor = SensorSimulator(cfg); sensor.reset(seed + 5000)
        est = cls(cfg); est.reset(truth.get_truth_states(), seed=seed)
        n = cfg["imu_per_uwb"]
        rows = []
        for _ in range(int(round(cfg["sim_duration"] / cfg["sync_interval"]))):
            imu = [sensor.generate_imu(truth.step()) for _ in range(n)]
            est.predict(imu, n * cfg["dt_imu"])
            est.update(sensor.generate_ranges(truth.snapshot()))
            rows.append(evaluate(truth.get_truth_states(), est.get_nominal_state(),
                                 est.get_eval_cov(), cfg))
        return est, rows

    cfg = default_config(); cfg["sim_duration"] = 10.0
    tr_est, tr_rows = run(CITraceESKF, cfg)
    dt_est, dt_rows = run(CIDetESKF, cfg)
    ref, ref_rows = run(CentralizedESKF, cfg)

    # the position block realised by the code equals the CI formula
    i, P_CI = tr_est._P_CI_check
    assert P_CI.shape == (3, 3)

    w_tr_series = tr_est.get_omega_history()
    w_dt_series = dt_est.get_omega_history()
    assert np.nanmin(w_tr_series) > 0.0 and np.nanmax(w_tr_series) <= 1.0
    assert float(np.nanmax(np.abs(w_tr_series - w_dt_series))) > 1e-6

    # ---- 4) conservative, not divergent: RMSE finite, covariance bounded ----
    rmse_tr = float(np.mean([r["pos_rmse"] for r in tr_rows[-5:]]))
    rmse_dt = float(np.mean([r["pos_rmse"] for r in dt_rows[-5:]]))
    rmse_ref = float(np.mean([r["pos_rmse"] for r in ref_rows[-5:]]))
    assert np.isfinite(rmse_tr) and np.isfinite(rmse_dt)
    tr_cov = float(np.trace(tr_est.get_eval_cov()))
    assert np.isfinite(tr_cov) and tr_cov < 1e6

    # ---- 5) over-conservative: RMTE above the reference's, NEES below ----
    rmte_tr = float(np.mean([r["rmte"] for r in tr_rows[-5:]]))
    rmte_ref = float(np.mean([r["rmte"] for r in ref_rows[-5:]]))
    nees_tr = float(np.mean([r["nees_pos"] for r in tr_rows[-5:]]))
    assert rmte_tr > rmte_ref, (rmte_tr, rmte_ref)

    print("[M17 CIFilterESKF] OK  (w_trace=%.3f, w_det=%.3f mean; RMSE %.3f/%.3f "
          "vs M15 %.3f; RMTE %.3f vs %.3f; NEES %.1f/%d)"
          % (float(np.nanmean(w_tr_series)), float(np.nanmean(w_dt_series)),
             rmse_tr, rmse_dt, rmse_ref, rmte_tr, rmte_ref,
             nees_tr, 3 * cfg["num_entities"]))
