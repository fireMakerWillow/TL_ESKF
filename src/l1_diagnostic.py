"""M12 StructuralResidualDiagnostic (L1 evaluation kernel) -- REWRITTEN for v3.

What the dual layer throws away, measured directly.  Let P_x be the centralized
full joint covariance (the gold standard) and P_G the dual layer's position
layer.  For a pair i != j the structural truncation error is

    E^ij = P_x^ij - L_i P_G^ij L_j^T = [[0, E_pz], [E_zp, E_zz]]

whose position-position block vanishes BY CONSTRUCTION, because the first three
rows of the regression lift are the identity.  ``E_pp = 0`` and ``H E^ij H^T =
0`` are therefore implementation assertions, not experimental findings: if they
fail, the block layout of L or the choice of H is wrong.

The quantity that actually costs accuracy is not E itself but what E becomes
after one interval of dynamics, because the position layer re-imports the
propagated cross block:

    dP_G^ij = H Phibar_i E^ij Phibar_j^T H^T

and through it the innovation variance of every cooperative range is wrong by

    dS = -u^T (dP_G^ij + dP_G^ij^T) u.

Which P_G goes into E (a point the design leaves implicit)
----------------------------------------------------------
``E_pp = 0`` is asserted as a CONSTRUCTION identity, and it is one only when the
position cross block subtracted off is the position marginal of P_x itself:

    E_pp = P_x,pp^ij - P_G^ij,      so E_pp = 0  <=>  P_G^ij = H P_x^ij H^T.

So the structural residual is defined with the PROJECTION of the gold standard,
``P_G^ij := H P_x^ij H^T``.  It then measures exactly what it is supposed to
measure -- the cost of REPRESENTING a full 15x15 cross block by its 3x3 position
part lifted through L -- and E_pp vanishes to machine precision whatever the
estimators are doing (``mode='structural'``, the main line).

Feeding M19's own P_G in instead mixes that representation loss with a second,
unrelated quantity: how far the dual-layer filter has drifted from the
centralized one by this epoch.  That drift is worth recording, so it is reported
separately as ``pos_layer_dev`` (v2's D1), and ``mode='realised'`` is available
for ablation -- but in that mode E_pp is NOT zero and the identity assertion is
correctly switched off.

Why everything relative is computed WHITENED
--------------------------------------------
A 15x15 covariance block mixes m^2, m^2/s, m*rad, m^2/s^2 ...  Taking a
Frobenius norm of it adds quantities that cannot be added, and the number then
depends on whether lengths are in metres or centimetres.  Every relative eta
below is therefore formed in the whitened space

    W_i = (P_x^ii + lambda diag(P_x^ii))^{-1/2},   Ebar = W_i E W_j^T

which makes each state contribute in units of its own standard deviation.  Note
the regulariser is ``lambda diag(P)``, NOT ``lambda I``: an additive ``lambda I``
is itself dimensional, so it would re-break exactly the unit invariance the
whitening is there to buy (test T17 fails with ``lambda I`` and passes with
``lambda diag``).  With a congruent regulariser the whitened norms are exactly
invariant, since ``||W_i M W_j^T||_F^2 = tr(A_i^{-1} M A_j^{-1} M^T)`` and both
A's transform congruently.

For the same reason ``eta_dyn``'s floor is ``tau * sqrt(||.||_ii ||.||_jj)``
rather than a bare ``tau``: the numerator and denominator there carry m^2, so an
absolute floor would leak units into the ratio.

The four metrics
----------------
    eta_struct  ||Ebar||_F / (||Pbar_x^ij||_F + tau)   full-state structure loss
    eta_abs     ||Ebar||_2                             absolute, safe when the
                                                       true cross block ~ 0
    eta_dyn     ||dP_G^ij||_F / (||H Phibar P_x Phibar^T H^T||_F + floor)
                                                       the RQ2 quantity
    eta_S       |dS| / S                               what the filter feels

eta_S is reported separately because S itself shrinks when two entities are
strongly correlated, so a small eta_dyn can still produce a large eta_S.

Four-channel decomposition.  With ``H Phibar_i = [F_p | F_vel F_att F_ba F_bg]``
from M7, dP_G expands as the bilinear form

    dP_G = sum_{a,b} F_a,i E_ab F_b,j^T,     a, b in {p, vel, att, ba, bg}

with the (p, p) term absent because E_pp = 0.  The share attributed to channel c
is the sum of the Frobenius norms of every term touching c (counting the (c, c)
term once).  Norms of terms are used rather than a norm of the summed terms so
that the shares are additive and cannot cancel; the exactness of the expansion
itself is asserted separately.  The analytic orders to compare against are

    vel O(T),  att O(|f| T^2),  acc bias O(T^2),  gyro bias O(|f| T^3).

A note on the ``att`` order.  The plan writes O(a_c T^2) with a_c = v^2/R, but
the coupling block is ``-0.5 R [f^b]_x T^2`` and f^b = a^n - g, so gravity sits
inside it and dominates whenever a_c << |g| (see ``config.coupling_prediction_
full``).  Both curves are returned, and the v-sensitive part is the EXCESS over
the hover baseline.
"""

import numpy as np

from config import (POS_DIM, STATE_DIM, Z_CHANNELS, Z_CHANNEL_SLICES,
                    IDX_P, IDX_TH, coupling_prediction, coupling_prediction_full,
                    coupling_excess, specific_force_magnitude)
from l1_linalg import get_block, invsqrt_psd, symmetrize
from l1_process_model import split_position_row
from l1_regression import lift_matrix

_H = np.zeros((POS_DIM, STATE_DIM))
_H[:, IDX_P] = np.eye(POS_DIM)

# state-space slices of the five channels, position first
CHANNEL_SLICES = {"pos": IDX_P}
for _name in Z_CHANNELS:
    _sl = Z_CHANNEL_SLICES[_name]
    CHANNEL_SLICES[_name] = slice(POS_DIM + _sl.start, POS_DIM + _sl.stop)
ALL_CHANNELS = ("pos",) + tuple(Z_CHANNELS)


# ----------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------
def position_selector():
    """The 3x15 selector H used throughout this module."""
    return _H.copy()


def whiten_matrix(P_ii, lam):
    """W = (P + lam diag(P))^{-1/2} -- the congruent whitener (see docstring)."""
    P_ii = symmetrize(np.asarray(P_ii, dtype=float))
    d = np.clip(np.diag(P_ii), 0.0, None)
    scale = float(np.max(d)) if d.size else 1.0
    scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    reg = float(lam) * np.diag(np.where(d > 0.0, d, scale))
    floor = 1e-18 * max(scale, 1e-300) ** 1
    return invsqrt_psd(P_ii + reg, floor=floor)


def structural_residual(P_x_ij, P_G_ij, L_i, L_j):
    """E^ij = P_x^ij - L_i P_G^ij L_j^T."""
    return (np.asarray(P_x_ij, dtype=float)
            - np.asarray(L_i, dtype=float) @ np.asarray(P_G_ij, dtype=float)
            @ np.asarray(L_j, dtype=float).T)


def dynamic_residual(E_ij, Phi_bar_i, Phi_bar_j):
    """dP_G^ij = H Phibar_i E^ij Phibar_j^T H^T (3x3)."""
    return (_H @ np.asarray(Phi_bar_i, dtype=float) @ np.asarray(E_ij, dtype=float)
            @ np.asarray(Phi_bar_j, dtype=float).T @ _H.T)


def innovation_variance_error(dP_G, u):
    """dS = -u^T (dP_G + dP_G^T) u for one inter-entity range."""
    u = np.asarray(u, dtype=float).ravel()
    dP_G = np.asarray(dP_G, dtype=float)
    return float(-u @ (dP_G + dP_G.T) @ u)


def channel_decomposition(E_ij, Phi_bar_i, Phi_bar_j):
    """Bilinear expansion of dP_G into the five state channels.

    Returns ``(terms, shares, total)`` where ``terms[a][b]`` is the Frobenius
    norm of ``F_a,i E_ab F_b,j^T``, ``shares[c]`` is the normalised weight of
    channel c, and ``total`` is the reassembled dP_G (identical to
    :func:`dynamic_residual` up to round-off).
    """
    E_ij = np.asarray(E_ij, dtype=float)
    Fi = split_position_row(Phi_bar_i)
    Fj = split_position_row(Phi_bar_j)
    blocks_i = {"pos": Fi["F_p"], **{c: Fi[c] for c in Z_CHANNELS}}
    blocks_j = {"pos": Fj["F_p"], **{c: Fj[c] for c in Z_CHANNELS}}

    terms = {a: {} for a in ALL_CHANNELS}
    total = np.zeros((POS_DIM, POS_DIM))
    norms = np.zeros((len(ALL_CHANNELS), len(ALL_CHANNELS)))
    for ia, a in enumerate(ALL_CHANNELS):
        for ib, b in enumerate(ALL_CHANNELS):
            block = E_ij[CHANNEL_SLICES[a], CHANNEL_SLICES[b]]
            term = blocks_i[a] @ block @ blocks_j[b].T
            total += term
            n = float(np.linalg.norm(term, "fro"))
            terms[a][b] = n
            norms[ia, ib] = n

    weights = {}
    for ia, c in enumerate(ALL_CHANNELS):
        weights[c] = float(norms[ia, :].sum() + norms[:, ia].sum() - norms[ia, ia])
    mass = sum(weights.values())
    shares = {c: (weights[c] / mass if mass > 0 else float("nan"))
              for c in ALL_CHANNELS}
    return {"term_norms": terms, "matrix": norms, "weights": weights,
            "shares": shares, "total": total}


def analytic_orders(config, T=None):
    """Analytic magnitudes of the four non-position channels of H Phibar.

    ``vel ~ T``, ``att ~ |f| T^2 / sqrt(2)``-scaled, ``acc_bias ~ T^2/2``,
    ``gyro_bias ~ |f| T^3/6``: the reference curve each channel's measured share
    is checked against in M22.
    """
    T = float(config["sync_interval"] if T is None else T)
    f = specific_force_magnitude(config)
    return {
        "vel": T,
        "att": float(np.sqrt(2.0) * 0.5 * f * T ** 2),
        "acc_bias": 0.5 * T ** 2,
        "gyro_bias": float(np.sqrt(2.0) * f * T ** 3 / 6.0),
        "att_plan_curve": coupling_prediction(config, T),
        "att_full_curve": coupling_prediction_full(config, T),
        "att_excess_curve": coupling_excess(config, T),
    }


def coupling_scalar(Phi_bar):
    """kappa_i = ||Phibar_i[0:3, 6:9]||_F -- the measured coupling strength."""
    return float(np.linalg.norm(np.asarray(Phi_bar, dtype=float)[IDX_P, IDX_TH],
                                "fro"))


# ----------------------------------------------------------------------
# reference implementation (the appendix cross-check target, T18)
# ----------------------------------------------------------------------
def cross_structure_metrics(P_x, P_G=None, K=None, lam=1e-6, tau=1e-12, eps=1e-9,
                            Phi_bar_list=None, mode="structural"):
    """Straight-line reference implementation of the eta family.

    Deliberately written as a plain double loop that recomputes L and W inside
    the loop, so that agreeing with :class:`StructuralResidualDiagnostic` (which
    caches both and vectorises) is real evidence and not a tautology.  This is
    the function the paper's appendix listing is checked against (T18).
    """
    P_x = np.asarray(P_x, dtype=float)
    K = int(P_x.shape[0] // STATE_DIM) if K is None else int(K)
    P_G = None if P_G is None else np.asarray(P_G, dtype=float)
    out = {}
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            P_ii = get_block(P_x, i, i, STATE_DIM)
            P_jj = get_block(P_x, j, j, STATE_DIM)
            P_ij = get_block(P_x, i, j, STATE_DIM)
            L_i = lift_matrix(P_ii, eps)
            L_j = lift_matrix(P_jj, eps)
            if mode == "realised" and P_G is not None:
                P_G_ij = get_block(P_G, i, j, POS_DIM)
            else:
                P_G_ij = _H @ P_ij @ _H.T
            E = P_ij - L_i @ P_G_ij @ L_j.T
            W_i = whiten_matrix(P_ii, lam)
            W_j = whiten_matrix(P_jj, lam)
            Ebar = W_i @ E @ W_j.T
            Pbar = W_i @ P_ij @ W_j.T
            entry = {
                "eta_struct": float(np.linalg.norm(Ebar, "fro")
                                    / (np.linalg.norm(Pbar, "fro") + tau)),
                "eta_abs": float(np.linalg.norm(Ebar, 2)),
                "E_pp_norm": float(np.linalg.norm(E[IDX_P, IDX_P], "fro")),
            }
            if Phi_bar_list is not None:
                Fi = np.asarray(Phi_bar_list[i], dtype=float)
                Fj = np.asarray(Phi_bar_list[j], dtype=float)
                num = _H @ Fi @ E @ Fj.T @ _H.T
                den = _H @ Fi @ P_ij @ Fj.T @ _H.T
                ref = np.sqrt(
                    np.linalg.norm(_H @ Fi @ P_ii @ Fi.T @ _H.T, "fro")
                    * np.linalg.norm(_H @ Fj @ P_jj @ Fj.T @ _H.T, "fro"))
                entry["eta_dyn"] = float(np.linalg.norm(num, "fro")
                                         / (np.linalg.norm(den, "fro") + tau * ref))
                entry["dP_G"] = num
            out[(i, j)] = entry
    return out


# ----------------------------------------------------------------------
# the object M20 / M22 use
# ----------------------------------------------------------------------
class StructuralResidualDiagnostic:
    """Whitened structural diagnostics of the dual-layer truncation."""

    def __init__(self, config, mode="structural"):
        if mode not in ("structural", "realised"):
            raise ValueError(f"unknown diagnostic mode {mode!r}")
        self.cfg = config
        self.mode = mode
        self.K = int(config["num_entities"])
        self.lam = float(config["whiten_lambda"])
        self.tau = float(config["diag_tau"])
        self.eps = float(config["reg_epsilon"])
        self.sigma_r = float(config["sigma_range"])
        self.assert_tol = 1e-8

    # -- helpers -------------------------------------------------------
    def _lifts_and_whiteners(self, P_x):
        """L_i and W_i built ON SITE from P_x^ii.

        The lift is deliberately NOT taken from M19's internal cache: the
        diagnostic has to measure the structural gap, not the implementation
        detail of how one estimator happened to regularise its own lift.
        """
        L, W, diag = [], [], []
        for i in range(self.K):
            P_ii = symmetrize(get_block(P_x, i, i, STATE_DIM))
            diag.append(P_ii)
            L.append(lift_matrix(P_ii, self.eps))
            W.append(whiten_matrix(P_ii, self.lam))
        return L, W, diag

    def _check_identities(self, E, i, j, report, enforce=True):
        """E_pp = 0 and H E H^T = 0 -- construction, not experiment.

        The two are the same statement (H selects the position rows and
        columns), and both are recorded so that a drift shows up as a number
        rather than only as an exception.
        """
        e_pp = float(np.max(np.abs(E[IDX_P, IDX_P])))
        e_h = float(np.max(np.abs(_H @ E @ _H.T)))
        scale = max(float(np.max(np.abs(E))), 1.0)
        report["E_pp_max"] = max(report.get("E_pp_max", 0.0), e_pp)
        report["HEH_max"] = max(report.get("HEH_max", 0.0), e_h)
        if not enforce:
            return
        if e_pp > self.assert_tol * scale or e_h > self.assert_tol * scale:
            raise AssertionError(
                f"structural identity violated on pair ({i},{j}): "
                f"|E_pp|={e_pp:.3e}, |H E H^T|={e_h:.3e}.  The block layout of "
                "L or the choice of H is wrong."
            )

    # -- main entry ----------------------------------------------------
    def evaluate(self, P_x, P_G=None, Phi_bar_list=None, innovations=None,
                 sample_point="post_uwb", pairs=None, mode=None):
        """Diagnose one sampling instant.

        Parameters
        ----------
        P_x           (15K, 15K) centralized joint covariance (gold standard)
        P_G           (3K, 3K)   dual-layer position layer.  Optional in the
                      default structural mode, where it only feeds the separate
                      ``pos_layer_dev`` drift number; REQUIRED in realised mode
        Phi_bar_list  optional per-entity accumulated transitions -> eta_dyn
        innovations   optional list of ``{'i','j','u','S'}`` records -> eta_S
        sample_point  one of ``diag_sample_points``; stored with every row so
                      M22 can pick the right instant for the causal chain
        pairs         optional explicit ``[(i, j), ...]``; default all i < j
        mode          overrides the instance mode for this call
        """
        mode = self.mode if mode is None else mode
        if mode not in ("structural", "realised"):
            raise ValueError(f"unknown diagnostic mode {mode!r}")
        P_x = np.asarray(P_x, dtype=float)
        K = self.K
        if P_x.shape != (STATE_DIM * K,) * 2:
            raise ValueError(f"P_x must be {(STATE_DIM * K,) * 2}, got {P_x.shape}")
        if P_G is not None:
            P_G = np.asarray(P_G, dtype=float)
            if P_G.shape != (POS_DIM * K,) * 2:
                raise ValueError(
                    f"P_G must be {(POS_DIM * K,) * 2}, got {P_G.shape}")
        elif mode == "realised":
            raise ValueError("realised mode requires the estimator's P_G")

        L, W, P_diag = self._lifts_and_whiteners(P_x)
        pair_list = ([(i, j) for i in range(K) for j in range(i + 1, K)]
                     if pairs is None else list(pairs))

        report = {"sample_point": sample_point, "mode": mode}
        rows = {}
        dyn_cache = {}
        for (i, j) in pair_list:
            P_ij = get_block(P_x, i, j, STATE_DIM)
            P_G_proj = _H @ P_ij @ _H.T          # position marginal of the truth
            if mode == "realised":
                P_G_ij = get_block(P_G, i, j, POS_DIM)
            else:
                P_G_ij = P_G_proj
            E = structural_residual(P_ij, P_G_ij, L[i], L[j])
            self._check_identities(E, i, j, report, enforce=(mode == "structural"))

            Ebar = W[i] @ E @ W[j].T
            Pbar = W[i] @ P_ij @ W[j].T
            row = {
                "eta_struct": float(np.linalg.norm(Ebar, "fro")
                                    / (np.linalg.norm(Pbar, "fro") + self.tau)),
                "eta_abs": float(np.linalg.norm(Ebar, 2)),
                "cross_energy": float(np.linalg.norm(Pbar, "fro")),
                # Per-pair identity residuals allow class-stratified summary
                # aggregation without retaining complete diagnostic reports.
                "E_pp_max": float(np.max(np.abs(E[IDX_P, IDX_P]))),
                "HEH_max": float(np.max(np.abs(_H @ E @ _H.T))),
            }
            if P_G is not None:
                # v2's D1, kept as a SEPARATE number: how far the dual layer's
                # own position cross block has drifted from the gold standard.
                dev = get_block(P_G, i, j, POS_DIM) - P_G_proj
                row["pos_layer_dev"] = float(np.linalg.norm(dev, "fro"))
                row["pos_layer_dev_rel"] = float(
                    np.linalg.norm(dev, "fro")
                    / (np.linalg.norm(P_G_proj, "fro") + self.tau))

            if Phi_bar_list is not None:
                Fi = np.asarray(Phi_bar_list[i], dtype=float)
                Fj = np.asarray(Phi_bar_list[j], dtype=float)
                dP = dynamic_residual(E, Fi, Fj)
                den = _H @ Fi @ P_ij @ Fj.T @ _H.T
                ref = np.sqrt(
                    np.linalg.norm(_H @ Fi @ P_diag[i] @ Fi.T @ _H.T, "fro")
                    * np.linalg.norm(_H @ Fj @ P_diag[j] @ Fj.T @ _H.T, "fro"))
                row["eta_dyn"] = float(np.linalg.norm(dP, "fro")
                                       / (np.linalg.norm(den, "fro")
                                          + self.tau * ref))
                row["dP_G_norm"] = float(np.linalg.norm(dP, "fro"))
                decomposition = channel_decomposition(E, Fi, Fj)
                for c in ALL_CHANNELS:
                    row[f"share_{c}"] = decomposition["shares"][c]
                dyn_cache[(i, j)] = dP
            rows[(i, j)] = row

        # ---- eta_S at the innovation instant ----
        if innovations:
            eta_S, dS_list = [], []
            for rec in innovations:
                i, j = int(rec["i"]), int(rec["j"])
                key = (i, j) if (i, j) in dyn_cache else (j, i)
                if key not in dyn_cache:
                    continue
                dP = dyn_cache[key] if key == (i, j) else dyn_cache[key].T
                dS = innovation_variance_error(dP, rec["u"])
                S = float(rec["S"])
                dS_list.append(dS)
                if S > 0.0:
                    eta_S.append(abs(dS) / S)
                    rows.setdefault(key, {})["eta_S"] = abs(dS) / S
                    rows[key]["dS"] = dS
            report["eta_S_mean"] = float(np.mean(eta_S)) if eta_S else float("nan")
            report["eta_S_max"] = float(np.max(eta_S)) if eta_S else float("nan")
            report["dS_max_abs"] = (float(np.max(np.abs(dS_list)))
                                    if dS_list else float("nan"))
        else:
            report["eta_S_mean"] = float("nan")
            report["eta_S_max"] = float("nan")
            report["dS_max_abs"] = float("nan")

        # ---- aggregates ----
        for key in ("eta_struct", "eta_abs", "eta_dyn", "dP_G_norm",
                    "pos_layer_dev", "pos_layer_dev_rel"):
            values = [r[key] for r in rows.values() if key in r]
            report[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
            report[f"{key}_max"] = float(np.max(values)) if values else float("nan")
        for c in ALL_CHANNELS:
            values = [r[f"share_{c}"] for r in rows.values() if f"share_{c}" in r]
            report[f"share_{c}"] = float(np.mean(values)) if values else float("nan")

        # ---- kappa and its analytic partner ----
        if Phi_bar_list is not None:
            kappa = [coupling_scalar(Phi_bar_list[i]) for i in range(K)]
            report["kappa"] = kappa
            report["kappa_mean"] = float(np.mean(kappa))
            report["kappa_plan_curve"] = coupling_prediction(self.cfg)
            report["kappa_full_curve"] = coupling_prediction_full(self.cfg)
            report["kappa_excess_curve"] = coupling_excess(self.cfg)
        report["pairs"] = rows
        return report


if __name__ == "__main__":
    from config import default_config, validate_config, IDX_V
    from l1_linalg import set_block

    cfg = default_config(); validate_config(cfg)
    cfg["num_entities"] = 3; cfg["class_sizes"] = (1, 1, 1)
    cfg["entity_class_map"] = None
    cfg["circle_phase_offsets"] = None; cfg["circle_directions"] = None
    cfg["circle_center_offsets"] = None
    validate_config(cfg)
    K = cfg["num_entities"]
    rng = np.random.default_rng(0)
    diag = StructuralResidualDiagnostic(cfg)

    # ---- build a P_x that EXACTLY satisfies the two-layer structure ----
    A = rng.standard_normal((POS_DIM * K, POS_DIM * K))
    P_G = symmetrize(A @ A.T + np.eye(POS_DIM * K))
    P_x = np.zeros((STATE_DIM * K, STATE_DIM * K))
    L_list = []
    for i in range(K):
        M = rng.standard_normal((STATE_DIM, STATE_DIM))
        P_ii = symmetrize(M @ M.T + np.eye(STATE_DIM))
        P_ii[IDX_P, IDX_P] = get_block(P_G, i, i, POS_DIM)
        L_list.append(lift_matrix(P_ii, cfg["reg_epsilon"]))
    for i in range(K):
        for j in range(K):
            set_block(P_x, i, j, STATE_DIM,
                      L_list[i] @ get_block(P_G, i, j, POS_DIM) @ L_list[j].T)
        D_i = np.zeros((STATE_DIM, STATE_DIM))
        D_i[POS_DIM:, POS_DIM:] = 0.3 * np.eye(STATE_DIM - POS_DIM)
        set_block(P_x, i, i, STATE_DIM, get_block(P_x, i, i, STATE_DIM) + D_i)

    Phi_bar = []
    for i in range(K):
        F = np.eye(STATE_DIM) + 0.05 * rng.standard_normal((STATE_DIM, STATE_DIM))
        Phi_bar.append(F)

    rep0 = diag.evaluate(P_x, P_G, Phi_bar)
    # ---- 1) exact structure -> zero residual, and the identities hold ----
    # (the residual floor ~1e-8 is the regularised lift, not a structural leak)
    assert rep0["eta_struct_max"] < 1e-6, rep0["eta_struct_max"]
    assert rep0["eta_dyn_max"] < 1e-6, rep0["eta_dyn_max"]
    assert rep0["E_pp_max"] < 1e-9 and rep0["HEH_max"] < 1e-9

    # ---- 2) injecting cross-entity velocity correlation is detected ----
    P_bad = P_x.copy()
    extra = np.zeros((STATE_DIM, STATE_DIM)); extra[IDX_V, IDX_V] = 0.2 * np.eye(3)
    set_block(P_bad, 0, 1, STATE_DIM, get_block(P_bad, 0, 1, STATE_DIM) + extra)
    set_block(P_bad, 1, 0, STATE_DIM, get_block(P_bad, 1, 0, STATE_DIM) + extra.T)
    rep1 = diag.evaluate(P_bad, P_G, Phi_bar)
    assert rep1["eta_struct_max"] > 1e-3
    assert rep1["eta_dyn_max"] > 1e-6
    # ... and it lands in the velocity channel
    row01 = rep1["pairs"][(0, 1)]
    assert row01["share_vel"] > 0.5, row01

    # ---- 3) E_pp stays exactly zero whatever P_G does (T16), and the drift
    #         of P_G away from the gold standard is reported SEPARATELY ----
    P_G_rand = symmetrize(rng.standard_normal((3 * K, 3 * K)))
    P_G_rand += (abs(np.min(np.linalg.eigvalsh(P_G_rand))) + 1.0) * np.eye(3 * K)
    rep2 = diag.evaluate(P_x, P_G_rand, Phi_bar)
    assert rep2["mode"] == "structural"
    assert rep2["E_pp_max"] < 1e-9 and rep2["HEH_max"] < 1e-9
    assert rep2["eta_struct_max"] < 1e-6            # structure is still exact
    assert rep2["pos_layer_dev_max"] > 1.0          # but P_G itself is way off
    assert rep0["pos_layer_dev_max"] < 1e-9         # ... and zero when it is not
    # realised mode mixes the two, so it does NOT assert the identity
    rep2r = diag.evaluate(P_x, P_G_rand, Phi_bar, mode="realised")
    assert rep2r["E_pp_max"] > 1.0 and rep2r["eta_struct_max"] > 1e-3
    try:
        diag.evaluate(P_x, None, Phi_bar, mode="realised")
        raise AssertionError("realised mode ran without a P_G")
    except ValueError:
        pass
    assert diag.evaluate(P_x, None, Phi_bar)["eta_struct_max"] < 1e-6

    # ---- 4) unit invariance: metres -> centimetres (T17) ----
    s = 100.0
    Su = np.diag(np.concatenate([np.full(3, s), np.full(3, s), np.ones(3),
                                 np.full(3, s), np.ones(3)]))
    Sk = np.kron(np.eye(K), Su)
    P_x_cm = Sk @ P_bad @ Sk.T
    P_G_cm = s ** 2 * P_G
    Phi_cm = [Su @ Phi_bar[i] @ np.linalg.inv(Su) for i in range(K)]
    rep_cm = diag.evaluate(P_x_cm, P_G_cm, Phi_cm)
    for key in ("eta_struct_max", "eta_abs_max", "eta_dyn_max"):
        a, b = rep1[key], rep_cm[key]
        assert abs(a - b) < 1e-7 * max(abs(a), 1.0), (key, a, b)
    for c in ALL_CHANNELS:
        assert abs(rep1[f"share_{c}"] - rep_cm[f"share_{c}"]) < 1e-7, c
    # the same test with the plan's literal lambda*I whitener would NOT pass:
    W_I = invsqrt_psd(get_block(P_bad, 0, 0, STATE_DIM) + 1e-6 * np.eye(STATE_DIM))
    W_I_cm = invsqrt_psd(get_block(P_x_cm, 0, 0, STATE_DIM) + 1e-6 * np.eye(STATE_DIM))
    E_m = structural_residual(get_block(P_bad, 0, 1, STATE_DIM),
                              get_block(P_G, 0, 1, POS_DIM), L_list[0], L_list[1])
    n_m = np.linalg.norm(W_I @ E_m @ W_I.T, "fro")
    n_cm = np.linalg.norm(W_I_cm @ (Su @ E_m @ Su.T) @ W_I_cm.T, "fro")
    assert abs(n_m - n_cm) > 1e-6 * max(n_m, 1.0), (n_m, n_cm)

    # ---- 5) the batch object and the reference loop agree (T18) ----
    ref = cross_structure_metrics(P_bad, P_G, K, cfg["whiten_lambda"],
                                  cfg["diag_tau"], cfg["reg_epsilon"], Phi_bar)
    for (i, j), row in rep1["pairs"].items():
        for key in ("eta_struct", "eta_abs", "eta_dyn"):
            assert abs(row[key] - ref[(i, j)][key]) < 1e-12, (i, j, key)

    # ---- 6) the channel expansion is exact ----
    # rebuild E through the SAME on-site lifts the diagnostic uses, so the
    # comparison below tests the algebra and not the lift regularisation
    L_site, _, _ = diag._lifts_and_whiteners(P_bad)
    P01 = get_block(P_bad, 0, 1, STATE_DIM)
    E01 = structural_residual(P01, _H @ P01 @ _H.T, L_site[0], L_site[1])
    dec = channel_decomposition(E01, Phi_bar[0], Phi_bar[1])
    assert np.max(np.abs(dec["total"]
                         - dynamic_residual(E01, Phi_bar[0], Phi_bar[1]))) < 1e-12
    assert abs(sum(dec["shares"].values()) - 1.0) < 1e-12
    # the (pos, pos) term is structurally absent
    assert dec["term_norms"]["pos"]["pos"] < 1e-12

    # ---- 7) eta_S: a strongly correlated pair inflates it above eta_dyn ----
    u = np.array([1.0, 0.0, 0.0])
    dP = dynamic_residual(E01, Phi_bar[0], Phi_bar[1])
    dS = innovation_variance_error(dP, u)
    P_pos_pair = np.block([[get_block(P_G, 0, 0, 3), get_block(P_G, 0, 1, 3)],
                           [get_block(P_G, 1, 0, 3), get_block(P_G, 1, 1, 3)]])
    h = np.concatenate([u, -u])
    S_pair = float(h @ P_pos_pair @ h) + cfg["sigma_range"] ** 2
    rep_S = diag.evaluate(P_bad, P_G, Phi_bar,
                          innovations=[{"i": 0, "j": 1, "u": u, "S": S_pair}],
                          sample_point="at_innovation")
    assert rep_S["sample_point"] == "at_innovation"
    assert abs(rep_S["pairs"][(0, 1)]["dS"] - dS) < 1e-12 * max(abs(dS), 1.0)
    assert np.isfinite(rep_S["eta_S_mean"])

    # ---- 8) analytic channel orders scale in T as advertised ----
    o1 = analytic_orders(cfg, 0.1)
    o2 = analytic_orders(cfg, 0.2)
    assert abs(o2["vel"] / o1["vel"] - 2.0) < 1e-12
    assert abs(o2["att"] / o1["att"] - 4.0) < 1e-12
    assert abs(o2["acc_bias"] / o1["acc_bias"] - 4.0) < 1e-12
    assert abs(o2["gyro_bias"] / o1["gyro_bias"] - 8.0) < 1e-12
    assert o1["att_full_curve"] > 10.0 * o1["att_plan_curve"]   # gravity dominates

    # ---- 9) a wrongly built L is caught by the identity assertion ----
    class BrokenDiagnostic(StructuralResidualDiagnostic):
        def _lifts_and_whiteners(self, P_x_local):
            L, W, d = super()._lifts_and_whiteners(P_x_local)
            L = [Li.copy() for Li in L]
            L[0][IDX_P, :] = 0.5 * np.eye(POS_DIM)        # break H L = I
            return L, W, d

    try:
        BrokenDiagnostic(cfg).evaluate(P_bad, P_G, Phi_bar)
        raise AssertionError("a broken lift was not detected")
    except AssertionError as exc:
        assert "structural identity violated" in str(exc), exc

    # ---- 10) shape validation ----
    try:
        diag.evaluate(np.eye(4), P_G, Phi_bar)
        raise AssertionError("evaluate accepted a wrongly shaped P_x")
    except ValueError:
        pass

    print("[M12 StructuralResidualDiagnostic] OK  (exact structure eta=%.1e, "
          "injected eta_struct=%.3f eta_dyn=%.1e, vel share=%.2f, unit-invariant "
          "to %.1e)"
          % (rep0["eta_struct_max"], rep1["eta_struct_max"], rep1["eta_dyn_max"],
             row01["share_vel"],
             abs(rep1["eta_struct_max"] - rep_cm["eta_struct_max"])))
