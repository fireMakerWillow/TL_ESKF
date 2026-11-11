"""M10 Metrics (L1 evaluation kernel) -- accuracy, self-reported spread, NEES.

Three families of numbers, deliberately kept apart because they answer three
different questions (plan sections 8.1 / 8.2):

    RMSE   how far the estimate is from the truth        (needs the truth)
    RMTE   how uncertain the filter CLAIMS to be         (needs no truth)
    NEES   whether the claim matches the error           (needs both)

RMSE alone cannot expose an inconsistent filter and NEES alone cannot say
whether an inconsistency is expensive, so all three are reported together, and
in v3 all three are additionally reported per entity class A / B / C.  Class C
sees no anchor at all, so its absolute position information arrives ONLY through
the cooperative link: it is the class on which the treatment of the cross-entity
correlations actually shows up.

Error vectors are built with M2's ``boxminus`` -- ``dtheta = Log(R_hat^T R)`` --
and never by subtracting Euler angles, which at large tilt is not the tangent
error the covariance describes and would forge both RMSE and NEES.

RMTE (root mean trace error)
----------------------------
    RMTE = sqrt( (1/K) sum_i tr(H P_L,i H^T) )

i.e. the root-mean position variance the filter reports for itself.  It uses the
DIAGONAL blocks only, so it is insensitive to how a method handles cross blocks
and therefore CANNOT replace the joint position NEES; it is the companion that
turns a NEES verdict into a magnitude ("overconfident by how many metres").
Read it paired with the RMSE: ``RMSE/RMTE >> 1`` is overconfidence, ``<< 1`` is
over-conservatism.

For that reading to hold, the RMSE must be the PER-ENTITY VECTOR magnitude

    RMSE = sqrt( (1/K) sum_i ||e_i||^2 ),

because ``E[||e_i||^2] = tr(P_i)`` makes the ratio exactly 1 for a consistent
filter.  Averaging over the 3K scalar components instead -- the other common
convention -- divides by 3K and lands a consistent filter at 1/sqrt(3) = 0.577,
which would be silently read as "over-conservative" for every method in the
table.  All channels here therefore use the per-entity magnitude convention, so
a position RMSE is sqrt(3) times the per-axis figure and the attitude RMSE is
the magnitude of the rotation-vector error.

Note on ``rmte_definition``.  Section 1.5 of the design lists the value
``'pairwise_relative'`` while section [M10] writes the trace formula above,
which is diagonal-only.  The two are different metrics, so both are implemented
here and the main line uses ``'diagonal_trace'`` (the formula actually written
in [M10] and the one the RMSE/RMTE pairing needs).  ``'pairwise_relative'``
averages ``tr(P^ii + P^jj - P^ij - P^ji)`` over pairs; it is a genuine relative
-geometry spread but it DOES read the cross blocks, so it must not be quoted as
if it were the diagonal-only quantity.

Monte-Carlo contract.  Per-epoch NEES samples are returned raw; the chi-square
verdict is formed by :func:`summarize_nees` over the Monte-Carlo ensemble, never
by averaging inside one trial -- the degrees of freedom of the ANEES bound are
only correct when the samples are independent trials.

Channels a method does not estimate are reported as ``NaN`` (M26 has no
attitude and no biases); they are never filled with a placeholder number that
would read like a consistency result.
"""

import numpy as np
from scipy.stats import chi2

from config import (STATE_DIM, POS_DIM, ENTITY_CLASSES,
                    IDX_P, IDX_V, IDX_TH, IDX_BA, IDX_BG)
from l1_linalg import gather_indices, safe_inverse, symmetrize
from l1_so3 import boxminus

CHANNELS = ("pos", "vel", "att", "acc_bias", "gyro_bias")
CHANNEL_SLICES = {"pos": IDX_P, "vel": IDX_V, "att": IDX_TH,
                  "acc_bias": IDX_BA, "gyro_bias": IDX_BG}
RMTE_DEFINITIONS = ("diagonal_trace", "pairwise_relative")


# ----------------------------------------------------------------------
# error construction
# ----------------------------------------------------------------------
def error_vectors(truth_states, nominal_states):
    """Stack ``dx_i = truth_i boxminus nominal_i`` into an ``(K, 15)`` array."""
    if len(truth_states) != len(nominal_states):
        raise ValueError("truth and nominal state lists differ in length")
    K = len(truth_states)
    E = np.zeros((K, STATE_DIM))
    for i in range(K):
        E[i] = boxminus(truth_states[i], nominal_states[i])
    return E


def position_error(truth_states, nominal_states):
    """Stacked position error ``(3K,)`` in entity order."""
    return error_vectors(truth_states, nominal_states)[:, IDX_P].ravel()


# ----------------------------------------------------------------------
# RMSE
# ----------------------------------------------------------------------
def _rms(block):
    """RMS of the per-entity error MAGNITUDE (see the RMTE note above)."""
    block = np.atleast_2d(np.asarray(block, dtype=float))
    if block.size == 0:
        return float("nan")
    return float(np.sqrt(np.sum(block ** 2) / block.shape[0]))


def _rms_per_component(block):
    """RMS over scalar components -- the other convention, kept for reference."""
    block = np.asarray(block, dtype=float).ravel()
    if block.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(block ** 2)))


def rmse_breakdown(E, channels=CHANNELS):
    """Per-channel RMSE of a stacked error matrix ``(K, 15)``.

    Channels outside ``channels`` are returned as NaN rather than computed, so
    a position-only estimator never reports an attitude accuracy it does not
    have.
    """
    E = np.asarray(E, dtype=float)
    active = set(channels)
    out = {}
    for name in CHANNELS:
        if name not in active:
            out[f"{name}_rmse"] = float("nan")
            continue
        out[f"{name}_rmse"] = _rms(E[:, CHANNEL_SLICES[name]])
    out["att_rmse_deg"] = (np.degrees(out["att_rmse"])
                           if np.isfinite(out["att_rmse"]) else float("nan"))
    return out


def rmse_by_class(E, class_map, channel="pos"):
    """RMSE of one channel restricted to each entity class."""
    E = np.asarray(E, dtype=float)
    sl = CHANNEL_SLICES[channel]
    out = {}
    for label in ENTITY_CLASSES:
        idx = [i for i, c in enumerate(class_map) if c == label]
        out[label] = _rms(E[idx][:, sl]) if idx else float("nan")
    return out


# ----------------------------------------------------------------------
# RMTE -- the filter's self-reported position spread (truth-free)
# ----------------------------------------------------------------------
def _diag_block(P, i):
    return np.asarray(P)[i * POS_DIM:(i + 1) * POS_DIM,
                         i * POS_DIM:(i + 1) * POS_DIM]


def _cross_block(P, i, j):
    return np.asarray(P)[i * POS_DIM:(i + 1) * POS_DIM,
                         j * POS_DIM:(j + 1) * POS_DIM]


def rmte(P_pos, K, definition="diagonal_trace", indices=None):
    """Root mean trace of the reported position covariance.

    ``P_pos`` is the (3K, 3K) joint position covariance a method reports.
    ``indices`` restricts the average to a subset of entities (a class).
    """
    if definition not in RMTE_DEFINITIONS:
        raise ValueError(f"unknown rmte_definition {definition!r}")
    P_pos = np.asarray(P_pos, dtype=float)
    idx = list(range(int(K))) if indices is None else list(indices)
    if not idx:
        return float("nan")
    if definition == "diagonal_trace":
        total = sum(float(np.trace(_diag_block(P_pos, i))) for i in idx)
        return float(np.sqrt(max(total, 0.0) / len(idx)))
    # pairwise_relative: mean trace of the RELATIVE position covariance
    if len(idx) < 2:
        return float("nan")
    total, pairs = 0.0, 0
    for a, i in enumerate(idx):
        for j in idx[a + 1:]:
            rel = (_diag_block(P_pos, i) + _diag_block(P_pos, j)
                   - _cross_block(P_pos, i, j) - _cross_block(P_pos, j, i))
            total += float(np.trace(rel))
            pairs += 1
    return float(np.sqrt(max(total, 0.0) / pairs))


def rmte_by_class(P_pos, class_map, definition="diagonal_trace"):
    """RMTE restricted to each entity class."""
    K = len(class_map)
    out = {}
    for label in ENTITY_CLASSES:
        idx = [i for i, c in enumerate(class_map) if c == label]
        out[label] = (rmte(P_pos, K, definition, idx) if idx else float("nan"))
    return out


# ----------------------------------------------------------------------
# NEES
# ----------------------------------------------------------------------
def nees(e, P, eps=1e-12):
    """Generic normalised estimation error squared ``e^T P^{-1} e``."""
    e = np.asarray(e, dtype=float).ravel()
    if e.size == 0:
        return float("nan")
    return float(e @ safe_inverse(P, eps) @ e)


def joint_position_nees(pos_err, P_pos, eps=1e-12):
    """Joint position NEES over all entities (dof = 3K)."""
    return nees(pos_err, P_pos, eps)


def marginal_position_nees(pos_err, P_pos, indices, eps=1e-12):
    """Marginal position NEES of an entity subset (dof = 3|subset|).

    The marginal of a Gaussian is the SUBMATRIX of the joint covariance, so the
    cross blocks inside the subset are kept and only the outside is dropped.
    """
    idx = gather_indices(indices, POS_DIM)
    if idx.size == 0:
        return float("nan")
    e = np.asarray(pos_err, dtype=float).ravel()[idx]
    return nees(e, np.asarray(P_pos, dtype=float)[np.ix_(idx, idx)], eps)


def nees_by_class(pos_err, P_pos, class_map, eps=1e-12):
    """Per-class marginal position NEES and the matching degrees of freedom."""
    out = {}
    for label in ENTITY_CLASSES:
        idx = [i for i, c in enumerate(class_map) if c == label]
        out[label] = (marginal_position_nees(pos_err, P_pos, idx, eps)
                      if idx else float("nan"))
        out[f"{label}_dof"] = POS_DIM * len(idx)
    return out


def full_state_nees(E, P_full, eps=1e-12):
    """Full-state NEES (dof = 15K).  Internal acceptance only (T12/T13)."""
    if P_full is None:
        return float("nan")
    return nees(np.asarray(E, dtype=float).ravel(), P_full, eps)


def channel_nees(E, P_full, channel, eps=1e-12):
    """Per-channel NEES over all entities (dof = 3K).  Internal only."""
    if P_full is None:
        return float("nan")
    E = np.asarray(E, dtype=float)
    K = E.shape[0]
    sl = CHANNEL_SLICES[channel]
    idx = np.concatenate([np.arange(i * STATE_DIM + sl.start,
                                    i * STATE_DIM + sl.stop) for i in range(K)])
    P = np.asarray(P_full, dtype=float)[np.ix_(idx, idx)]
    return nees(E[:, sl].ravel(), P, eps)


# ----------------------------------------------------------------------
# chi-square verdicts
# ----------------------------------------------------------------------
def single_run_bounds(dof, confidence):
    """Two-sided chi-square interval for ONE NEES sample."""
    alpha = 1.0 - float(confidence)
    return (float(chi2.ppf(alpha / 2.0, dof)),
            float(chi2.ppf(1.0 - alpha / 2.0, dof)))


def anees_bounds(dof, num_samples, confidence):
    """Two-sided interval for the AVERAGE of ``num_samples`` NEES values."""
    alpha = 1.0 - float(confidence)
    n = int(num_samples)
    return (float(chi2.ppf(alpha / 2.0, n * dof) / n),
            float(chi2.ppf(1.0 - alpha / 2.0, n * dof) / n))


def consistency_verdict(value, lo, hi):
    """'overconfident' above the band, 'overconservative' below it."""
    if not np.isfinite(value):
        return "unavailable"
    if value > hi:
        return "overconfident"
    if value < lo:
        return "overconservative"
    return "in_band"


def summarize_nees(samples, dof, confidence):
    """ANEES of a Monte-Carlo ensemble with its band and verdict."""
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0 or dof <= 0:
        return {"anees": float("nan"), "anees_ideal": float(dof),
                "anees_bounds": (float("nan"), float("nan")),
                "anees_consistent": False, "verdict": "unavailable",
                "single_run_bounds": (float("nan"), float("nan")),
                "in_bound_fraction": float("nan"), "num_samples": 0}
    anees = float(np.mean(s))
    lo1, hi1 = single_run_bounds(dof, confidence)
    lob, hib = anees_bounds(dof, s.size, confidence)
    return {
        "anees": anees,
        "anees_ideal": float(dof),
        "anees_bounds": (lob, hib),
        "anees_consistent": bool(lob <= anees <= hib),
        "verdict": consistency_verdict(anees, lob, hib),
        "single_run_bounds": (lo1, hi1),
        "in_bound_fraction": float(np.mean((s >= lo1) & (s <= hi1))),
        "num_samples": int(s.size),
    }


def normalised_anees(samples, dof):
    """ANEES / dof -- the scale-free number used by the M22 zone rules."""
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0 or dof <= 0:
        return float("nan")
    return float(np.mean(s) / dof)


# ----------------------------------------------------------------------
# one-epoch evaluation (the entry point M20 calls)
# ----------------------------------------------------------------------
def evaluate(truth_states, nominal_states, eval_cov, config,
             full_cov=None, channels=CHANNELS):
    """Evaluate one epoch of one estimator.

    Parameters
    ----------
    truth_states, nominal_states : list[NominalState]
    eval_cov  : (3K, 3K) joint POSITION covariance as the method reports it
    full_cov  : optional (15K, 15K) full covariance; ``None`` -> internal NEES
                columns are NaN (this is how M26 is handled)
    channels  : the channels the method actually estimates

    Returns a flat dict of scalars; every key exists for every method so the
    Monte-Carlo table is rectangular.
    """
    class_map = list(config["entity_class_map"])
    K = len(class_map)
    conf = float(config["nees_confidence"])
    definition = config.get("rmte_definition", "diagonal_trace")

    E = error_vectors(truth_states, nominal_states)
    if E.shape[0] != K:
        raise ValueError(f"expected {K} entities, got {E.shape[0]}")
    eval_cov = np.asarray(eval_cov, dtype=float)
    if eval_cov.shape != (POS_DIM * K,) * 2:
        raise ValueError(
            f"eval_cov must be {(POS_DIM * K, POS_DIM * K)}, got {eval_cov.shape}")

    out = rmse_breakdown(E, channels)
    pos_err = E[:, IDX_P].ravel()

    # ---- per-class RMSE / RMTE ----
    for label, value in rmse_by_class(E, class_map, "pos").items():
        out[f"pos_rmse_{label}"] = value
    out["rmte"] = rmte(eval_cov, K, definition)
    for label, value in rmte_by_class(eval_cov, class_map, definition).items():
        out[f"rmte_{label}"] = value
    out["rmse_rmte_ratio"] = (out["pos_rmse"] / out["rmte"]
                              if out["rmte"] > 0 else float("nan"))

    # ---- joint and per-class position NEES ----
    out["nees_pos"] = joint_position_nees(pos_err, eval_cov)
    out["nees_pos_dof"] = POS_DIM * K
    lo, hi = single_run_bounds(POS_DIM * K, conf)
    out["nees_pos_verdict"] = consistency_verdict(out["nees_pos"], lo, hi)
    per_class = nees_by_class(pos_err, eval_cov, class_map)
    for label in ENTITY_CLASSES:
        out[f"nees_pos_{label}"] = per_class[label]
        out[f"nees_pos_{label}_dof"] = per_class[f"{label}_dof"]

    # ---- internal-only full-state and per-channel NEES (T12 / T13) ----
    out["nees_full"] = full_state_nees(E, full_cov)
    out["nees_full_dof"] = STATE_DIM * K if full_cov is not None else 0
    # NOTE the ``nees_ch_`` prefix.  These are the per-channel full-state NEES
    # columns of T12 and they are NOT the main-line metrics: writing the
    # position one to ``nees_pos`` would silently overwrite the joint position
    # NEES computed above with a NaN for every method that reports no full
    # covariance (M26, and any method evaluated without ``full_cov``), i.e. the
    # paper's headline consistency column would go blank exactly where the
    # reported structure differs from the full one.
    for name in CHANNELS:
        out[f"nees_ch_{name}"] = (channel_nees(E, full_cov, name)
                                  if (full_cov is not None and name in channels)
                                  else float("nan"))
    return out


def aggregate_trials(per_trial, config, keys=("nees_pos",), dofs=None):
    """Aggregate per-trial NEES samples into ANEES verdicts.

    ``per_trial[key]`` is a sequence with ONE value per Monte-Carlo trial (a
    per-trial time average of the raw samples is acceptable input; what is not
    acceptable is averaging over time INSIDE a trial and then treating the
    result as a chi-square sample with the per-epoch dof).
    """
    conf = float(config["nees_confidence"])
    K = int(config["num_entities"])
    default_dof = {"nees_pos": POS_DIM * K,
                   "nees_full": STATE_DIM * K}
    for label in ENTITY_CLASSES:
        n = list(config["entity_class_map"]).count(label)
        default_dof[f"nees_pos_{label}"] = POS_DIM * n
    dofs = dict(default_dof, **(dofs or {}))
    return {key: summarize_nees(per_trial[key], dofs.get(key, POS_DIM * K), conf)
            for key in keys if key in per_trial}


if __name__ == "__main__":
    from config import default_config, validate_config
    from l1_so3 import NominalState, Exp, boxplus

    cfg = default_config(); validate_config(cfg)
    rng = np.random.default_rng(0)
    K = cfg["num_entities"]
    class_map = cfg["entity_class_map"]

    # ---- 1) boxminus round trip: a known perturbation is recovered ----
    noms, truths, dxs = [], [], []
    for _ in range(K):
        nom = NominalState(p=rng.standard_normal(3), v=rng.standard_normal(3),
                           R=Exp(0.5 * rng.standard_normal(3)),
                           b_a=0.01 * rng.standard_normal(3),
                           b_g=1e-3 * rng.standard_normal(3))
        dx = 1e-3 * rng.standard_normal(STATE_DIM)
        noms.append(nom); dxs.append(dx); truths.append(boxplus(nom, dx))
    E = error_vectors(truths, noms)
    assert np.allclose(E, np.array(dxs), atol=1e-8)

    # ---- 2) large tilt: boxminus stays exact where Euler would not ----
    nom = NominalState(R=Exp([0.0, np.radians(89.0), 0.0]))
    tilted = boxplus(nom, np.array([0] * 6 + [0.0, 0.02, 0.0] + [0] * 6, float))
    assert np.allclose(boxminus(tilted, nom)[IDX_TH], [0.0, 0.02, 0.0], atol=1e-9)

    # ---- 3) RMSE channels and the per-class split ----
    br = rmse_breakdown(E)
    assert abs(br["pos_rmse"] - _rms(E[:, IDX_P])) < 1e-15
    # the per-entity magnitude convention is sqrt(3) times the per-axis one
    assert abs(br["pos_rmse"] - np.sqrt(3.0) * _rms_per_component(E[:, IDX_P])) < 1e-12
    assert abs(br["pos_rmse"]
               - np.sqrt(np.mean(np.sum(E[:, IDX_P] ** 2, axis=1)))) < 1e-15
    assert abs(br["att_rmse_deg"] - np.degrees(br["att_rmse"])) < 1e-15
    partial = rmse_breakdown(E, channels=("pos",))
    assert np.isfinite(partial["pos_rmse"]) and np.isnan(partial["vel_rmse"])
    assert np.isnan(partial["att_rmse_deg"])
    by_c = rmse_by_class(E, class_map, "pos")
    # the class RMSEs recombine into the global one with the right weights
    recomb = np.sqrt(sum(class_map.count(c) * by_c[c] ** 2 for c in ENTITY_CLASSES)
                     / K)
    assert abs(recomb - br["pos_rmse"]) < 1e-12

    # ---- 4) RMTE: definition, class split, and exact alpha scaling (T21) ----
    blocks = [symmetrize(np.diag([0.04, 0.05, 0.09]) + 0.01 * np.eye(3))
              for _ in range(K)]
    P_pos = np.zeros((3 * K, 3 * K))
    for i in range(K):
        P_pos[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = blocks[i]
    expect = np.sqrt(np.mean([np.trace(b) for b in blocks]))
    assert abs(rmte(P_pos, K) - expect) < 1e-15
    alpha = 3.0
    assert abs(rmte(alpha ** 2 * P_pos, K) - alpha * rmte(P_pos, K)) < 1e-12
    cls = rmte_by_class(P_pos, class_map)
    assert all(np.isfinite(cls[c]) for c in ENTITY_CLASSES)

    # RMTE ignores cross blocks; the pairwise definition does not
    P_corr = P_pos.copy()
    P_corr[0:3, 3:6] = P_corr[3:6, 0:3] = 0.02 * np.eye(3)
    assert abs(rmte(P_corr, K) - rmte(P_pos, K)) < 1e-15
    assert rmte(P_corr, K, "pairwise_relative") < rmte(P_pos, K, "pairwise_relative")

    # ---- 4b) a CONSISTENT filter lands at RMSE/RMTE = 1 ----
    cov = np.diag([0.04, 0.09, 0.01])
    chol = np.linalg.cholesky(cov)
    P_cons = np.zeros((3 * K, 3 * K))
    for i in range(K):
        P_cons[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = cov
    draws = np.stack([(chol @ rng.standard_normal((3, K))).T
                      for _ in range(2000)])            # (N, K, 3)
    emp = float(np.sqrt(np.mean(np.sum(draws ** 2, axis=2))))
    assert abs(emp / rmte(P_cons, K) - 1.0) < 0.03, emp / rmte(P_cons, K)

    # ---- 5) RMTE is truth-free, RMSE is not ----
    shift = np.array([3.0, -1.0, 2.0])
    shifted_truth = [t.copy() for t in truths]
    for st in shifted_truth:
        st.p = st.p + shift
    E_shift = error_vectors(shifted_truth, noms)
    assert rmse_breakdown(E_shift)["pos_rmse"] != br["pos_rmse"]
    assert rmte(P_pos, K) == expect          # bit-identical: no truth was read
    # shifting truth AND nominal together leaves the error, hence RMSE, intact
    shifted_nom = [n.copy() for n in noms]
    for st in shifted_nom:
        st.p = st.p + shift
    assert abs(rmse_breakdown(error_vectors(shifted_truth, shifted_nom))["pos_rmse"]
               - br["pos_rmse"]) < 1e-12

    # ---- 6) NEES of Gaussian samples averages to the dof ----
    dof = 3 * K
    P = 0.25 * np.eye(dof)
    Lc = np.linalg.cholesky(P)
    samples = [joint_position_nees(Lc @ rng.standard_normal(dof), P)
               for _ in range(4000)]
    summ = summarize_nees(samples, dof, 0.95)
    assert abs(summ["anees"] - dof) < 0.5 and summ["anees_consistent"]
    assert abs(summ["in_bound_fraction"] - 0.95) < 0.03

    # a covariance that is too small is called overconfident
    bad = [joint_position_nees(Lc @ rng.standard_normal(dof), 0.25 * P)
           for _ in range(4000)]
    assert summarize_nees(bad, dof, 0.95)["verdict"] == "overconfident"
    good = [joint_position_nees(Lc @ rng.standard_normal(dof), 4.0 * P)
            for _ in range(4000)]
    assert summarize_nees(good, dof, 0.95)["verdict"] == "overconservative"

    # ---- 7) marginal NEES of a class matches its own dof ----
    idx_A = [i for i, c in enumerate(class_map) if c == "A"]
    marg = [marginal_position_nees(Lc @ rng.standard_normal(dof), P, idx_A)
            for _ in range(4000)]
    assert abs(np.mean(marg) - 3 * len(idx_A)) < 0.4, np.mean(marg)

    # What discarding TRUE cross correlations does to the joint NEES.  It does
    # NOT bias the mean: E[e^T P_bd^{-1} e] = tr(P_bd^{-1} P) = 3K whenever the
    # diagonal blocks are right.  What it does is over-DISPERSE the statistic
    # (the samples are no longer chi-square), which collapses the in-band
    # fraction.  This is why DCL/local-only's inconsistency in M21 must be read as
    # "the diagonal blocks themselves shrink too fast" -- the mere absence of
    # off-diagonal blocks would be invisible to an ANEES mean.
    Pc = 0.25 * (np.eye(dof) + 0.6 * np.kron(np.ones((K, K)) - np.eye(K), np.eye(3)))
    Lf = np.linalg.cholesky(Pc)
    P_blockdiag = np.zeros_like(Pc)
    for i in range(K):
        P_blockdiag[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = Pc[i * 3:(i + 1) * 3,
                                                               i * 3:(i + 1) * 3]
    full_s, drop_s = [], []
    for _ in range(4000):
        e = Lf @ rng.standard_normal(dof)
        full_s.append(joint_position_nees(e, Pc))
        drop_s.append(joint_position_nees(e, P_blockdiag))
    assert abs(np.mean(full_s) - dof) < 0.6
    assert abs(np.mean(drop_s) - dof) < 1.5, np.mean(drop_s)
    assert np.var(drop_s) > 3.0 * np.var(full_s)
    band_full = summarize_nees(full_s, dof, 0.95)["in_bound_fraction"]
    band_drop = summarize_nees(drop_s, dof, 0.95)["in_bound_fraction"]
    assert band_drop < band_full - 0.2, (band_drop, band_full)

    # the failure mode that DOES move the mean: shrunken diagonal blocks
    shrunk = [joint_position_nees(Lf @ rng.standard_normal(dof), 0.5 * P_blockdiag)
              for _ in range(3000)]
    assert summarize_nees(shrunk, dof, 0.95)["verdict"] == "overconfident"

    # ---- 8) full-state and per-channel NEES ----
    P_full = np.eye(STATE_DIM * K)
    Ef = rng.standard_normal((K, STATE_DIM))
    assert abs(full_state_nees(Ef, P_full) - float(np.sum(Ef ** 2))) < 1e-8
    assert abs(channel_nees(Ef, P_full, "att")
               - float(np.sum(Ef[:, IDX_TH] ** 2))) < 1e-8
    assert np.isnan(full_state_nees(Ef, None))

    # ---- 9) the one-epoch evaluation entry point ----
    row = evaluate(truths, noms, P_pos, cfg, full_cov=None)
    for label in ENTITY_CLASSES:
        assert f"nees_pos_{label}" in row and f"rmte_{label}" in row
    assert np.isnan(row["nees_full"]) and row["nees_pos_dof"] == 3 * K
    assert row["nees_pos_C_dof"] == 3 * class_map.count("C")
    row_pos_only = evaluate(truths, noms, P_pos, cfg, channels=("pos",))
    assert np.isnan(row_pos_only["vel_rmse"]) and np.isnan(row_pos_only["nees_ch_att"])
    try:
        evaluate(truths, noms, np.eye(3), cfg)
        raise AssertionError("evaluate accepted a wrongly shaped eval_cov")
    except ValueError:
        pass

    # ---- 10) the Monte-Carlo aggregator ----
    agg = aggregate_trials({"nees_pos": samples}, cfg, keys=("nees_pos",))
    assert agg["nees_pos"]["num_samples"] == len(samples)
    assert abs(normalised_anees(samples, dof) - 1.0) < 0.05

    print("[M10 Metrics] OK  (ANEES=%.2f/%d, RMTE=%.4f m scales exactly, "
          "in-band %.2f -> %.2f when cross blocks are dropped)"
          % (summ["anees"], dof, rmte(P_pos, K), band_full, band_drop))
