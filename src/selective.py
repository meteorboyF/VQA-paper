"""
Selective prediction utilities (E7 / RQ2).

risk_coverage_curve  — build (coverage, risk) arrays from confidences + correctness.
aurc                 — Area Under the Risk-Coverage curve.
coverage_at_risk     — coverage achieved at a given max-risk level.
global_gate          — single confidence threshold policy.
defect_conditioned_gate — per-defect threshold policy (the headline C1 result).
"""
import numpy as np
from src.calibration import ece


# ── Risk-coverage curve ───────────────────────────────────────────────────────

def risk_coverage_curve(confs: np.ndarray, correct: np.ndarray,
                        n_points: int = 50):
    """
    Sort samples by descending confidence.
    Returns (coverage_array, risk_array) both of length n.
    coverage[k] = fraction of dataset answered after considering top-k.
    risk[k]     = 1 − accuracy among the answered top-k.
    """
    order  = np.argsort(-confs)
    c      = correct[order].astype(float)
    n      = len(c)
    cov    = np.arange(1, n + 1) / n
    risk   = 1 - np.cumsum(c) / np.arange(1, n + 1)
    return cov, risk


def aurc(confs: np.ndarray, correct: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (lower is better)."""
    cov, risk = risk_coverage_curve(confs, correct)
    return float(np.trapz(risk, cov))


def coverage_at_risk(confs: np.ndarray, correct: np.ndarray,
                     max_risk: float) -> float:
    """Maximum coverage achievable while keeping risk ≤ max_risk."""
    cov, risk = risk_coverage_curve(confs, correct)
    mask = risk <= max_risk
    return float(cov[mask].max()) if mask.any() else 0.0


# ── Gating policies ───────────────────────────────────────────────────────────

def global_gate(confs: np.ndarray, threshold: float) -> np.ndarray:
    """Answer if conf ≥ threshold, abstain otherwise. Returns boolean mask."""
    return confs >= threshold


def defect_conditioned_gate(
    confs: np.ndarray,
    pred_defect_ids: np.ndarray,
    thresholds: dict,
    fallback_threshold: float,
) -> np.ndarray:
    """
    Per-defect threshold policy. For each sample, pick the threshold that
    corresponds to its predicted defect. Falls back to `fallback_threshold`
    if the defect has no learned threshold.
    """
    answered = np.zeros(len(confs), dtype=bool)
    for i, (conf, defect_id) in enumerate(zip(confs, pred_defect_ids)):
        t = thresholds.get(int(defect_id), fallback_threshold)
        answered[i] = conf >= t
    return answered


def find_global_threshold(confs: np.ndarray, correct: np.ndarray,
                           target_coverage: float = 0.80,
                           split_name: str = "cal") -> float:
    """
    Find the global confidence threshold that achieves ~target_coverage on cal.
    Frozen-knob: must be called with split_name="cal".
    """
    from src.env import assert_no_rep_leakage
    assert_no_rep_leakage(split_name)
    order = np.argsort(-confs)
    idx   = int(np.floor(target_coverage * len(confs))) - 1
    return float(confs[order[idx]])


def find_defect_thresholds(
    confs: np.ndarray,
    correct: np.ndarray,
    defect_ids: np.ndarray,
    n_defects: int,
    target_coverage: float = 0.80,
    split_name: str = "cal",
) -> dict:
    """
    Fit a per-defect confidence threshold on the cal split.
    For each defect subgroup, find the threshold achieving target_coverage
    within that group. Frozen-knob: must be called with split_name="cal".
    """
    from src.env import assert_no_rep_leakage
    assert_no_rep_leakage(split_name)
    thresholds = {}
    for d in range(-1, n_defects):
        mask = (defect_ids == d)
        if mask.sum() < 10:
            thresholds[int(d)] = 0.5
            continue
        t = find_global_threshold(confs[mask], correct[mask],
                                   target_coverage=target_coverage,
                                   split_name=split_name)
        thresholds[int(d)] = t
    return thresholds


# ── Comparison helpers ────────────────────────────────────────────────────────

def compare_policies(
    confs_raw: np.ndarray,
    confs_temp: np.ndarray,
    confs_defect: np.ndarray,
    correct: np.ndarray,
) -> dict:
    """
    Compute AURC for three policies: global raw, global temp-scaled,
    defect-aware (already applied via confs_defect).
    Returns dict with AURC values.
    """
    return dict(
        aurc_random    = 1 - float(correct.mean()),    # random-confidence floor
        aurc_global_raw   = aurc(confs_raw,    correct),
        aurc_global_temp  = aurc(confs_temp,   correct),
        aurc_defect_aware = aurc(confs_defect, correct),
    )


def risk_coverage_for_figure(confs: np.ndarray, correct: np.ndarray,
                              label: str, n_grid: int = 200) -> dict:
    """Return a downsampled curve for figure plotting."""
    cov, risk = risk_coverage_curve(confs, correct)
    step = max(1, len(cov) // n_grid)
    return dict(
        label    = label,
        coverage = cov[::step].tolist(),
        risk     = risk[::step].tolist(),
        aurc     = float(np.trapz(risk, cov)),
    )
