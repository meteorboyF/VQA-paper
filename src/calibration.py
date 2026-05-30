"""
Calibration utilities (E7).

temperature_scale  — 1-parameter LBFGS on NLL (cal split only).
ece                — Expected Calibration Error.
brier_score        — Brier score for binary predictions.
defect_aware_calibration — per-defect temperature scalers.

Frozen-knob rule: temperature T is ALWAYS selected on cal, never on rep.
All threshold-selection functions assert split=="cal" via env.assert_no_rep_leakage.
"""
import numpy as np
import torch
import torch.nn as nn

from src.env import assert_no_rep_leakage


# ── Temperature scaling ───────────────────────────────────────────────────────

def temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray,
    split_name: str = "cal",
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """
    Fit a single temperature scalar T on the cal split.
    Raises if called with rep/test data (frozen-knob assertion).
    Returns scalar T.
    """
    assert_no_rep_leakage(split_name)

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    T = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / T.clamp(min=1e-6)
        if scaled.dim() == 1:
            # binary case
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled, labels_t.float()
            )
        else:
            loss = nn.functional.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.item())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Divide logits by temperature; return calibrated probabilities."""
    scaled = np.array(logits, dtype=np.float64) / max(T, 1e-6)
    if scaled.ndim == 1:
        return 1.0 / (1.0 + np.exp(-scaled))   # sigmoid for binary
    exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


# ── ECE ───────────────────────────────────────────────────────────────────────

def ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (confidence-weighted)."""
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        acc_bin  = correct[mask].mean()
        conf_bin = confs[mask].mean()
        e += mask.mean() * abs(acc_bin - conf_bin)
    return float(e)


def ece_diagram_data(confs: np.ndarray, correct: np.ndarray,
                     n_bins: int = 15) -> dict:
    """Return per-bin data for plotting reliability diagrams."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_frac = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            bin_acc.append(float("nan"))
            bin_conf.append((lo + hi) / 2)
            bin_frac.append(0.0)
        else:
            bin_acc.append(float(correct[mask].mean()))
            bin_conf.append(float(confs[mask].mean()))
            bin_frac.append(float(mask.mean()))
    return dict(bin_acc=bin_acc, bin_conf=bin_conf, bin_frac=bin_frac,
                bin_edges=bins.tolist())


# ── Brier score ───────────────────────────────────────────────────────────────

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels.astype(float)) ** 2))


# ── Per-defect temperature scaling ───────────────────────────────────────────

def defect_aware_calibration(
    confs: np.ndarray,
    correct: np.ndarray,
    defect_ids: np.ndarray,
    n_defects: int,
    split_name: str = "cal",
) -> dict:
    """
    Fit a separate temperature scalar per defect group on the cal split.
    confs     — raw (pre-calibration) VQA confidence per sample
    correct   — binary correctness per sample
    defect_ids — integer defect label per sample (−1 = no defect)
    Returns dict: {defect_id: T_float, ...} plus "global" key for the fallback.

    Frozen-knob: must be called with split_name="cal".
    """
    assert_no_rep_leakage(split_name)
    scalers = {}
    for d in range(-1, n_defects):
        mask = (defect_ids == d)
        if mask.sum() < 10:     # too few samples to fit
            scalers[int(d)] = 1.0
            continue
        logits_d = np.log(confs[mask].clip(1e-6, 1 - 1e-6) /
                           (1 - confs[mask].clip(1e-6, 1 - 1e-6)))
        T_d = temperature_scale(logits_d, correct[mask].astype(int),
                                 split_name=split_name)
        scalers[int(d)] = T_d
    # Fallback global temperature on everything
    logits_all = np.log(confs.clip(1e-6, 1 - 1e-6) /
                        (1 - confs.clip(1e-6, 1 - 1e-6)))
    scalers["global"] = temperature_scale(logits_all, correct.astype(int),
                                           split_name=split_name)
    return scalers


def apply_defect_aware_calibration(
    confs: np.ndarray,
    defect_ids: np.ndarray,
    scalers: dict,
) -> np.ndarray:
    """Apply per-defect temperature scalers to raw confidences."""
    out = np.array(confs, dtype=np.float64)
    for d, T in scalers.items():
        if d == "global":
            continue
        mask = (defect_ids == int(d))
        if not mask.any():
            continue
        logits_d = np.log(confs[mask].clip(1e-6, 1 - 1e-6) /
                          (1 - confs[mask].clip(1e-6, 1 - 1e-6)))
        out[mask] = apply_temperature(logits_d, T)
    return out
