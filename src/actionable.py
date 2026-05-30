"""
E5 — Actionable Recovery metric (contribution C2).

ARR (Actionable Recovery Rate): among quality-unanswerable images, the fraction
where the top predicted defect matches an actual ground-truth flaw.

FRR (False-Refilm Rate): among answerable images, the fraction wrongly told to
refilm (i.e., the model predicts a defect on a clean photo).

These two metrics MUST always be reported together — ARR alone is gameable.
"""
import numpy as np
from src.stats import bootstrap_ci


DEFECT_TO_ACTION = {
    "blur":        "Hold the camera steady and refocus before retaking",
    "dark":        "Add light or move to a brighter area and retake",
    "bright":      "Reduce glare / avoid direct light and retake",
    "obstruction": "Move your finger or object off the lens and retake",
    "framing":     "Step back so the whole item is in frame and retake",
    "rotation":    "Rotate the phone upright and retake",
    "unrecognizable": "The photo is too unclear — retake in better conditions",
}

DEFECT_ORDER = ["blur", "bright", "dark", "obstruction", "framing",
                "rotation", "unrecognizable"]


def top_predicted_defect(pred_probs: np.ndarray, defect_names: list) -> list:
    """
    For each sample, return the name of the highest-probability predicted defect,
    or None if all probabilities are below 0.5.
    pred_probs: (N, n_defects) array of sigmoid probabilities.
    """
    top_idx = pred_probs.argmax(axis=1)
    top_conf = pred_probs.max(axis=1)
    results = []
    for idx, conf in zip(top_idx, top_conf):
        if conf >= 0.5:
            results.append(defect_names[idx])
        else:
            results.append(None)
    return results


def actionable_recovery_rate(
    pred_defect_probs: np.ndarray,
    gt_defects: np.ndarray,
    answerable: np.ndarray,
    defect_names: list,
) -> dict:
    """
    pred_defect_probs: (N, n_defects) sigmoid probabilities from E4.
    gt_defects:  (N, n_defects) binary ground-truth multi-hot array.
    answerable:  (N,) binary — 1 = answerable, 0 = quality-unanswerable.
    defect_names: list of defect labels in column order.

    Returns dict with ARR, FRR, per-defect breakdown, and bootstrap CIs.
    """
    pred_top = top_predicted_defect(pred_defect_probs, defect_names)
    n = len(answerable)

    # ── ARR: among unanswerable images ───────────────────────────────────────
    unanswerable_mask = (answerable == 0)
    arr_hits  = []
    for i in np.where(unanswerable_mask)[0]:
        top = pred_top[i]
        if top is None:
            # No predicted defect — no actionable advice → not a hit
            arr_hits.append(0)
        else:
            defect_idx = defect_names.index(top)
            hit = int(gt_defects[i, defect_idx] == 1)
            arr_hits.append(hit)
    arr_hits = np.array(arr_hits, float)
    ARR = float(arr_hits.mean()) if len(arr_hits) > 0 else float("nan")

    # ── FRR: among answerable images ─────────────────────────────────────────
    answerable_mask = (answerable == 1)
    frr_hits = []
    for i in np.where(answerable_mask)[0]:
        top = pred_top[i]
        # Any predicted defect on an answerable photo = wrong refilm instruction
        frr_hits.append(1 if top is not None else 0)
    frr_hits = np.array(frr_hits, float)
    FRR = float(frr_hits.mean()) if len(frr_hits) > 0 else float("nan")

    # ── Per-defect ARR breakdown ──────────────────────────────────────────────
    per_defect = {}
    for d_name in defect_names:
        d_idx = defect_names.index(d_name)
        gt_d  = gt_defects[:, d_idx]
        # Unanswerable AND this defect is the GT flaw
        mask_d = unanswerable_mask & (gt_d == 1)
        if mask_d.sum() == 0:
            per_defect[d_name] = dict(ARR=float("nan"), n=0)
            continue
        hits_d = []
        for i in np.where(mask_d)[0]:
            top = pred_top[i]
            hits_d.append(1 if top == d_name else 0)
        per_defect[d_name] = dict(
            ARR=float(np.mean(hits_d)),
            n=int(mask_d.sum()),
        )

    # ── Bootstrap CIs ────────────────────────────────────────────────────────
    def _arr_fn(answerable_b, gt_b, probs_b):
        una = (answerable_b == 0)
        if una.sum() == 0:
            return float("nan")
        top_b = top_predicted_defect(probs_b, defect_names)
        hits = [int(gt_b[i, defect_names.index(top_b[i])] == 1)
                if top_b[i] is not None else 0
                for i in np.where(una)[0]]
        return float(np.mean(hits))

    # Simple bootstrap
    rng = np.random.default_rng(42)
    arr_boots, frr_boots = [], []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        ans_b  = answerable[idx]
        gt_b   = gt_defects[idx]
        prob_b = pred_defect_probs[idx]
        top_b  = top_predicted_defect(prob_b, defect_names)

        una = (ans_b == 0)
        if una.sum() > 0:
            hits_a = [int(gt_b[k, defect_names.index(top_b[k])] == 1)
                      if top_b[k] is not None else 0
                      for k in np.where(una)[0]]
            arr_boots.append(float(np.mean(hits_a)))

        ans = (ans_b == 1)
        if ans.sum() > 0:
            hits_f = [1 if top_b[k] is not None else 0
                      for k in np.where(ans)[0]]
            frr_boots.append(float(np.mean(hits_f)))

    arr_lo, arr_hi = np.percentile(arr_boots, [2.5, 97.5]) if arr_boots else (float("nan"), float("nan"))
    frr_lo, frr_hi = np.percentile(frr_boots, [2.5, 97.5]) if frr_boots else (float("nan"), float("nan"))

    return dict(
        ARR=ARR, ARR_ci95=(float(arr_lo), float(arr_hi)),
        FRR=FRR, FRR_ci95=(float(frr_lo), float(frr_hi)),
        per_defect=per_defect,
        defect_to_action=DEFECT_TO_ACTION,
        n_unanswerable=int(unanswerable_mask.sum()),
        n_answerable=int(answerable_mask.sum()),
    )
