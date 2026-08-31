"""E4b - Guidance-threshold selection and sensitivity (CPU, cached data).

The reported guidance policy uses a fixed 0.5 probability cutoff for issuing
retake guidance (review Critical #5: arbitrary, especially for rare defects
with uncalibrated multi-label scores). E4b addresses this with cal-only
threshold selection and a full sweep:

  1. Per-label thresholds selected on the calibration split (max F1 per label).
  2. A global-cutoff sweep tau in {0.05..0.95}: GDMR and AIRB on the report
     split as a function of tau, so the 0.5 operating point is visible inside
     its sensitivity curve rather than presented as a magic number.
  3. GDMR/AIRB re-evaluated with the cal-selected per-label thresholds
     (top defect = argmax of prob/threshold ratio among labels above their
     own threshold), for a cost-free comparison against the 0.5 policy.

Needs: master.parquet, split_ids (E1), defect_logits_{bb}.npy (E4). No GPU.
"""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, progress, resultlog
from src.data_assembly import QUALITY_FLAWS

EXP = "E4B"
RESULTS_E4B = os.path.join(config.RESULTS, "E4b_thresholds")
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)


def required_artifacts():
    return [os.path.join(RESULTS_E4B, f"thresholds_{bb}.json")
            for bb in config.BACKBONES]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _gdmr_airb(probs, gt_defects, answerable, top_fn):
    """Generic GDMR/AIRB for a policy given a top-defect chooser."""
    top = top_fn(probs)
    una = answerable == 0
    ans = answerable == 1
    hits = [int(gt_defects[i, top[i]] == 1) if top[i] is not None else 0
            for i in np.where(una)[0]]
    gdmr = float(np.mean(hits)) if hits else float("nan")
    airb = float(np.mean([top[i] is not None for i in np.where(ans)[0]]))
    return gdmr, airb


def _top_global(tau):
    def fn(probs):
        idx = probs.argmax(axis=1)
        mx = probs.max(axis=1)
        return [int(i) if m >= tau else None for i, m in zip(idx, mx)]
    return fn


def _top_per_label(taus):
    taus = np.asarray(taus, dtype=np.float64)

    def fn(probs):
        ratio = probs / taus[None, :]
        above = probs >= taus[None, :]
        out = []
        for r, a in zip(ratio, above):
            if not a.any():
                out.append(None)
            else:
                r_masked = np.where(a, r, -np.inf)
                out.append(int(r_masked.argmax()))
        return out
    return fn


def main():
    progress.install_error_hook("E4b threshold sensitivity")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E4B, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E4B, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E4B)
        return

    pbar = progress.notebook_bar("E4b thresholds", total=1 + len(config.BACKBONES))

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]
    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]

    gt_cal = master.iloc[val_idx[cal_pos]][defect_cols].values
    gt_rep = master.iloc[val_idx[rep_pos]][defect_cols].values
    ans_rep = master.iloc[val_idx[rep_pos]]["answerable"].values.astype(int)
    progress.step(pbar, "cached E1 data loaded")

    from sklearn.metrics import f1_score

    all_results = {}
    for bb in config.BACKBONES:
        out_json = os.path.join(RESULTS_E4B, f"thresholds_{bb}.json")
        cached = None if config.FORCE_RERUN else expstate.load_json_valid(out_json)
        if cached is not None:
            all_results[bb] = cached
            progress.step(pbar, f"{bb} cache reused")
            continue

        logits_path = os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy")
        assert os.path.exists(logits_path), f"Run E4 first! Missing: {logits_path}"
        logits = np.load(logits_path)
        # E4 caches (n_val, 7) rep-and-cal val-order logits or (n_rep, 7);
        # handle both by length.
        if logits.shape[0] == len(val_idx):
            probs_cal = _sigmoid(logits[cal_pos])
            probs_rep = _sigmoid(logits[rep_pos])
        elif logits.shape[0] == len(rep_pos):
            raise AssertionError(
                "defect_logits only cover the report split; E4b needs cal-split "
                "probabilities for threshold selection. Re-run E4 with cal logits cached.")
        else:
            probs_cal = _sigmoid(logits[cal_pos])
            probs_rep = _sigmoid(logits[rep_pos])

        # 1. per-label cal-selected thresholds (max F1 on cal).
        env.assert_no_rep_leakage("cal")
        grid = np.linspace(0.01, 0.99, 99)
        taus = []
        for li in range(len(DEFECT_NAMES)):
            scores = [f1_score(gt_cal[:, li], (probs_cal[:, li] >= t).astype(int),
                               zero_division=0) for t in grid]
            taus.append(float(grid[int(np.argmax(scores))]))
        print(f"[E4b] {bb} cal-selected per-label thresholds: "
              + "  ".join(f"{n}={t:.2f}" for n, t in zip(DEFECT_NAMES, taus)))

        # 2. global-cutoff sweep on rep.
        sweep = []
        for tau in SWEEP:
            g, a = _gdmr_airb(probs_rep, gt_rep, ans_rep, _top_global(tau))
            sweep.append({"tau": float(tau), "GDMR": g, "AIRB": a})

        # 3. policies compared at fixed operating points.
        g05, a05 = _gdmr_airb(probs_rep, gt_rep, ans_rep, _top_global(0.5))
        gpl, apl = _gdmr_airb(probs_rep, gt_rep, ans_rep, _top_per_label(taus))

        result = {
            "backbone": bb,
            "per_label_thresholds": dict(zip(DEFECT_NAMES, taus)),
            "sweep": sweep,
            "policy_global_0.5": {"GDMR": g05, "AIRB": a05},
            "policy_per_label_cal": {"GDMR": gpl, "AIRB": apl},
        }
        expstate.write_json_atomic(out_json, result)
        all_results[bb] = result
        print(f"[E4b] {bb}: 0.5 policy GDMR/AIRB={g05:.4f}/{a05:.4f}  "
              f"per-label policy={gpl:.4f}/{apl:.4f}")
        progress.step(pbar, f"{bb} thresholds computed")

    resultlog.log_run(EXP, metrics=all_results,
                      params={"backbones": config.BACKBONES,
                              "sweep": SWEEP.tolist()},
                      results_dir=RESULTS_E4B, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E4B, artifacts=required_artifacts())
    pbar.close()
    print("[E4b DONE] Threshold sensitivity ready for the guidance section.")
