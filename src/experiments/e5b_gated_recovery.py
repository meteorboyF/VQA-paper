"""E5b - Refusal-gated actionable recovery (CPU, cached data).

The E5 False Refilm Rate (~0.78) is an UNGATED burden proxy: it counts any
predicted defect on any answerable image, even though the paper's deployment
design only issues retake guidance AFTER the confidence gate refuses. This
experiment computes the deployment-aligned versions:

  For coverage targets c in {0.9, 0.8, 0.7}:
    - refuse when VQA confidence < tau_c, tau_c = (1-c) quantile of CAL confs
    - gated FRR  = fraction of answerable rep examples that are refused AND
                   told to retake (burden per answerable interaction)
    - gated ARR  = among refused unanswerable rep examples, fraction whose
                   top predicted defect matches a ground-truth defect
    - plus refusal rates and achieved coverage/risk for context

All thresholds come from cal only (frozen-knob rule). Needs Drive caches:
master.parquet, split_ids, E6 predictions, defect_logits_{bb}.npy. No GPU.
"""
import json
import os

import numpy as np
import pandas as pd

from src import actionable, config, env, expstate, progress, resultlog
from src.data_assembly import QUALITY_FLAWS

EXP = "E5B"
RESULTS_E5B = os.path.join(config.RESULTS, "E5b_gated_recovery")
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
COVERAGE_TARGETS = (0.9, 0.8, 0.7)


def required_artifacts():
    return [os.path.join(RESULTS_E5B, f"gated_arr_frr_{bb}.json")
            for bb in config.BACKBONES]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _gated_metrics(conf_rep, corr_rep, ans_rep, retake_suggested, gt_defects,
                   top_pred, tau):
    refused = conf_rep < tau
    answered = ~refused
    ans_mask = ans_rep == 1
    una_mask = ans_rep == 0

    # gated FRR: answerable AND refused AND told to retake
    frr_hits = (refused & retake_suggested & ans_mask)
    gated_frr = float(frr_hits.sum() / max(ans_mask.sum(), 1))

    # gated ARR: among refused unanswerable examples, top predicted defect
    # matches a GT defect (same hit rule as E5, restricted to refusals)
    ref_una = np.where(refused & una_mask)[0]
    arr_hits = []
    for i in ref_una:
        top = top_pred[i]
        if top is None:
            arr_hits.append(0)
        else:
            arr_hits.append(int(gt_defects[i, DEFECT_NAMES.index(top)] == 1))
    gated_arr = float(np.mean(arr_hits)) if arr_hits else float("nan")

    return {
        "tau": float(tau),
        "coverage_achieved": float(answered.mean()),
        "risk_on_answered": float(1 - corr_rep[answered].mean()) if answered.sum() else float("nan"),
        "refusal_rate_overall": float(refused.mean()),
        "refusal_rate_answerable": float(refused[ans_mask].mean()),
        "refusal_rate_unanswerable": float(refused[una_mask].mean()),
        "gated_FRR": gated_frr,
        "retake_given_refused_answerable": float(
            retake_suggested[refused & ans_mask].mean()) if (refused & ans_mask).sum() else float("nan"),
        "gated_ARR": gated_arr,
        "n_refused_unanswerable": int(len(ref_una)),
    }


def main():
    progress.install_error_hook("E5b gated recovery")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E5B, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E5B, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E5B)
        return

    pbar = progress.notebook_bar("E5b gated recovery", total=2 + len(config.BACKBONES))
    progress.step(pbar, "Environment checked: CPU-only, cached data")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    vqa_preds = pd.read_parquet(os.path.join(config.RESULTS_E6, "vqa_predictions.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]

    val_preds = vqa_preds[vqa_preds["split"] == "val"].reset_index(drop=True)
    conf_cal = val_preds.iloc[cal_pos]["confidence"].values.astype(np.float64)
    conf_rep = val_preds.iloc[rep_pos]["confidence"].values.astype(np.float64)
    corr_rep = (val_preds.iloc[rep_pos]["correct"].values > 0).astype(int)

    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    rep_master = master.iloc[val_idx[rep_pos]]
    gt_defects = rep_master[defect_cols].values
    ans_rep = rep_master["answerable"].values.astype(int)
    progress.step(pbar, "cached E1/E6 data loaded")

    # Thresholds are chosen on cal only.
    env.assert_no_rep_leakage("cal")
    taus = {c: float(np.quantile(conf_cal, 1.0 - c)) for c in COVERAGE_TARGETS}
    print(f"[E5b] cal-selected confidence thresholds: "
          + "  ".join(f"cov={c:.0%}: tau={t:.4f}" for c, t in taus.items()))

    all_results = {}
    for bb in config.BACKBONES:
        out_path = os.path.join(RESULTS_E5B, f"gated_arr_frr_{bb}.json")
        if os.path.exists(out_path) and not config.FORCE_RERUN:
            with open(out_path) as f:
                all_results[bb] = json.load(f)
            print(f"[E5b] cache hit: {out_path}")
            progress.step(pbar, f"{bb} cache reused")
            continue

        logits_path = os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy")
        assert os.path.exists(logits_path), f"Run E4 first! Missing: {logits_path}"
        probs_rep = _sigmoid(np.load(logits_path)[rep_pos])
        top_pred = actionable.top_predicted_defect(probs_rep, DEFECT_NAMES)
        retake_suggested = np.array([t is not None for t in top_pred])

        per_coverage = {}
        for c in COVERAGE_TARGETS:
            per_coverage[f"coverage_{c:.1f}"] = _gated_metrics(
                conf_rep, corr_rep, ans_rep, retake_suggested,
                gt_defects, top_pred, taus[c])

        # Ungated numbers for side-by-side context (matches E5 definitions)
        ungated_frr = float(retake_suggested[ans_rep == 1].mean())

        result = {
            "backbone": bb,
            "ungated_FRR_for_reference": ungated_frr,
            "per_coverage": per_coverage,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results[bb] = result

        print(f"\n[E5b] {bb}: ungated FRR={ungated_frr:.4f}")
        for key, m in per_coverage.items():
            print(f"  {key}: refusal={m['refusal_rate_overall']:.3f}  "
                  f"gated FRR={m['gated_FRR']:.4f}  gated ARR={m['gated_ARR']:.4f}  "
                  f"(n refused unanswerable={m['n_refused_unanswerable']})")
        progress.step(pbar, f"{bb} gated recovery computed")

    resultlog.log_run(EXP, metrics=all_results,
                      params={"backbones": config.BACKBONES,
                              "coverage_targets": list(COVERAGE_TARGETS),
                              "taus": taus},
                      results_dir=RESULTS_E5B, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E5B, artifacts=required_artifacts())
    pbar.close()
    print("[E5b DONE] Use gated FRR in Sec. VI-D of the paper alongside the "
          "ungated burden proxy.")
