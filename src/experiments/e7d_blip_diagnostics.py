"""E7d - Selective-prediction diagnostics on the SECOND VQA model (CPU).

Reruns the full E7b/E7c method battery against BLIP confidence from E6b:

  per backbone: defect-conditioned group temperature, learned
  confidence+predicted-defects, triage score alone, confidence+triage,
  full stack (confidence+triage+predicted defects); plus the
  backbone-independent oracle (confidence + ground-truth defects).

Also computes refusal-gated ARR/FRR at 90% target coverage using the BLIP
confidence gate, and applies BH-FDR within this family of tests.

If the ViLT conclusions replicate on BLIP, the paper's claim upgrades from
"for one weak VQA model" to "across two frozen VQA models of different
architectures and confidence types (discriminative max-softmax and
generative sequence probability)".
"""
import json
import os

import numpy as np
import pandas as pd

from src import (actionable, calibration, config, env, expstate, progress,
                 resultlog, selective, stats)
from src.data_assembly import QUALITY_FLAWS
from src.experiments.e6b_vqaconf_blip import out_path as blip_parquet_path
from src.experiments.e7c_risk_signals import (_compare_scores, _fit_lr_score,
                                              _logit, _sigmoid, _triage_scores)

EXP = "E7D"
RESULTS_E7D = os.path.join(config.RESULTS, "E7d_blip_diagnostics")
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
N_DEFECTS = len(DEFECT_NAMES)
GATE_COVERAGE = 0.9


def required_artifacts():
    return [os.path.join(RESULTS_E7D, f"diagnostics_{bb}.json")
            for bb in config.BACKBONES] + [os.path.join(RESULTS_E7D, "summary.json")]


def main():
    progress.install_error_hook("E7d BLIP diagnostics")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E7D, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E7D, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E7D)
        return

    blip_path = blip_parquet_path()
    if not os.path.exists(blip_path):
        raise FileNotFoundError(f"Run E6b first; missing {blip_path}")

    pbar = progress.notebook_bar("E7d BLIP diagnostics", total=4 + len(config.BACKBONES))
    progress.step(pbar, "Environment checked: CPU-only diagnostics")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    preds = pd.read_parquet(blip_path)
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]

    val_preds = preds[preds["split"] == "val"].reset_index(drop=True)
    if len(val_preds) != len(val_idx):
        raise ValueError(f"val alignment mismatch: master={len(val_idx)} "
                         f"blip={len(val_preds)}")
    conf_cal = val_preds.iloc[cal_pos]["confidence"].values.astype(np.float64)
    conf_rep = val_preds.iloc[rep_pos]["confidence"].values.astype(np.float64)
    corr_cal = (val_preds.iloc[cal_pos]["correct"].values > 0).astype(int)
    corr_rep = (val_preds.iloc[rep_pos]["correct"].values > 0).astype(int)

    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    Y_val = master.iloc[val_idx][defect_cols].values.astype(np.float32)
    Y_cal, Y_rep = Y_val[cal_pos], Y_val[rep_pos]
    ans_rep = master.iloc[val_idx[rep_pos]]["answerable"].values.astype(int)

    global_aurc = selective.aurc(conf_rep, corr_rep)
    print(f"[E7d] BLIP: rep accuracy={corr_rep.mean():.4f}  "
          f"mean confidence={conf_rep.mean():.4f}  "
          f"global AURC={global_aurc:.4f}  (random={1 - corr_rep.mean():.4f})")
    progress.step(pbar, "BLIP predictions and labels loaded")

    all_results = {}
    oracle_result = None
    for bb in config.BACKBONES:
        out_json = os.path.join(RESULTS_E7D, f"diagnostics_{bb}.json")
        if os.path.exists(out_json) and not config.FORCE_RERUN:
            with open(out_json) as f:
                all_results[bb] = json.load(f)
            print(f"[E7d] cache hit: {out_json}")
            progress.step(pbar, f"{bb} cache reused")
            continue

        defect_logits_val = np.load(
            os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy"))
        prob_cal = _sigmoid(defect_logits_val[cal_pos])
        prob_rep = _sigmoid(defect_logits_val[rep_pos])
        top_cal = prob_cal.argmax(axis=1)
        top_rep = prob_rep.argmax(axis=1)

        triage_val = _triage_scores(bb, master, val_idx)
        tri_cal, tri_rep = triage_val[cal_pos], triage_val[rep_pos]

        # 1. defect-conditioned group temperature (E7-style)
        scalers = calibration.defect_aware_calibration(
            conf_cal, corr_cal, top_cal, N_DEFECTS, split_name="cal")
        score_group_temp = calibration.apply_defect_aware_calibration(
            conf_rep, top_rep, scalers)
        m_group = _compare_scores(corr_rep, conf_rep, score_group_temp)

        # 2. learned confidence + predicted defects (E7b-style)
        Xc = np.column_stack([_logit(conf_cal), prob_cal])
        Xr = np.column_stack([_logit(conf_rep), prob_rep])
        m_pred = _compare_scores(corr_rep, conf_rep,
                                 _fit_lr_score(Xc, corr_cal, Xr))

        # 3. triage alone / 4. confidence+triage / 5. full stack (E7c-style)
        m_triage = _compare_scores(corr_rep, conf_rep, tri_rep)
        Xc2 = np.column_stack([_logit(conf_cal), tri_cal])
        Xr2 = np.column_stack([_logit(conf_rep), tri_rep])
        m_combo = _compare_scores(corr_rep, conf_rep,
                                  _fit_lr_score(Xc2, corr_cal, Xr2))
        Xc3 = np.column_stack([_logit(conf_cal), tri_cal, prob_cal])
        Xr3 = np.column_stack([_logit(conf_rep), tri_rep, prob_rep])
        m_full = _compare_scores(corr_rep, conf_rep,
                                 _fit_lr_score(Xc3, corr_cal, Xr3))

        # oracle (backbone-independent): confidence + ground-truth defects
        if oracle_result is None:
            Xco = np.column_stack([_logit(conf_cal), Y_cal, Y_cal.sum(axis=1)])
            Xro = np.column_stack([_logit(conf_rep), Y_rep, Y_rep.sum(axis=1)])
            oracle_result = _compare_scores(corr_rep, conf_rep,
                                            _fit_lr_score(Xco, corr_cal, Xro))

        # refusal-gated recovery with the BLIP gate at 90% target coverage
        env.assert_no_rep_leakage("cal")
        tau = float(np.quantile(conf_cal, 1.0 - GATE_COVERAGE))
        refused = conf_rep < tau
        top_named = actionable.top_predicted_defect(prob_rep, DEFECT_NAMES)
        retake = np.array([t is not None for t in top_named])
        ans_mask, una_mask = ans_rep == 1, ans_rep == 0
        gated_frr = float((refused & retake & ans_mask).sum() / max(ans_mask.sum(), 1))
        ref_una = np.where(refused & una_mask)[0]
        hits = [int(Y_rep[i, DEFECT_NAMES.index(top_named[i])] == 1)
                if top_named[i] is not None else 0 for i in ref_una]
        gated_arr = float(np.mean(hits)) if hits else float("nan")

        result = {
            "backbone": bb,
            "vqa_model": config.VQA_MODEL_ID_2,
            "global_confidence_aurc": float(global_aurc),
            "rep_accuracy": float(corr_rep.mean()),
            "methods": {
                "predicted_defect_group_temperature": m_group,
                "predicted_defect_risk_model": m_pred,
                "triage_score_alone": m_triage,
                "confidence_plus_triage": m_combo,
                "confidence_plus_triage_plus_defects": m_full,
                "oracle_gt_defect_risk_model": oracle_result,
            },
            "gated_recovery_cov0.9": {
                "tau": tau,
                "coverage_achieved": float((~refused).mean()),
                "risk_on_answered": float(1 - corr_rep[~refused].mean()),
                "gated_ARR": gated_arr,
                "gated_FRR": gated_frr,
                "n_refused_unanswerable": int(len(ref_una)),
            },
        }
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        all_results[bb] = result
        print(f"[E7d] {bb}: pred-defects {m_pred['improvement_vs_global']:+.5f} "
              f"(p={m_pred['p']:.3f})  full-stack {m_full['improvement_vs_global']:+.5f} "
              f"(p={m_full['p']:.3f})  gated ARR={gated_arr:.3f} FRR={gated_frr:.3f}")
        progress.step(pbar, f"{bb} BLIP diagnostics done")

    # BH-FDR within the E7d family (oracle counted once)
    tests = []
    oracle_added = False
    for bb, res in all_results.items():
        for name, m in res["methods"].items():
            if name == "oracle_gt_defect_risk_model":
                if oracle_added:
                    continue
                oracle_added = True
                label = name
            else:
                label = f"{name}[{bb}]"
            tests.append({"test": label, "p": float(m["p"]),
                          "improvement": float(m["improvement_vs_global"])})
    reject = stats.benjamini_hochberg([t["p"] for t in tests]) if tests else []
    for t, r in zip(tests, reject):
        t["bh_fdr_reject_at_0.05"] = bool(r)
    n_pos_sig = sum(1 for t in tests
                    if t.get("bh_fdr_reject_at_0.05") and t["improvement"] > 0)
    print(f"\n[E7d] BH-FDR over {len(tests)} BLIP tests: "
          f"{int(np.sum(reject))} significant, {n_pos_sig} of them POSITIVE improvements")

    summary = {
        "vqa_model": config.VQA_MODEL_ID_2,
        "global_confidence_aurc": float(global_aurc),
        "rep_accuracy": float(corr_rep.mean()),
        "backbones": {bb: all_results[bb]["methods"] for bb in all_results},
        "gated_recovery": {bb: all_results[bb]["gated_recovery_cov0.9"]
                           for bb in all_results},
        "bh_fdr": {"n_tests": len(tests),
                   "n_significant": int(np.sum(reject)) if len(tests) else 0,
                   "n_positive_significant": n_pos_sig,
                   "tests": tests},
    }
    with open(os.path.join(RESULTS_E7D, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    resultlog.log_run(EXP, metrics=summary,
                      params={"backbones": config.BACKBONES,
                              "vqa_model": config.VQA_MODEL_ID_2,
                              "N_BOOT": config.N_BOOT},
                      results_dir=RESULTS_E7D, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E7D, artifacts=required_artifacts())
    progress.step(pbar, "E7d summary + BH-FDR written")
    pbar.close()
    print("[E7d DONE] If conclusions replicate, the paper's claim holds across "
          "two frozen VQA models with different confidence types.")
