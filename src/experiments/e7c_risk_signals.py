"""E7c - Risk-signal baselines for selective prediction (CPU, cached data).

Closes the main reviewer gap in the paper's headline claim. E7b showed that
DEFECT-derived signals do not beat global VQA confidence, but the paper says
"confidence is the strongest signal we evaluated" - and the most obvious
untested competitor is our own E3 answerability-triage score. This experiment
compares, per backbone:

  1. triage score alone                      (no confidence at all)
  2. learned risk: confidence + triage score (fit on cal)
  3. learned risk: confidence + triage score + predicted defects (full stack)

against global VQA confidence, with paired-bootstrap AURC tests. It also
applies Benjamini-Hochberg FDR correction across ALL selective-prediction
p-values from E7, E7b, and E7c, giving the paper one multiple-comparisons-
safe statement.

Needs Drive caches only: master.parquet, split_ids, E6 predictions,
emb_{bb}.npy, triage_{bb}.pt, defect_logits_{bb}.npy. No GPU.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, progress, resultlog, selective, stats

EXP = "E7C"
RESULTS_E7C = os.path.join(config.RESULTS, "E7c_risk_signals")
RESULTS_E7B = os.path.join(config.RESULTS, "E7b_diagnostics")


def required_artifacts():
    return [os.path.join(RESULTS_E7C, f"risk_signals_{bb}.json")
            for bb in config.BACKBONES] + [os.path.join(RESULTS_E7C, "summary.json")]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(p):
    p = np.asarray(p, dtype=np.float64).clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_lr_score(X_cal, y_cal, X_rep):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y_cal)) < 2:
        return np.full(len(X_rep), float(np.mean(y_cal)))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, solver="lbfgs"))
    clf.fit(X_cal, y_cal.astype(int))
    return clf.predict_proba(X_rep)[:, 1]


def _compare_scores(corr_rep, baseline_score, method_score):
    """AURC improvement = AURC(baseline) - AURC(method); positive is better."""
    aurc_method = selective.aurc(method_score, corr_rep)
    delta, lo, hi, p = stats.paired_bootstrap_delta(
        lambda y, s: selective.aurc(s, y),
        corr_rep, baseline_score, method_score, n_boot=config.N_BOOT)
    return {
        "aurc": float(aurc_method),
        "improvement_vs_global": float(delta),
        "ci_lo": float(lo), "ci_hi": float(hi), "p": float(p),
    }


def _triage_scores(bb, master, val_idx):
    """Recompute triage logits on the FULL val split from the saved seed-0
    model (E3 only cached rep logits, but the learned combiner needs cal)."""
    import torch
    from src import heads
    emb_path = os.path.join(config.ARTIFACTS, f"emb_{bb}.npy")
    model_path = os.path.join(config.ARTIFACTS, f"triage_{bb}.pt")
    for p in (emb_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Run E2/E3 first; missing {p}")
    emb = np.load(emb_path).astype(np.float32)
    X_val = emb[val_idx]
    model = heads.MLPHead(X_val.shape[1], 1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(X_val), 4096):
            batch = torch.tensor(X_val[i:i + 4096], dtype=torch.float32)
            out.append(model(batch).squeeze(-1).float().numpy())
    return np.concatenate(out)


def _collect_all_pvalues(e7c_results):
    """Gather every selective-prediction p-value from E7, E7b, and E7c."""
    tests = []
    for bb in config.BACKBONES:
        p7 = os.path.join(config.RESULTS_E7, f"aurc_comparison_{bb}.json")
        if os.path.exists(p7):
            with open(p7) as f:
                e7 = json.load(f)
            if "delta_aurc_p" in e7:
                tests.append({"family": "E7", "test": f"defect_conditioned_calibration[{bb}]",
                              "p": float(e7["delta_aurc_p"]),
                              "improvement": float(e7.get("delta_aurc", np.nan))})
    oracle_added = False
    for bb in config.BACKBONES:
        p7b = os.path.join(RESULTS_E7B, f"diagnostics_{bb}.json")
        if not os.path.exists(p7b):
            continue
        with open(p7b) as f:
            d = json.load(f)
        for name, m in d.get("methods", {}).items():
            if name == "oracle_gt_defect_risk_model":
                if oracle_added:
                    continue           # identical test repeated per backbone
                oracle_added = True
                label = "oracle_gt_defect_risk_model"
            else:
                label = f"{name}[{bb}]"
            tests.append({"family": "E7b", "test": label, "p": float(m["p"]),
                          "improvement": float(m["improvement_vs_global"])})
    for bb, res in e7c_results.items():
        for name, m in res["methods"].items():
            tests.append({"family": "E7c", "test": f"{name}[{bb}]", "p": float(m["p"]),
                          "improvement": float(m["improvement_vs_global"])})
    return tests


def main():
    progress.install_error_hook("E7c risk-signal baselines")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E7C, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E7C, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E7C)
        return

    pbar = progress.notebook_bar("E7c risk-signal baselines",
                                 total=3 + len(config.BACKBONES))
    progress.step(pbar, "Environment checked: CPU-only, cached data")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    vqa_preds = pd.read_parquet(os.path.join(config.RESULTS_E6, "vqa_predictions.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]

    val_preds = vqa_preds[vqa_preds["split"] == "val"].reset_index(drop=True)
    conf_cal = val_preds.iloc[cal_pos]["confidence"].values.astype(np.float64)
    conf_rep = val_preds.iloc[rep_pos]["confidence"].values.astype(np.float64)
    corr_cal = (val_preds.iloc[cal_pos]["correct"].values > 0).astype(int)
    corr_rep = (val_preds.iloc[rep_pos]["correct"].values > 0).astype(int)
    global_aurc = selective.aurc(conf_rep, corr_rep)
    print(f"[E7c] global confidence AURC={global_aurc:.4f}  "
          f"(n_cal={len(corr_cal)}, n_rep={len(corr_rep)})")
    progress.step(pbar, "cached E1/E6 data loaded")

    all_results = {}
    for bb in config.BACKBONES:
        out_path = os.path.join(RESULTS_E7C, f"risk_signals_{bb}.json")
        if os.path.exists(out_path) and not config.FORCE_RERUN:
            with open(out_path) as f:
                all_results[bb] = json.load(f)
            print(f"[E7c] cache hit: {out_path}")
            progress.step(pbar, f"{bb} cache reused")
            continue

        triage_val = _triage_scores(bb, master, val_idx)
        tri_cal, tri_rep = triage_val[cal_pos], triage_val[rep_pos]

        defect_logits_val = np.load(
            os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy"))
        dprob_cal = _sigmoid(defect_logits_val[cal_pos])
        dprob_rep = _sigmoid(defect_logits_val[rep_pos])

        # 1. Triage score alone (is the answerability head a better ranker?)
        m_triage = _compare_scores(corr_rep, conf_rep, tri_rep)

        # 2. Confidence + triage score, combined on cal
        X_cal2 = np.column_stack([_logit(conf_cal), tri_cal])
        X_rep2 = np.column_stack([_logit(conf_rep), tri_rep])
        m_combo = _compare_scores(
            corr_rep, conf_rep, _fit_lr_score(X_cal2, corr_cal, X_rep2))

        # 3. Full reliability-layer stack: confidence + triage + pred. defects
        X_cal3 = np.column_stack([_logit(conf_cal), tri_cal, dprob_cal])
        X_rep3 = np.column_stack([_logit(conf_rep), tri_rep, dprob_rep])
        m_full = _compare_scores(
            corr_rep, conf_rep, _fit_lr_score(X_cal3, corr_cal, X_rep3))

        result = {
            "backbone": bb,
            "global_confidence_aurc": float(global_aurc),
            "methods": {
                "triage_score_alone": m_triage,
                "confidence_plus_triage": m_combo,
                "confidence_plus_triage_plus_defects": m_full,
            },
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results[bb] = result
        print(f"[E7c] {bb}: triage-alone AURC={m_triage['aurc']:.4f} "
              f"(improvement {m_triage['improvement_vs_global']:+.5f}, p={m_triage['p']:.3f})  "
              f"conf+triage {m_combo['improvement_vs_global']:+.5f} (p={m_combo['p']:.3f})  "
              f"full-stack {m_full['improvement_vs_global']:+.5f} (p={m_full['p']:.3f})")
        progress.step(pbar, f"{bb} risk-signal comparison done")

    # ── BH-FDR across every selective-prediction test in E7 + E7b + E7c ──
    tests = _collect_all_pvalues(all_results)
    pvals = [t["p"] for t in tests]
    reject = stats.benjamini_hochberg(pvals) if pvals else np.array([])
    for t, r in zip(tests, reject):
        t["bh_fdr_reject_at_0.05"] = bool(r)
    n_sig = int(np.sum(reject)) if len(reject) else 0
    print(f"\n[E7c] BH-FDR over {len(tests)} selective-prediction tests: "
          f"{n_sig} remain significant at FDR 0.05")
    for t in tests:
        if t.get("bh_fdr_reject_at_0.05"):
            print(f"  significant: {t['family']} {t['test']} "
                  f"improvement={t['improvement']:+.5f} p={t['p']:.3f}")

    summary = {
        "global_confidence_aurc": float(global_aurc),
        "backbones": {bb: all_results[bb]["methods"] for bb in all_results},
        "bh_fdr": {
            "n_tests": len(tests),
            "n_significant_at_fdr_0.05": n_sig,
            "tests": tests,
        },
    }
    with open(os.path.join(RESULTS_E7C, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    resultlog.log_run(EXP, metrics=summary,
                      params={"backbones": config.BACKBONES, "N_BOOT": config.N_BOOT},
                      results_dir=RESULTS_E7C, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E7C, artifacts=required_artifacts())
    progress.step(pbar, "E7c summary + BH-FDR written")
    pbar.close()
    print("[E7c DONE] If no method beats confidence, the paper's claim can say "
          "'strongest among all signals evaluated, including our own triage head'.")
