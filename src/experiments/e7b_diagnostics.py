"""E7b - CPU-only diagnostics for defect-aware selective prediction.

This experiment does not train image models and does not spend GPU. It reuses
the cached E1/E4/E6 artifacts to answer a narrow question:

    Does defect information improve VQA selective prediction beyond global
    VQA confidence when the risk score is learned on the calibration split?

All learned knobs are fit on cal only and evaluated on rep only.
"""
import json
import os

import numpy as np
import pandas as pd

from src import calibration, config, env, expstate, progress, resultlog, selective, stats
from src.data_assembly import QUALITY_FLAWS

EXP = "E7B"
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
N_DEFECTS = len(DEFECT_NAMES)
RESULTS_E7B = os.path.join(config.RESULTS, "E7b_diagnostics")


def required_artifacts():
    return [os.path.join(RESULTS_E7B, f"diagnostics_{bb}.json")
            for bb in config.BACKBONES] + [os.path.join(RESULTS_E7B, "summary.json")]


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.asarray(p, dtype=np.float64).clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _entropy(probs):
    probs = np.asarray(probs, dtype=np.float64).clip(1e-8, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def _one_hot(ids, n):
    out = np.zeros((len(ids), n), dtype=np.float32)
    ok = (ids >= 0) & (ids < n)
    out[np.arange(len(ids))[ok], ids[ok]] = 1.0
    return out


def _multi_hot_to_top(Y):
    """Return top GT defect id, using -1 for rows with no positive GT defect."""
    ids = np.full(len(Y), -1, dtype=int)
    has = Y.sum(axis=1) > 0
    ids[has] = np.argmax(Y[has], axis=1)
    return ids


def _fit_lr_score(X_cal, y_cal, X_rep):
    """Fit a small logistic risk model on cal; return P(correct) on rep."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_cal)) < 2:
        return np.full(len(X_rep), float(np.mean(y_cal)))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    clf.fit(X_cal, y_cal.astype(int))
    return clf.predict_proba(X_rep)[:, 1]


def _compare_scores(corr_rep, baseline_score, method_score):
    """AURC improvement = AURC(baseline) - AURC(method), so positive is better."""
    aurc_base = selective.aurc(baseline_score, corr_rep)
    aurc_method = selective.aurc(method_score, corr_rep)
    delta, lo, hi, p = stats.paired_bootstrap_delta(
        lambda y, s: selective.aurc(s, y),
        corr_rep,
        baseline_score,
        method_score,
        n_boot=config.N_BOOT,
    )
    return {
        "aurc": float(aurc_method),
        "improvement_vs_global": float(delta),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p": float(p),
        "global_aurc": float(aurc_base),
    }


def _subgroup_breakdown(corr, conf, group_ids, names):
    rows = {}
    for gid, name in enumerate(names):
        mask = group_ids == gid
        if mask.sum() == 0:
            continue
        rows[name] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(corr[mask])),
            "mean_confidence": float(np.mean(conf[mask])),
            "aurc_global_conf": float(selective.aurc(conf[mask], corr[mask])),
        }
    no_def = group_ids == -1
    if no_def.sum():
        rows["no_positive_defect"] = {
            "n": int(no_def.sum()),
            "accuracy": float(np.mean(corr[no_def])),
            "mean_confidence": float(np.mean(conf[no_def])),
            "aurc_global_conf": float(selective.aurc(conf[no_def], corr[no_def])),
        }
    return rows


def _method_table(methods):
    ordered = []
    for name, m in methods.items():
        ordered.append({
            "method": name,
            "aurc": m["aurc"],
            "improvement_vs_global": m["improvement_vs_global"],
            "ci_lo": m["ci_lo"],
            "ci_hi": m["ci_hi"],
            "p": m["p"],
        })
    return sorted(ordered, key=lambda r: r["aurc"])


def main():
    progress.install_error_hook("E7b diagnostics")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E7B, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E7B, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E7B)
        return

    pbar = progress.notebook_bar("E7b diagnostics", total=3 + len(config.BACKBONES))
    progress.step(pbar, "Environment checked: CPU-only diagnostics")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    vqa_preds = pd.read_parquet(os.path.join(config.RESULTS_E6, "vqa_predictions.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))

    val_master = master[master["split"] == "val"].reset_index(drop=True)
    val_preds = vqa_preds[vqa_preds["split"] == "val"].reset_index(drop=True)
    if len(val_master) != len(val_preds):
        raise ValueError(f"val alignment mismatch: master={len(val_master)} preds={len(val_preds)}")

    conf_cal = val_preds.iloc[cal_pos]["confidence"].values.astype(np.float64)
    conf_rep = val_preds.iloc[rep_pos]["confidence"].values.astype(np.float64)
    corr_cal = (val_preds.iloc[cal_pos]["correct"].values > 0).astype(int)
    corr_rep = (val_preds.iloc[rep_pos]["correct"].values > 0).astype(int)

    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    Y_val = val_master[defect_cols].values.astype(np.float32)
    Y_cal = Y_val[cal_pos]
    Y_rep = Y_val[rep_pos]
    gt_top_cal = _multi_hot_to_top(Y_cal)
    gt_top_rep = _multi_hot_to_top(Y_rep)
    progress.step(pbar, "cached E1/E6 data loaded")

    global_aurc = selective.aurc(conf_rep, corr_rep)
    base_cal = _logit(conf_cal).reshape(-1, 1)
    base_rep = _logit(conf_rep).reshape(-1, 1)
    score_global_lr = _fit_lr_score(base_cal, corr_cal, base_rep)
    progress.step(pbar, "global learned baseline fitted")

    all_results = {}
    for bb in config.BACKBONES:
        out_path = os.path.join(RESULTS_E7B, f"diagnostics_{bb}.json")
        if os.path.exists(out_path) and not config.FORCE_RERUN:
            with open(out_path) as f:
                all_results[bb] = json.load(f)
            print(f"[E7b] cache hit: {out_path}")
            progress.step(pbar, f"{bb} cache reused")
            continue

        logits_path = os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy")
        if not os.path.exists(logits_path):
            raise FileNotFoundError(f"Run E4 first; missing {logits_path}")
        defect_logits_val = np.load(logits_path)
        prob_val = _sigmoid(defect_logits_val)
        prob_cal = prob_val[cal_pos]
        prob_rep = prob_val[rep_pos]
        top_cal = prob_cal.argmax(axis=1)
        top_rep = prob_rep.argmax(axis=1)

        pred_features_cal = np.column_stack([
            _logit(conf_cal),
            prob_cal,
            prob_cal.max(axis=1),
            _entropy(prob_cal),
            _one_hot(top_cal, N_DEFECTS),
        ])
        pred_features_rep = np.column_stack([
            _logit(conf_rep),
            prob_rep,
            prob_rep.max(axis=1),
            _entropy(prob_rep),
            _one_hot(top_rep, N_DEFECTS),
        ])
        score_pred_defect = _fit_lr_score(pred_features_cal, corr_cal, pred_features_rep)

        oracle_features_cal = np.column_stack([
            _logit(conf_cal),
            Y_cal,
            Y_cal.sum(axis=1),
            _one_hot(gt_top_cal, N_DEFECTS),
        ])
        oracle_features_rep = np.column_stack([
            _logit(conf_rep),
            Y_rep,
            Y_rep.sum(axis=1),
            _one_hot(gt_top_rep, N_DEFECTS),
        ])
        score_oracle_defect = _fit_lr_score(oracle_features_cal, corr_cal, oracle_features_rep)

        # Current E7's per-defect temperature approach, recomputed here.
        scalers = calibration.defect_aware_calibration(
            conf_cal, corr_cal, top_cal, N_DEFECTS, split_name="cal")
        score_group_temp = calibration.apply_defect_aware_calibration(
            conf_rep, top_rep, scalers)

        methods = {
            "global_logistic_confidence": _compare_scores(corr_rep, conf_rep, score_global_lr),
            "predicted_defect_risk_model": _compare_scores(corr_rep, conf_rep, score_pred_defect),
            "oracle_gt_defect_risk_model": _compare_scores(corr_rep, conf_rep, score_oracle_defect),
            "predicted_defect_group_temperature": _compare_scores(corr_rep, conf_rep, score_group_temp),
        }
        result = {
            "backbone": bb,
            "n_cal": int(len(corr_cal)),
            "n_rep": int(len(corr_rep)),
            "rep_accuracy": float(np.mean(corr_rep)),
            "global_confidence_aurc": float(global_aurc),
            "methods": methods,
            "ranked_methods": _method_table(methods),
            "predicted_defect_subgroups": _subgroup_breakdown(corr_rep, conf_rep, top_rep, DEFECT_NAMES),
            "gt_top_defect_subgroups": _subgroup_breakdown(corr_rep, conf_rep, gt_top_rep, DEFECT_NAMES),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results[bb] = result

        best = result["ranked_methods"][0]
        print(f"[E7b] {bb}: best={best['method']} AURC={best['aurc']:.4f} "
              f"improvement={best['improvement_vs_global']:.5f} "
              f"CI=[{best['ci_lo']:.5f},{best['ci_hi']:.5f}] p={best['p']:.4f}")
        progress.step(pbar, f"{bb} diagnostics completed")

    summary = {
        "global_confidence_aurc": float(global_aurc),
        "backbones": {
            bb: {
                "best_method": all_results[bb]["ranked_methods"][0],
                "predicted_defect_risk_model":
                    all_results[bb]["methods"]["predicted_defect_risk_model"],
                "oracle_gt_defect_risk_model":
                    all_results[bb]["methods"]["oracle_gt_defect_risk_model"],
            }
            for bb in config.BACKBONES
        },
    }
    summary_path = os.path.join(RESULTS_E7B, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    resultlog.log_run(EXP, metrics=summary,
                      params={"backbones": config.BACKBONES, "N_BOOT": config.N_BOOT},
                      results_dir=RESULTS_E7B, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E7B, artifacts=required_artifacts())
    progress.step(pbar, "E7b result logged")
    pbar.close()
    print("[E7b DONE] Inspect summary.json before deciding whether to run E9.")
