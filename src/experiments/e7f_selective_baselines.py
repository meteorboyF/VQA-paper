"""E7f - Extra selective-prediction baselines from cached predictions (CPU).

Review Section 4 asks for standard uncertainty baselines. Two caveats keep
this experiment honest about what the caches contain:

  - ViLT max-softmax IS the max-probability baseline (already the paper's
    global-confidence reference). Entropy and top-two margin require the full
    answer distribution, which E6 does not cache; if a re-harvest adds a
    `topk_probs` column, this script picks it up automatically.
  - For BLIP, answer-length and per-token statistics ARE computable from the
    cached predictions and are exactly the kind of shallow generative-signal
    baseline the review requests.

Scores evaluated (vs each answerer's own global confidence, paired bootstrap,
BH-FDR within family):

  blip: neg_answer_len_chars, neg_answer_len_words,
        lr(conf, len_chars, len_words)  [cal-fit logistic combiner]
  vilt: lr(conf, question_len_words)    [question-length control]
  both: entropy / top2-margin when `topk_probs` exists in the parquet.

Needs: E1 splits + E6/E6b predictions. No GPU.
"""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, progress, resultlog, selective
from src.stats import benjamini_hochberg, paired_bootstrap_delta

EXP = "E7F"
RESULTS_E7F = os.path.join(config.RESULTS, "E7f_selective_baselines")
GATES = {"vilt": ("E6_vqaconf", "vqa_predictions.parquet"),
         "blip": ("E6b_vqaconf_blip", "vqa_predictions_blip.parquet")}


def required_artifacts():
    return [os.path.join(RESULTS_E7F, f"baselines_{gate}.json") for gate in GATES]


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


def main():
    progress.install_error_hook("E7f selective baselines")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E7F, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E7F, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E7F)
        return

    pbar = progress.notebook_bar("E7f selective baselines", total=len(GATES))

    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))

    def _aurc_fn(y, s):
        return selective.aurc(s, y)

    all_results = {}
    for gate, (subdir, fname) in GATES.items():
        pq = os.path.join(config.RESULTS, subdir, fname)
        out_json = os.path.join(RESULTS_E7F, f"baselines_{gate}.json")
        if not os.path.exists(pq):
            print(f"[E7f] gate '{gate}' predictions missing; skipping")
            with open(out_json, "w") as f:
                json.dump({"gate": gate, "skipped": True}, f)
            progress.step(pbar, f"{gate} skipped")
            continue
        if os.path.exists(out_json) and not config.FORCE_RERUN:
            with open(out_json) as f:
                all_results[gate] = json.load(f)
            progress.step(pbar, f"{gate} cache reused")
            continue

        preds = pd.read_parquet(pq)
        vp = preds[preds["split"] == "val"].reset_index(drop=True)
        conf_cal = vp.iloc[cal_pos]["confidence"].values.astype(np.float64)
        conf_rep = vp.iloc[rep_pos]["confidence"].values.astype(np.float64)
        corr_cal = (vp.iloc[cal_pos]["correct"].values > 0).astype(int)
        corr_rep = (vp.iloc[rep_pos]["correct"].values > 0).astype(int)
        global_aurc = selective.aurc(conf_rep, corr_rep)

        def _col(name, sub):
            return vp.iloc[sub][name].astype(str).values

        len_chars_cal = np.array([len(s) for s in _col("pred", cal_pos)], float)
        len_chars_rep = np.array([len(s) for s in _col("pred", rep_pos)], float)
        len_words_cal = np.array([len(s.split()) for s in _col("pred", cal_pos)], float)
        len_words_rep = np.array([len(s.split()) for s in _col("pred", rep_pos)], float)
        qlen_cal = np.array([len(s.split()) for s in _col("question", cal_pos)], float)
        qlen_rep = np.array([len(s.split()) for s in _col("question", rep_pos)], float)

        candidates = {}
        if gate == "blip":
            candidates["neg_answer_len_chars"] = (-len_chars_cal, -len_chars_rep)
            candidates["neg_answer_len_words"] = (-len_words_cal, -len_words_rep)
            candidates["lr_conf_len"] = "fit"
        candidates["lr_conf_qlen"] = "fit_q"

        # Entropy / margin only if a topk column exists (future re-harvest).
        if "topk_probs" in vp.columns:
            tk_cal = np.stack(vp.iloc[cal_pos]["topk_probs"].values)
            tk_rep = np.stack(vp.iloc[rep_pos]["topk_probs"].values)
            eps = 1e-12
            candidates["neg_entropy_topk"] = (
                (tk_cal * np.log(tk_cal + eps)).sum(1),
                (tk_rep * np.log(tk_rep + eps)).sum(1))
            candidates["top2_margin"] = (
                tk_cal[:, 0] - tk_cal[:, 1], tk_rep[:, 0] - tk_rep[:, 1])
        else:
            print(f"[E7f] {gate}: no `topk_probs` column cached; entropy/margin "
                  "baselines require an E6 re-harvest that saves the top-k "
                  "answer distribution")

        rows, pvals = [], []
        for name, spec in candidates.items():
            if spec == "fit":
                score = _fit_lr_score(
                    np.column_stack([conf_cal, len_chars_cal, len_words_cal]),
                    corr_cal,
                    np.column_stack([conf_rep, len_chars_rep, len_words_rep]))
            elif spec == "fit_q":
                score = _fit_lr_score(
                    np.column_stack([conf_cal, qlen_cal]), corr_cal,
                    np.column_stack([conf_rep, qlen_rep]))
            else:
                score = np.asarray(spec[1], dtype=np.float64)
            d, lo, hi, p = paired_bootstrap_delta(
                _aurc_fn, corr_rep, conf_rep, score, n_boot=config.N_BOOT)
            rows.append({"score": name,
                         "aurc": float(selective.aurc(score, corr_rep)),
                         "improvement_vs_conf": float(d),
                         "ci95": [lo, hi], "p": p})
            pvals.append(p)
        rejected = benjamini_hochberg(pvals) if pvals else []
        for r, rej in zip(rows, rejected):
            r["bh_fdr_significant"] = bool(rej)

        result = {"gate": gate, "global_confidence_aurc": float(global_aurc),
                  "baselines": rows}
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        all_results[gate] = result

        print(f"\n[E7f] {gate} (global AURC={global_aurc:.4f})")
        for r in rows:
            print(f"  {r['score']:>22s}: AURC={r['aurc']:.4f} "
                  f"improvement={r['improvement_vs_conf']:+.4f} p={r['p']:.3f}")
        progress.step(pbar, f"{gate} baselines computed")

    resultlog.log_run(EXP, metrics=all_results, params={},
                      results_dir=RESULTS_E7F, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E7F, artifacts=required_artifacts())
    pbar.close()
    print("[E7f DONE] Shallow uncertainty baselines contextualize the learned "
          "risk models.")
