"""E7e - Controls for the "predicted defects help BLIP" interpretation (GPU-lite).

Review Section 2: the gain of continuous predicted-defect scores over the
ground-truth-defect oracle does NOT establish that the learned scores encode
a "richer description of defects" -- they could carry embedding-derived
difficulty, label uncertainty, prevalence, or answerability correlations.
E7e runs the disambiguating controls, all as identical-capacity logistic
risk models over the BLIP (and ViLT) confidence:

  conf_plus_pred_defects   - continuous predicted-defect probs (reference)
  conf_plus_thresh_defects - the SAME predictions binarized at the E4b
                             cal-selected per-label thresholds
  conf_plus_gt_defects     - ground-truth binary defects (oracle)
  conf_plus_embed_pca      - confidence + top-16 PCA of the raw backbone
                             embedding (no defect supervision at all)
  conf_plus_permuted_head  - defect head retrained on PERMUTED defect targets
                             (same capacity/optimization; any gain is pure
                             embedding signal leaking through the head)

If conf_plus_embed_pca or conf_plus_permuted_head matches the reference gain,
the "richer defect description" reading is unsupported and the paper's hedged
interpretation stands. Paired-bootstrap AURC deltas vs global confidence,
BH-FDR within this family.

Needs: E1/E2/E4/E4b caches + E6/E6b predictions. Permuted-head training needs
any GPU (five small MLPs); everything else is CPU.
"""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, heads, progress, resultlog, selective, train_eval
from src.data_assembly import QUALITY_FLAWS
from src.stats import benjamini_hochberg, paired_bootstrap_delta

EXP = "E7E"
RESULTS_E7E = os.path.join(config.RESULTS, "E7e_oracle_controls")
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
GATES = {"vilt": ("E6_vqaconf", "vqa_predictions.parquet"),
         "blip": ("E6b_vqaconf_blip", "vqa_predictions_blip.parquet")}
PCA_DIM = 16


def required_artifacts():
    return [os.path.join(RESULTS_E7E, f"controls_{gate}_{bb}.json")
            for gate in GATES for bb in config.BACKBONES]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


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


def _permuted_head_probs(bb, master, val_idx, cal_pos, rep_pos, device):
    """Train the standard defect MLP on permuted defect targets; return
    sigmoid probabilities on the cal and rep splits (mean over 5 seeds)."""
    import torch
    emb = np.load(os.path.join(config.ARTIFACTS, f"emb_{bb}.npy")).astype(np.float32)
    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    Y = master[defect_cols].values.astype(np.float32)
    train_mask = (master["split"] == "train").values

    rng = np.random.default_rng(config.SEED)
    Y_perm = Y.copy()
    Y_perm[train_mask] = Y[train_mask][rng.permutation(train_mask.sum())]

    X_train, Y_train = emb[train_mask], Y_perm[train_mask]
    X_cal = emb[val_idx[cal_pos]]
    Y_cal = Y_perm[val_idx[cal_pos]]
    X_rep = emb[val_idx[rep_pos]]

    cal_probs, rep_probs = [], []
    for seed in config.SEEDS:
        model = heads.MLPHead(emb.shape[1], len(DEFECT_NAMES), multilabel=True)
        model, _ = train_eval.train_head(model, X_train, Y_train, X_cal, Y_cal,
                                         seed=seed, device=device,
                                         loss_variant="pos_weight")
        with torch.inference_mode():
            dev = next(model.parameters()).device
            lc = model(torch.tensor(X_cal, dtype=torch.float32).to(dev))
            lr = model(torch.tensor(X_rep, dtype=torch.float32).to(dev))
        cal_probs.append(_sigmoid(lc.float().cpu().numpy()))
        rep_probs.append(_sigmoid(lr.float().cpu().numpy()))
    return np.mean(cal_probs, axis=0), np.mean(rep_probs, axis=0)


def main():
    progress.install_error_hook("E7e oracle controls")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E7E, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E7E, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E7E)
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pbar = progress.notebook_bar("E7e oracle controls",
                                 total=1 + len(GATES) * len(config.BACKBONES))

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]
    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    gt_cal = master.iloc[val_idx[cal_pos]][defect_cols].values.astype(float)
    gt_rep = master.iloc[val_idx[rep_pos]][defect_cols].values.astype(float)
    progress.step(pbar, "cached E1 data loaded")

    def _aurc_fn(y, s):
        return selective.aurc(s, y)

    all_results = {}
    for gate, (subdir, fname) in GATES.items():
        pq = os.path.join(config.RESULTS, subdir, fname)
        if not os.path.exists(pq):
            print(f"[E7e] gate '{gate}' predictions missing; skipping")
            continue
        preds = pd.read_parquet(pq)
        vp = preds[preds["split"] == "val"].reset_index(drop=True)
        conf_cal = vp.iloc[cal_pos]["confidence"].values.astype(np.float64)
        conf_rep = vp.iloc[rep_pos]["confidence"].values.astype(np.float64)
        corr_cal = (vp.iloc[cal_pos]["correct"].values > 0).astype(int)
        corr_rep = (vp.iloc[rep_pos]["correct"].values > 0).astype(int)
        global_aurc = selective.aurc(conf_rep, corr_rep)

        for bb in config.BACKBONES:
            out_json = os.path.join(RESULTS_E7E, f"controls_{gate}_{bb}.json")
            if os.path.exists(out_json) and not config.FORCE_RERUN:
                with open(out_json) as f:
                    all_results[f"{gate}_{bb}"] = json.load(f)
                progress.step(pbar, f"{gate}/{bb} cache reused")
                continue

            defect_logits = np.load(
                os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy"))
            pd_cal = _sigmoid(defect_logits[cal_pos])
            pd_rep = _sigmoid(defect_logits[rep_pos])

            # E4b thresholds if available, else 0.5 everywhere.
            e4b_path = os.path.join(config.RESULTS, "E4b_thresholds",
                                    f"thresholds_{bb}.json")
            if os.path.exists(e4b_path):
                with open(e4b_path) as f:
                    taus = np.array([json.load(f)["per_label_thresholds"][d]
                                     for d in DEFECT_NAMES])
            else:
                taus = np.full(len(DEFECT_NAMES), 0.5)

            # Embedding PCA features (fit on cal only; frozen-knob).
            from sklearn.decomposition import PCA
            emb = np.load(os.path.join(config.ARTIFACTS, f"emb_{bb}.npy")).astype(np.float32)
            emb_cal = emb[val_idx[cal_pos]]
            emb_rep = emb[val_idx[rep_pos]]
            env.assert_no_rep_leakage("cal")
            pca = PCA(n_components=PCA_DIM, random_state=config.SEED).fit(emb_cal)
            pca_cal, pca_rep = pca.transform(emb_cal), pca.transform(emb_rep)

            # Permuted-target head (trains 5 small MLPs).
            perm_cal, perm_rep = _permuted_head_probs(
                bb, master, val_idx, cal_pos, rep_pos, device)

            variants = {
                "conf_plus_pred_defects": (
                    np.column_stack([conf_cal, pd_cal]),
                    np.column_stack([conf_rep, pd_rep])),
                "conf_plus_thresh_defects": (
                    np.column_stack([conf_cal, (pd_cal >= taus).astype(float)]),
                    np.column_stack([conf_rep, (pd_rep >= taus).astype(float)])),
                "conf_plus_gt_defects": (
                    np.column_stack([conf_cal, gt_cal]),
                    np.column_stack([conf_rep, gt_rep])),
                "conf_plus_embed_pca": (
                    np.column_stack([conf_cal, pca_cal]),
                    np.column_stack([conf_rep, pca_rep])),
                "conf_plus_permuted_head": (
                    np.column_stack([conf_cal, perm_cal]),
                    np.column_stack([conf_rep, perm_rep])),
            }

            rows, pvals = [], []
            for name, (Xc, Xr) in variants.items():
                score = _fit_lr_score(Xc, corr_cal, Xr)
                d, lo, hi, p = paired_bootstrap_delta(
                    _aurc_fn, corr_rep, conf_rep, score, n_boot=config.N_BOOT)
                # improvement = AURC(conf) - AURC(score); positive = better
                rows.append({"variant": name, "aurc": float(selective.aurc(score, corr_rep)),
                             "improvement": float(d), "ci95": [lo, hi], "p": p})
                pvals.append(p)
            rejected = benjamini_hochberg(pvals)
            for r, rej in zip(rows, rejected):
                r["bh_fdr_significant"] = bool(rej)

            result = {"gate": gate, "backbone": bb,
                      "global_confidence_aurc": float(global_aurc),
                      "variants": rows}
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            all_results[f"{gate}_{bb}"] = result

            print(f"\n[E7e] gate={gate} bb={bb} (global AURC={global_aurc:.4f})")
            for r in rows:
                print(f"  {r['variant']:>26s}: improvement={r['improvement']:+.4f} "
                      f"p={r['p']:.3f} {'*' if r['bh_fdr_significant'] else ''}")
            progress.step(pbar, f"{gate}/{bb} controls computed")

    resultlog.log_run(EXP, metrics={k: v["variants"] for k, v in all_results.items()},
                      params={"pca_dim": PCA_DIM},
                      results_dir=RESULTS_E7E, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E7E, artifacts=required_artifacts())
    pbar.close()
    print("[E7e DONE] Controls disambiguate the 'richer defect description' claim.")
