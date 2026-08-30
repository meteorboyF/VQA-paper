"""E10 - Question-conditioned triage (GPU; the review's Critical #1 fix).

VizWiz answerability is defined for an IMAGE-QUESTION pair, but E3's heads
see only the image embedding. E10 runs the comparison the review demands,
all under the identical frozen-knob 5-seed protocol:

  1. image-only linear probe          (weakest visual baseline)
  2. image-only MLP                   (replicates E3 for a same-run reference)
  3. question-only MLP                (CLIP text tower embedding)
  4. concat(image, question) MLP
  5. gated fusion MLP                 (concat + elementwise product + cosine)
  6. VQA-confidence-only              (score = frozen ViLT confidence)
  7. concat + VQA confidence MLP
  8. CLIP image-text cosine           (zero-training baseline)
  9. majority class                   (trivial baseline)

Image features per backbone come from E2 caches; question features use the
CLIP text tower for every backbone variant (kept fixed so differences track
the image side). Paired-bootstrap AUROC deltas vs the image-only MLP are
reported with BH-FDR over the family.

Needs: emb_{bb}.npy (E2), master.parquet + split_ids (E1), E6 predictions.
"""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, heads, progress, resultlog, text_features, train_eval
from src.stats import benjamini_hochberg, paired_bootstrap_delta

EXP = "E10"
RESULTS_E10 = os.path.join(config.RESULTS, "E10_question_triage")


def required_artifacts():
    return [os.path.join(RESULTS_E10, f"question_triage_{bb}.json")
            for bb in config.BACKBONES]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def _run_variant(name, X_train, y_train, X_cal, y_cal, X_rep, y_rep,
                 device, head_type="mlp"):
    """5-seed head training; returns metrics dict + mean rep scores."""
    dim = X_train.shape[1]

    def make_model():
        if head_type == "linear":
            return heads.LinearHead(dim, 1)
        return heads.MLPHead(dim, 1)

    def thresh_fn(y_c, logits, split_name="cal"):
        return train_eval.find_threshold(y_c, logits, split_name=split_name)

    res = train_eval.run_multi_seed(
        make_model, X_train, y_train, X_cal, y_cal, X_rep, y_rep,
        label_names=["answerable"], threshold_fn=thresh_fn,
        eval_fn=lambda y_r, lg, t: train_eval.evaluate_binary(y_r, lg, t),
        seeds=config.SEEDS, device=device, loss_variant="pos_weight")
    mean_rep = np.mean(res["_logits_rep"], axis=0)
    metrics = {k: v for k, v in res.items() if not k.startswith("_")}
    print(f"  [{name}] AUROC={metrics['AUROC']['mean']:.4f} "
          f"±{metrics['AUROC']['std']:.4f}  AUPRC={metrics['AUPRC']['mean']:.4f}")
    return metrics, mean_rep


def main():
    progress.install_error_hook("E10 question-conditioned triage")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E10, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E10, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E10)
        return

    env.check_gpu("E3")  # any GPU; heads are tiny
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pbar = progress.notebook_bar("E10 question triage",
                                 total=3 + len(config.BACKBONES))

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]
    train_mask = (master["split"] == "train").values
    y = master["answerable"].values.astype(int)
    progress.step(pbar, "master + splits loaded")

    # Question embeddings (CLIP text tower), shared across backbones.
    txt = text_features.extract_question_embeddings(
        master["question"].tolist(), device=device,
        force=config.FORCE_RERUN).astype(np.float32)
    progress.step(pbar, "question embeddings ready")

    # Frozen ViLT confidence, aligned to master row order via split.
    vqa_preds = pd.read_parquet(os.path.join(config.RESULTS_E6, "vqa_predictions.parquet"))
    conf = np.full(len(master), np.nan)
    for split in ("train", "val"):
        m = (master["split"] == split).values
        p = vqa_preds[vqa_preds["split"] == split]["confidence"].values
        if m.sum() == len(p):
            conf[m] = p
    assert np.isfinite(conf).all(), "E6 predictions do not cover master rows"
    progress.step(pbar, "ViLT confidence aligned")

    y_train = y[train_mask]
    y_cal = y[val_idx[cal_pos]]
    y_rep = y[val_idx[rep_pos]]

    def split3(M):
        return M[train_mask], M[val_idx[cal_pos]], M[val_idx[rep_pos]]

    def _norm(M):
        return M / np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-8)

    txt_n = _norm(txt)

    # Image-text cosine lives in the shared CLIP space only; computed once
    # from the CLIP image embedding and reused as a scalar feature for every
    # backbone variant (and as a zero-training score baseline).
    clip_emb_path = os.path.join(config.ARTIFACTS, "emb_clip.npy")
    assert os.path.exists(clip_emb_path), f"Run E2 first! Missing: {clip_emb_path}"
    clip_img_n = _norm(np.load(clip_emb_path).astype(np.float32))
    cos = (clip_img_n * txt_n).sum(axis=1, keepdims=True)

    all_results = {}
    for bb in config.BACKBONES:
        out_json = os.path.join(RESULTS_E10, f"question_triage_{bb}.json")
        if os.path.exists(out_json) and not config.FORCE_RERUN:
            with open(out_json) as f:
                all_results[bb] = json.load(f)
            print(f"[E10] cache hit: {out_json}")
            progress.step(pbar, f"{bb} cache reused")
            continue

        emb_path = os.path.join(config.ARTIFACTS, f"emb_{bb}.npy")
        assert os.path.exists(emb_path), f"Run E2 first! Missing: {emb_path}"
        img = np.load(emb_path).astype(np.float32)
        print(f"\n[E10] backbone={bb}  img_dim={img.shape[1]}  txt_dim={txt.shape[1]}")

        img_n = _norm(img)
        # Elementwise interaction requires matching dims (true only for CLIP);
        # other backbones use the concat + cosine fusion without the product.
        interaction = (img_n * txt_n if img.shape[1] == txt.shape[1]
                       else np.zeros((len(img), 0), np.float32))
        feature_sets = {
            "image_only_linear": (img, "linear"),
            "image_only_mlp": (img, "mlp"),
            "question_only_mlp": (txt, "mlp"),
            "concat_mlp": (np.hstack([img, txt]), "mlp"),
            "fusion_mlp": (np.hstack([img_n, txt_n, interaction, cos]), "mlp"),
            "concat_conf_mlp": (np.hstack([img, txt, conf[:, None].astype(np.float32)]),
                                "mlp"),
        }

        bb_res = {"variants": {}, "score_baselines": {}}
        mean_scores = {}
        for name, (X, ht) in feature_sets.items():
            Xtr, Xc, Xr = split3(X)
            metrics, mean_rep = _run_variant(
                name, Xtr, y_train, Xc, y_cal, Xr, y_rep, device, head_type=ht)
            bb_res["variants"][name] = metrics
            mean_scores[name] = mean_rep

        # Zero-training score baselines evaluated directly on rep.
        cos_rep = cos[val_idx[rep_pos], 0]
        conf_rep = conf[val_idx[rep_pos]]
        bb_res["score_baselines"] = {
            "clip_imgtext_cosine_auroc": float(_auroc(y_rep, cos_rep)),
            "vqa_confidence_auroc": float(_auroc(y_rep, conf_rep)),
            "majority_class_rate": float(max(y_train.mean(), 1 - y_train.mean())),
        }

        # Paired-bootstrap AUROC deltas vs image-only MLP + BH-FDR.
        ref = mean_scores["image_only_mlp"]
        deltas, pvals, names = [], [], []
        for name, sc in mean_scores.items():
            if name == "image_only_mlp":
                continue
            d, lo, hi, p = paired_bootstrap_delta(_auroc, y_rep, sc, ref,
                                                  n_boot=config.N_BOOT)
            deltas.append({"variant": name, "delta_auroc_vs_image_mlp": d,
                           "ci95": [lo, hi], "p": p})
            pvals.append(p)
            names.append(name)
        rejected = benjamini_hochberg(pvals) if pvals else []
        for d_entry, rej in zip(deltas, rejected):
            d_entry["bh_fdr_significant"] = bool(rej)
        bb_res["paired_deltas"] = deltas

        with open(out_json, "w") as f:
            json.dump(bb_res, f, indent=2)
        all_results[bb] = bb_res
        progress.step(pbar, f"{bb} question-triage variants done")

    resultlog.log_run(EXP, metrics={bb: {"n_variants": len(v["variants"])}
                                    for bb, v in all_results.items()},
                      params={"backbones": config.BACKBONES,
                              "text_encoder": "open_clip ViT-B-32 text tower"},
                      results_dir=RESULTS_E10, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E10, artifacts=required_artifacts())
    pbar.close()
    print("[E10 DONE] Question-conditioned triage table ready for the "
          "manuscript's triage section.")
