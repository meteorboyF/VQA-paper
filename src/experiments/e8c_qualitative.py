"""E8c - Qualitative figure F9 (CPU; stages val images only, ~3.3 GB).

The paper currently has no qualitative examples, which is a real weakness for
an assistive-technology venue. This builds the rule-sampled (QUAL_SEED=7,
never hand-picked) F9 grid from the rep split:

  row 1: ANSWERED   - high-confidence correct answers
  row 2: DANGER     - high-confidence WRONG answers (the case that motivates
                      refusal in the first place)
  row 3: GOOD REFUSAL - unanswerable, refused, predicted defect matches GT
                        (the retake action shown is what the user would hear)
  row 4: WRONG REASON - unanswerable, but the predicted defect is not a GT
                        defect (honest failure mode of the explanation head)

Also writes F9_manifest.json (image, question, prediction, confidence,
defect, action per panel) so the paper caption can quote examples verbatim.
"""
import json
import os

import numpy as np
import pandas as pd

from src import actionable, config, env, expstate, figures, progress, resultlog, staging
from src.data_assembly import QUALITY_FLAWS

EXP = "E8C"
RESULTS_E8C = os.path.join(config.RESULTS, "E8c_qualitative")
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
N_PER_ROW = 4


def required_artifacts():
    return [os.path.join(RESULTS_E8C, "F9_manifest.json")]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _shorten(text, n=38):
    text = str(text).strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def main():
    progress.install_error_hook("E8c qualitative F9")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E8C, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E8C, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E8C)
        return

    pbar = progress.notebook_bar("E8c qualitative F9", total=5)
    progress.step(pbar, "Environment checked")

    # Only the val images are needed (rep split lives in val).
    staging.stage_kinds(["images_val"])
    progress.step(pbar, "val images staged")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    vqa_preds = pd.read_parquet(os.path.join(config.RESULTS_E6, "vqa_predictions.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]

    val_preds = vqa_preds[vqa_preds["split"] == "val"].reset_index(drop=True)
    rep_preds = val_preds.iloc[rep_pos].reset_index(drop=True)
    rep_master = master.iloc[val_idx[rep_pos]].reset_index(drop=True)

    conf = rep_preds["confidence"].values.astype(np.float64)
    corr = (rep_preds["correct"].values > 0).astype(int)
    ans = rep_master["answerable"].values.astype(int)
    gt_defects = rep_master[[f"q_{d}" for d in DEFECT_NAMES]].values

    # First backbone's defect head provides the explanations (as in the paper)
    bb = config.BACKBONES[0]
    logits_path = os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy")
    assert os.path.exists(logits_path), f"Run E4 first! Missing: {logits_path}"
    probs = _sigmoid(np.load(logits_path)[rep_pos])
    top_pred = actionable.top_predicted_defect(probs, DEFECT_NAMES)

    def top_matches_gt(i):
        t = top_pred[i]
        return t is not None and gt_defects[i, DEFECT_NAMES.index(t)] == 1

    hi_conf = conf >= np.quantile(conf, 0.75)
    pools = {
        "ANSWERED": np.where(hi_conf & (corr == 1))[0],
        "DANGER: confident+wrong": np.where(hi_conf & (corr == 0) & (ans == 1))[0],
        "GOOD REFUSAL": np.array([i for i in np.where(ans == 0)[0] if top_matches_gt(i)]),
        "WRONG REASON": np.array([i for i in np.where(ans == 0)[0]
                                  if top_pred[i] is not None and not top_matches_gt(i)]),
    }
    progress.step(pbar, "candidate pools built: " +
                  ", ".join(f"{k}:{len(v)}" for k, v in pools.items()))

    rng = np.random.default_rng(config.QUAL_SEED)
    qual_data, manifest = [], []
    for row_label, pool in pools.items():
        if len(pool) == 0:
            print(f"[E8c] WARN: empty pool for '{row_label}'")
            continue
        chosen = rng.choice(pool, size=min(N_PER_ROW, len(pool)), replace=False)
        for i in chosen:
            i = int(i)
            image = rep_master.iloc[i]["image"]
            top = top_pred[i] or "none"
            action = actionable.DEFECT_TO_ACTION.get(top, "-")
            split_type = ("TP" if row_label in ("ANSWERED", "GOOD REFUSAL")
                          else "FP")
            qual_data.append({
                "image_path": staging.resolve_image_path(image, "val"),
                "defect": top,
                "split_type": split_type,
                "label": f"{row_label} | conf={conf[i]:.2f}",
            })
            manifest.append({
                "row": row_label,
                "image": image,
                "question": str(rep_master.iloc[i]["question"]),
                "vqa_prediction": str(rep_preds.iloc[i]["pred"]),
                "confidence": float(conf[i]),
                "correct": int(corr[i]),
                "answerable": int(ans[i]),
                "top_predicted_defect": top,
                "gt_defects": [d for d in DEFECT_NAMES
                               if gt_defects[i, DEFECT_NAMES.index(d)] == 1],
                "retake_action": action,
            })

    figures.set_fig_dir(config.FIGURES_DIR)
    fig_path = figures.f9_qualitative_grid(
        qual_data,
        title=f"Reliability-layer behavior on rep examples ({bb} defect head)",
        fig_name="F9_qualitative_grid",
    )
    print(f"[E8c] figure saved: {fig_path}")
    progress.step(pbar, "F9 grid rendered")

    manifest_path = os.path.join(RESULTS_E8C, "F9_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"backbone": bb, "qual_seed": config.QUAL_SEED,
                   "panels": manifest}, f, indent=2)

    resultlog.log_run(EXP,
                      metrics={"n_panels": len(manifest),
                               "pools": {k: int(len(v)) for k, v in pools.items()}},
                      params={"backbone": bb, "qual_seed": config.QUAL_SEED},
                      results_dir=RESULTS_E8C, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E8C, artifacts=required_artifacts())
    pbar.close()
    print("[E8c DONE] F9 + manifest ready; captions can quote the manifest verbatim.")
