"""E9 - Phase 2 groundability harvest (GPU). GATED behind config.RUN_E9:
set os.environ['VQA_RUN_E9'] = '1' before importing src to unlock."""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, progress, resultlog, staging

EXP = "E9"


def main():
    progress.install_error_hook("E9 groundability")

    # Re-read the env var at call time so the notebook can unlock E9 without
    # restarting the runtime.
    if not (config.RUN_E9 or os.environ.get("VQA_RUN_E9", "0") == "1"):
        pbar = progress.notebook_bar("E9 gated", total=1)
        print("=" * 70)
        print("E9 is GATED.  Uncomment  os.environ['VQA_RUN_E9'] = '1'  in this")
        print("cell to unlock. Do NOT run E9 until E0-E8 results are committed.")
        print("=" * 70)
        progress.step(pbar, "E9 gate checked; no compute used")
        pbar.close()
        return

    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()

    delta_path = os.path.join(config.RESULTS_E9, "triage_delta.json")
    if expstate.is_done(EXP, config.RESULTS_E9, required=[delta_path]):
        expstate.skip_banner(EXP, config.RESULTS_E9)
        return

    from src import figures, grounding
    env.check_gpu(EXP)  # raises on CPU runtime
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pbar = progress.notebook_bar("E9 groundability", total=10)
    progress.step(pbar, f"Environment checked: device={device}")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_mask = master["split"] == "val"
    val_idx = np.where(val_mask)[0]

    # ── 1. Deterministic subsample from the rep split ──
    subsample_ids_path = os.path.join(config.RESULTS_E9, "subsample_ids.json")
    if not os.path.exists(subsample_ids_path) or config.FORCE_RERUN:
        rng = np.random.default_rng(config.SEED)
        n_sub = min(config.E9_SUBSAMPLE_N, len(rep_pos))
        sub_pos = rng.choice(rep_pos, size=n_sub, replace=False)
        sub_idx = val_idx[sub_pos]   # absolute indices into master
        os.makedirs(config.RESULTS_E9, exist_ok=True)
        with open(subsample_ids_path, "w") as f:
            json.dump({"subsample_global_idx": sub_idx.tolist(),
                       "subsample_rep_pos": sub_pos.tolist(),
                       "n": int(n_sub), "seed": config.SEED}, f)
        print(f"[E9] Subsample: {n_sub} images")
    else:
        with open(subsample_ids_path) as f:
            sub_info = json.load(f)
        sub_idx = np.array(sub_info["subsample_global_idx"])
        sub_pos = np.array(sub_info["subsample_rep_pos"])
        n_sub = sub_info["n"]
        print(f"[E9] Loaded subsample: {n_sub} images")
    progress.step(pbar, f"Subsample ready: n={n_sub}")

    sub_df = master.iloc[sub_idx].reset_index(drop=True)

    # ── 2. Entity extraction (cached separately) ──
    entity_path = os.path.join(config.RESULTS_E9, "entities.json")
    if not os.path.exists(entity_path) or config.FORCE_RERUN:
        entities = [grounding.extract_entity(q) for q in sub_df["question"]]
        with open(entity_path, "w") as f:
            json.dump({"entities": entities,
                       "method": "spacy_noun_chunk_fallback"}, f)
        print(f"[E9] Entity extraction done. Sample: {entities[:3]}")
    else:
        with open(entity_path) as f:
            entities = json.load(f)["entities"]
        print("[E9] Loaded cached entities.")
    progress.step(pbar, "Entity extraction/cache ready")

    # ── 3. Grounding harvest (resume-safe) ──
    grounding_path = os.path.join(config.RESULTS_E9, "grounding_cache.parquet")
    staging.ensure_images()
    grounding_records = []
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        grounding_records.append({
            "global_idx": int(sub_idx[i]),
            "image": row["image"],
            "image_path": staging.resolve_image_path(row["image"], row["split"]),
            "phrase": entities[i],
        })

    cache_df = grounding.harvest_grounding(
        records=grounding_records,
        out_parquet=grounding_path,
        device=device,
        grounder=config.GROUNDER,
        force=config.FORCE_RERUN,
        shard_rows=200,
    )
    print(f"[E9] Grounding cache ready: {len(cache_df)} rows")
    progress.step(pbar, "Grounding cache ready")

    # ── 4. RQ3a: does groundability help triage? ──
    from src.stats import paired_bootstrap_delta, delong_auroc
    from sklearn.metrics import roc_auc_score as _roc
    from sklearn.linear_model import LogisticRegression

    bb0 = config.BACKBONES[0]
    emb_all = np.load(os.path.join(config.ARTIFACTS, f"emb_{bb0}.npy")).astype(np.float32)
    X_sub = emb_all[sub_idx]
    y_sub = master.iloc[sub_idx]["answerable"].values

    feat_cols = ["grounded", "n_boxes", "max_conf", "box_area_frac",
                 "touches_border", "centeredness"]
    cache_df = cache_df.set_index("global_idx").reindex(sub_idx)
    G_feat = cache_df[feat_cols].fillna(0).values.astype(np.float32)
    progress.step(pbar, "Groundability feature matrix assembled")

    sub_cal_pos, sub_rep_pos = env.make_cal_rep_split(
        np.arange(len(sub_idx)), cal_frac=config.CAL_FRAC, stratify_labels=y_sub)
    X_sub_cal = X_sub[sub_cal_pos]; y_sub_cal = y_sub[sub_cal_pos]
    X_sub_rep = X_sub[sub_rep_pos]; y_sub_rep = y_sub[sub_rep_pos]

    # Appearance-only
    lr_app = LogisticRegression(max_iter=500)
    lr_app.fit(X_sub_cal, y_sub_cal)
    scores_app = lr_app.predict_proba(X_sub_rep)[:, 1]
    progress.step(pbar, "Appearance-only triage fitted")

    # Appearance + groundability
    X_sub_cal_g = np.hstack([X_sub_cal, G_feat[sub_cal_pos]])
    X_sub_rep_g = np.hstack([X_sub_rep, G_feat[sub_rep_pos]])
    lr_gnd = LogisticRegression(max_iter=500)
    lr_gnd.fit(X_sub_cal_g, y_sub_cal)
    scores_gnd = lr_gnd.predict_proba(X_sub_rep_g)[:, 1]
    progress.step(pbar, "Appearance+groundability triage fitted")

    auc_app = float(_roc(y_sub_rep, scores_app))
    auc_gnd = float(_roc(y_sub_rep, scores_gnd))
    delta_au, ci_lo, ci_hi, p_au = paired_bootstrap_delta(
        lambda y, s: _roc(y, s), y_sub_rep, scores_gnd, scores_app,
        n_boot=config.N_BOOT)
    auc_a_d, auc_b_d, delta_d, z_d, p_d = delong_auroc(y_sub_rep, scores_gnd, scores_app)

    print(f"[E9] Appearance-only AUROC:    {auc_app:.4f}")
    print(f"[E9] +Groundability AUROC:     {auc_gnd:.4f}")
    print(f"[E9] dAUROC={delta_au:.4f} [{ci_lo:.4f},{ci_hi:.4f}] p={p_au:.4f}")
    print(f"[E9] DeLong: z={z_d:.3f}  p={p_d:.4f}")
    progress.step(pbar, "AUROC/bootstrap/DeLong comparison completed")

    delta_results = {
        "subsample_n": int(n_sub),
        "grounder": config.GROUNDER,
        "auroc_appearance": auc_app,
        "auroc_groundability": auc_gnd,
        "delta_AUROC": delta_au,
        "delta_AUROC_ci_lo": ci_lo,
        "delta_AUROC_ci_hi": ci_hi,
        "delta_AUROC_p": p_au,
        "delong_z": z_d, "delong_p": p_d,
    }
    os.makedirs(config.RESULTS_E9, exist_ok=True)
    with open(delta_path, "w") as f:
        json.dump(delta_results, f, indent=2)

    figures.set_fig_dir(config.FIGURES_DIR)
    figures.f10_groundability(delta_path)
    progress.step(pbar, "F10 groundability figure saved")

    resultlog.log_run(EXP, metrics=delta_results,
                      params={"grounder": config.GROUNDER, "n_sub": int(n_sub)},
                      results_dir=config.RESULTS_E9, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E9, artifacts=[delta_path])
    progress.step(pbar, "E9 result logged")
    pbar.close()
    print("[E9 DONE]")
