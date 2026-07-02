"""E8 - Ablations (C3 joint vs cascade, C4 backbone table) + figures F1-F9
(CPU or any GPU)."""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, figures, progress, resultlog
from src.data_assembly import QUALITY_FLAWS

EXP = "E8"
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]


def required_artifacts():
    arts = [os.path.join(config.RESULTS_E8, "c3_ablation.json")]
    for bb in config.BACKBONES:
        arts.append(os.path.join(config.RESULTS_E3, f"metrics_{bb}.json"))
        arts.append(os.path.join(config.RESULTS_E4, f"per_defect_auroc_{bb}.json"))
        arts.append(os.path.join(config.RESULTS_E5, f"arr_frr_{bb}.json"))
        arts.append(os.path.join(config.RESULTS_E7, f"aurc_comparison_{bb}.json"))
    return arts


def main():
    progress.install_error_hook("E8 ablations/figures")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()

    if expstate.is_done(EXP, config.RESULTS_E8, required=required_artifacts()):
        expstate.skip_banner(EXP, config.RESULTS_E8)
        return

    env.check_gpu(EXP)
    pbar = progress.notebook_bar("E8 ablations/figures", total=4 + len(config.BACKBONES))
    progress.step(pbar, "Environment checked")

    figures.set_fig_dir(config.FIGURES_DIR)

    # ── C3: Joint vs cascade ablation ──
    print("[E8] C3 ablation: joint vs cascade")
    import torch
    from src import heads, train_eval
    from sklearn.metrics import roc_auc_score
    device = "cuda" if torch.cuda.is_available() else "cpu"

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_mask = master["split"] == "val"
    val_idx = np.where(val_mask)[0]
    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    progress.dataframe_summary("master", master)
    progress.step(pbar, "master and split loaded")

    ablation_path = os.path.join(config.RESULTS_E8, "c3_ablation.json")
    if os.path.exists(ablation_path) and not config.FORCE_RERUN:
        with open(ablation_path) as f:
            ablation_results = json.load(f)
        print("[E8] cache hit: c3_ablation.json")
        for bb in ablation_results:
            progress.step(pbar, f"{bb} C3 ablation cached")
    else:
        ablation_results = {}
        for bb in config.BACKBONES:
            emb = np.load(os.path.join(config.ARTIFACTS, f"emb_{bb}.npy")).astype(np.float32)
            y_ans = master["answerable"].values
            Y_def = master[defect_cols].values.astype(np.float32)
            train_m = (master["split"] == "train").values
            X_train, y_train = emb[train_m], y_ans[train_m]
            Y_train = Y_def[train_m]
            X_cal = emb[val_idx[cal_pos]]; y_cal = y_ans[val_idx[cal_pos]]
            Y_cal = Y_def[val_idx[cal_pos]]
            X_rep = emb[val_idx[rep_pos]]; y_rep = y_ans[val_idx[rep_pos]]
            dim = X_train.shape[1]

            cascade_aurocs = []
            joint_aurocs = []
            for seed in config.SEEDS:
                env.seed_everything(seed)
                # Cascade: triage -> defect
                triage_model = heads.MLPHead(dim, 1)
                triage_model, _ = train_eval.train_head(
                    triage_model, X_train, y_train, X_cal, y_cal,
                    seed=seed, device=device, loss_variant="pos_weight")
                with torch.inference_mode():
                    t_logits_rep = triage_model(
                        torch.tensor(X_rep, dtype=torch.float32).to(device)
                    ).squeeze(-1).float().cpu().numpy()
                cascade_aurocs.append(float(roc_auc_score(y_rep, t_logits_rep)))

                # Joint head
                joint_model = heads.JointHead(dim, n_defect=len(DEFECT_NAMES))
                joint_model, _ = train_eval.train_joint_head(
                    joint_model, X_train, y_train, Y_train, X_cal, y_cal, Y_cal,
                    seed=seed, device=device, loss_variant="pos_weight")
                with torch.inference_mode():
                    j_triage, _ = joint_model(
                        torch.tensor(X_rep, dtype=torch.float32).to(device))
                    j_triage = j_triage.squeeze(-1).float().cpu().numpy()
                joint_aurocs.append(float(roc_auc_score(y_rep, j_triage)))

            ablation_results[bb] = {
                "joint_auroc_mean": float(np.mean(joint_aurocs)),
                "joint_auroc_std": float(np.std(joint_aurocs, ddof=1)),
                "cascade_auroc_mean": float(np.mean(cascade_aurocs)),
                "cascade_auroc_std": float(np.std(cascade_aurocs, ddof=1)),
                "delta_auroc_joint_minus_cascade":
                    float(np.mean(joint_aurocs) - np.mean(cascade_aurocs)),
            }
            print(f"  [E8] {bb}: joint AUROC={np.mean(joint_aurocs):.4f}"
                  f"+/-{np.std(joint_aurocs, ddof=1):.4f}  "
                  f"cascade={np.mean(cascade_aurocs):.4f}"
                  f"+/-{np.std(cascade_aurocs, ddof=1):.4f}")
            progress.step(pbar, f"{bb} C3 ablation completed")

        os.makedirs(config.RESULTS_E8, exist_ok=True)
        with open(ablation_path, "w") as f:
            json.dump(ablation_results, f, indent=2)
    progress.step(pbar, "C3 ablation JSON written")

    # ── Generate all paper figures ──
    print("\n[E8] Generating figures F1-F9...")
    saved = []
    saved.append(figures.f1_pipeline_schematic())

    label_stats_path = os.path.join(config.RESULTS_E1, "label_stats.json")
    if os.path.exists(label_stats_path):
        saved.append(figures.f2_cooccurrence(label_stats_path))

    e4_by_bb, e3_by_bb = {}, {}
    for bb in config.BACKBONES:
        p4 = os.path.join(config.RESULTS_E4, f"per_defect_auroc_{bb}.json")
        p3 = os.path.join(config.RESULTS_E3, f"metrics_{bb}.json")
        if os.path.exists(p4):
            with open(p4) as f:
                e4_by_bb[bb] = json.load(f)
        if os.path.exists(p3):
            with open(p3) as f:
                e3_by_bb[bb] = json.load(f)

    if e4_by_bb:
        saved.append(figures.f3_per_defect_auroc(e4_by_bb))

    if config.BACKBONES:
        bb0 = config.BACKBONES[0]
        e7_path = os.path.join(config.RESULTS_E7, f"aurc_comparison_{bb0}.json")
        if os.path.exists(e7_path):
            with open(e7_path) as f:
                e7 = json.load(f)
            calib_tmp = os.path.join(config.RESULTS_E7, "calib_diag.json")
            with open(calib_tmp, "w") as f:
                json.dump({"raw": e7.get("reliability", {}).get("raw", {}),
                           "temp": e7.get("reliability", {}).get("temp", {})}, f)
            saved.append(figures.f4_reliability_diagram(calib_tmp))

            rc_tmp = os.path.join(config.RESULTS_E7, "rc_data.json")
            with open(rc_tmp, "w") as f:
                json.dump(e7.get("risk_coverage", {}), f)
            saved.append(figures.f5_risk_coverage(rc_tmp))

    for bb in config.BACKBONES:
        p5 = os.path.join(config.RESULTS_E5, f"arr_frr_{bb}.json")
        if os.path.exists(p5):
            saved.append(figures.f6_arr_frr(p5))
            break

    combined = {bb: {**e3_by_bb.get(bb, {}), **e4_by_bb.get(bb, {})}
                for bb in config.BACKBONES}
    if combined:
        saved.append(figures.f7_backbone_comparison(combined))
    progress.step(pbar, "Figures F1-F7 generated from cached metrics")

    saved.append(figures.f8_roc_panels(
        triage_roc_data={bb: e3_by_bb.get(bb, {}) for bb in config.BACKBONES},
        defect_roc_data={d: {} for d in DEFECT_NAMES},
    ))
    print("[E8] F9 requires image paths - run with actual E3/E4 predictions attached.")

    print(f"\n[E8 SUMMARY] {len(saved)} figures saved:")
    for p in saved:
        print(f"  {p}")

    resultlog.log_run(EXP,
                      metrics={"ablations": ablation_results,
                               "figures_saved": len(saved)},
                      params={"backbones": config.BACKBONES, "seeds": config.SEEDS},
                      results_dir=config.RESULTS_E8, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E8,
                       artifacts=required_artifacts() + [str(p) for p in saved if p])
    progress.step(pbar, "E8 result logged")
    pbar.close()
    print("[E8 DONE] Commit results/ to the repo.")
