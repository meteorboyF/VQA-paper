"""E5 - Actionable Recovery metric ARR/FRR (CPU, numpy only)."""
import json
import os

import numpy as np
import pandas as pd

from src import actionable, config, env, expstate, progress, resultlog
from src.data_assembly import QUALITY_FLAWS

EXP = "E5"
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]


def required_artifacts():
    return [os.path.join(config.RESULTS_E5, f"arr_frr_{bb}.json")
            for bb in config.BACKBONES]


def main():
    progress.install_error_hook("E5 actionable recovery")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()

    if expstate.is_done(EXP, config.RESULTS_E5, required=required_artifacts()):
        expstate.skip_banner(EXP, config.RESULTS_E5)
        return

    env.check_gpu(EXP)
    pbar = progress.notebook_bar("E5 actionable recovery", total=3 + len(config.BACKBONES))
    progress.step(pbar, "Environment checked")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_mask = master["split"] == "val"
    val_idx = np.where(val_mask)[0]
    rep_idx = val_idx[rep_pos]

    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    Y_rep = master.iloc[rep_idx][defect_cols].values
    ans_rep = master.iloc[rep_idx]["answerable"].values
    progress.dataframe_summary("master", master)
    progress.step(pbar, "rep split labels loaded")

    all_results = {}
    for bb in config.BACKBONES:
        out_path = os.path.join(config.RESULTS_E5, f"arr_frr_{bb}.json")
        if os.path.exists(out_path) and not config.FORCE_RERUN:
            print(f"[E5] cache hit: {out_path}")
            with open(out_path) as f:
                all_results[bb] = json.load(f)
            progress.step(pbar, f"{bb} cache reused")
            continue

        logits_path = os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy")
        assert os.path.exists(logits_path), f"Run E4 first! Missing: {logits_path}"

        # Logits saved by E4 are on the full val split; slice to rep
        logits_val = np.load(logits_path)
        logits_rep = logits_val[rep_pos]
        probs_rep = 1 / (1 + np.exp(-logits_rep))

        result = actionable.actionable_recovery_rate(
            pred_defect_probs=probs_rep,
            gt_defects=Y_rep,
            answerable=ans_rep,
            defect_names=DEFECT_NAMES,
        )
        os.makedirs(config.RESULTS_E5, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        all_results[bb] = result
        print(f"[E5] {bb}: ARR={result['ARR']:.4f} "
              f"[{result['ARR_ci95'][0]:.4f},{result['ARR_ci95'][1]:.4f}]  "
              f"FRR={result['FRR']:.4f} "
              f"[{result['FRR_ci95'][0]:.4f},{result['FRR_ci95'][1]:.4f}]")
        progress.step(pbar, f"{bb} ARR/FRR computed")

    resultlog.log_run(EXP, metrics=all_results,
                      params={"backbones": config.BACKBONES},
                      results_dir=config.RESULTS_E5, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E5, artifacts=required_artifacts())
    progress.step(pbar, "E5 result logged")
    pbar.close()
    print("[E5 DONE]")
