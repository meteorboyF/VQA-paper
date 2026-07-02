"""E1 - Master data assembly (CPU). Joins VQA + QualityIssues annotations
into master.parquet and carves the deterministic cal/rep split."""
import os

import numpy as np

from src import config, data_assembly, env, expstate, progress, resultlog, staging

EXP = "E1"


def main():
    progress.install_error_hook("E1 master assembly")
    env.seed_everything()
    env.check_gpu(EXP)
    env.mount_drive()
    config.ensure_output_dirs()

    master_path = os.path.join(config.DATA_PROCESSED, "master.parquet")
    stats_path = os.path.join(config.RESULTS_E1, "label_stats.json")
    split_ids_path = os.path.join(config.RESULTS_E1, "split_ids.json")
    if expstate.is_done(EXP, config.RESULTS_E1,
                        required=[master_path, stats_path, split_ids_path]):
        expstate.skip_banner(EXP, config.RESULTS_E1)
        return

    pbar = progress.notebook_bar("E1 master assembly", total=6)
    progress.step(pbar, "Environment checked")
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    if os.path.exists(master_path) and not config.FORCE_RERUN:
        import pandas as pd
        master = pd.read_parquet(master_path)
        print(f"[E1] cache hit: master.parquet ({len(master)} rows)")
        progress.dataframe_summary("master", master)
    else:
        # A fresh runtime may not have the annotation zips staged yet.
        staging.ensure_annotations()
        vqa_paths = {s: staging.find_annotation_json("vqa", s) for s in ("train", "val")}
        quality_paths = {s: staging.find_annotation_json("quality", s) for s in ("train", "val")}
        missing = [k for k, v in {**{f"vqa_{k}": v for k, v in vqa_paths.items()},
                                  **{f"quality_{k}": v for k, v in quality_paths.items()}}.items()
                   if v is None]
        if missing:
            raise FileNotFoundError(
                f"[E1] Cannot find annotation files for: {missing}\n"
                f"Check that E0 staged the data correctly (rerun E0 if needed).")
        print("[E1] Building master.parquet...")
        print(f"  VQA paths:     {vqa_paths}")
        print(f"  Quality paths: {quality_paths}")
        master = data_assembly.build_master(vqa_paths, quality_paths, master_path)
        progress.dataframe_summary("master", master)
    progress.step(pbar, "master.parquet loaded/built")

    stats = data_assembly.label_stats(master, stats_path)
    progress.step(pbar, "label statistics written")

    # ── Deterministic cal/rep split ──
    if not os.path.exists(split_ids_path) or config.FORCE_RERUN:
        val_mask = master["split"] == "val"
        val_idx = np.where(val_mask)[0]
        strat_labs = master.loc[val_mask, "answerable"].values
        cal_pos, rep_pos = env.make_cal_rep_split(
            val_idx, cal_frac=config.CAL_FRAC, stratify_labels=strat_labs)
        env.save_split_ids(cal_pos, rep_pos, split_ids_path)
    else:
        print(f"[E1] split_ids already saved: {split_ids_path}")
    progress.step(pbar, "cal/rep split ready")

    print("\n[E1 SUMMARY]")
    print(f"  Total rows:  {len(master)}")
    print(f"  Splits: {master['split'].value_counts().to_dict()}")
    print(f"  Answerable rate: {master['answerable'].mean():.3f}")
    for flaw in data_assembly.QUALITY_FLAWS + ["unrecognizable"]:
        col = f"q_{flaw}"
        if col in master.columns:
            print(f"  {flaw}: {master[col].mean():.3f}")
    progress.step(pbar, "summary printed")

    resultlog.log_run(EXP, metrics=stats,
                      params={"seed": config.SEED, "cal_frac": config.CAL_FRAC},
                      results_dir=config.RESULTS_E1, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E1,
                       artifacts=[master_path, stats_path, split_ids_path])
    progress.step(pbar, "E1 result logged")
    pbar.close()
    print("[E1 DONE]")
