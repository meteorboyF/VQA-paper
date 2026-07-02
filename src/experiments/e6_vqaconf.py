"""E6 - Frozen ViLT confidence harvest (GPU; second/last expensive cell).
Caches predictions to Drive, resumes from shards after a crash."""
import os

import pandas as pd

from src import config, env, expstate, progress, resultlog, staging, vqa_confidence

EXP = "E6"


def main():
    progress.install_error_hook("E6 ViLT confidence harvest")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()

    out_path = os.path.join(config.RESULTS_E6, "vqa_predictions.parquet")
    if expstate.is_done(EXP, config.RESULTS_E6, required=[out_path]):
        expstate.skip_banner(EXP, config.RESULTS_E6)
        return

    pbar = progress.notebook_bar("E6 ViLT confidence harvest", total=5)

    if os.path.exists(out_path) and not config.FORCE_RERUN:
        # Final parquet exists (e.g. finished but marker missing) - no GPU needed.
        print(f"[E6] cache hit: {out_path}")
        df = pd.read_parquet(out_path)
        progress.step(pbar, "VQA prediction cache loaded", advance=3)
    else:
        env.check_gpu(EXP)  # raises on CPU runtime
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bs = config.batch_size_for("vqa")
        print(f"[E6] device={device}  tier={env.gpu_tier()}  batch_size={bs}")
        progress.step(pbar, f"Environment checked: device={device}")

        master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
        staging.ensure_images()
        master = staging.add_image_paths(master)

        records = master[["image", "split", "question", "answers",
                          "answerable", "image_path"]].copy()
        records["answerable"] = records["answerable"].astype(int)
        records = records.to_dict("records")
        progress.step(pbar, f"{len(records)} VQA records prepared")

        os.makedirs(config.RESULTS_E6, exist_ok=True)
        df = vqa_confidence.harvest(
            records=records,
            out_parquet=out_path,
            model_id=config.VQA_MODEL_ID,
            device=device,
            bs=bs,
            force=config.FORCE_RERUN,
        )
        progress.step(pbar, "ViLT confidence harvest completed")

    print(f"[E6] {len(df)} rows harvested")
    print(f"  mean VQA accuracy:    {df['correct'].mean():.4f}")
    print(f"  mean VQA confidence:  {df['confidence'].mean():.4f}")
    print(f"  answerable rate:      {df['answerable'].mean():.4f}")
    progress.dataframe_summary("vqa_predictions", df)
    progress.step(pbar, "VQA summary printed")

    resultlog.log_run(EXP,
                      metrics={
                          "n_samples": len(df),
                          "mean_accuracy": float(df["correct"].mean()),
                          "mean_confidence": float(df["confidence"].mean()),
                      },
                      params={"model": config.VQA_MODEL_ID},
                      results_dir=config.RESULTS_E6, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E6, artifacts=[out_path])
    progress.step(pbar, "E6 result logged")
    pbar.close()

    if config.AUTO_DISCONNECT:
        from google.colab import runtime
        runtime.unassign()
    print("[E6 DONE] E7/E8 run on CPU or any GPU.")
