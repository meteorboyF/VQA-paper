"""E6b - Second frozen VQA model: BLIP confidence harvest (GPU).

The paper's selective-prediction conclusions currently rest on ONE frozen VQA
model (ViLT, 18.5% VizWiz accuracy). The obvious reviewer objection is that
"confidence dominates and defects add nothing" might be an artifact of a weak
model. E6b harvests answers + confidences from a second, generative frozen
model (BLIP-VQA-base); confidence is the length-normalized sequence
probability (geometric mean of generated-token probabilities). E7d then
reruns the full diagnostic battery against it.

Output parquet has the same schema as E6, cached on Drive, shard-resumable.
"""
import os

import pandas as pd

from src import config, env, expstate, progress, resultlog, staging, vqa_confidence

EXP = "E6B"
RESULTS_E6B = os.path.join(config.RESULTS, "E6b_vqaconf_blip")


def out_path():
    return os.path.join(RESULTS_E6B, "vqa_predictions_blip.parquet")


def main():
    progress.install_error_hook("E6b BLIP confidence harvest")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E6B, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E6B, required=[out_path()]):
        expstate.skip_banner(EXP, RESULTS_E6B)
        return

    pbar = progress.notebook_bar("E6b BLIP confidence harvest", total=5)

    if os.path.exists(out_path()) and not config.FORCE_RERUN:
        print(f"[E6b] cache hit: {out_path()}")
        df = pd.read_parquet(out_path())
        progress.step(pbar, "BLIP prediction cache loaded", advance=3)
    else:
        env.check_gpu(EXP)  # raises on CPU runtime
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Generation holds decoder state; use half the discriminative batch.
        bs = max(4, config.batch_size_for("vqa") // 2)
        print(f"[E6b] device={device}  tier={env.gpu_tier()}  batch_size={bs}  "
              f"model={config.VQA_MODEL_ID_2}")
        progress.step(pbar, f"Environment checked: device={device}")

        master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
        staging.ensure_images()
        master = staging.add_image_paths(master)

        records = master[["image", "split", "question", "answers",
                          "answerable", "image_path"]].copy()
        records["answerable"] = records["answerable"].astype(int)
        records = records.to_dict("records")
        progress.step(pbar, f"{len(records)} VQA records prepared")

        df = vqa_confidence.harvest_generative(
            records=records,
            out_parquet=out_path(),
            model_id=config.VQA_MODEL_ID_2,
            device=device,
            bs=bs,
            force=config.FORCE_RERUN,
        )
        progress.step(pbar, "BLIP confidence harvest completed")

    print(f"[E6b] {len(df)} rows harvested")
    print(f"  mean VizWiz accuracy:  {df['correct'].mean():.4f}")
    print(f"  mean BLIP confidence:  {df['confidence'].mean():.4f}")
    progress.dataframe_summary("vqa_predictions_blip", df)
    progress.step(pbar, "BLIP summary printed")

    resultlog.log_run(EXP,
                      metrics={
                          "n_samples": len(df),
                          "mean_accuracy": float(df["correct"].mean()),
                          "mean_confidence": float(df["confidence"].mean()),
                      },
                      params={"model": config.VQA_MODEL_ID_2},
                      results_dir=RESULTS_E6B, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E6B, artifacts=[out_path()])
    progress.step(pbar, "E6b result logged")
    pbar.close()

    if config.AUTO_DISCONNECT:
        from google.colab import runtime
        runtime.unassign()
    print("[E6b DONE] Run E7d (CPU) next for the BLIP diagnostic battery.")
