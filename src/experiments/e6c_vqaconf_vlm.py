"""E6c - Third frozen answerer: modern VLM confidence harvest (GPU, optional).

Review Major #2: ViLT + BLIP-VQA-base cannot support a general
discriminative-versus-generative claim, and both are weak by 2026 standards.
E6c harvests a contemporary open VLM as a third frozen answerer so the
model-specificity of the selective-prediction finding can be tested against
a stronger confidence channel. Default model: BLIP-2 (Salesforce
blip2-opt-2.7b); override with VQA_MODEL_ID_3, e.g. a Qwen2-VL variant.

Confidence is the same length-normalized sequence probability used for BLIP
(geometric mean of generated-token probabilities under greedy decoding), so
the three answerers are compared under one confidence definition.

Output parquet has the same schema as E6/E6b, cached on Drive,
shard-resumable. After it completes, rerun the E7d diagnostic battery with
GATE="vlm" (the E5c/E7e scripts pick the parquet up automatically once it
exists under results/E6c_vqaconf_vlm/).

Runtime: A100 strongly recommended (~2-4x the E6b wall-clock); fp16 weights.
"""
import os

import pandas as pd

from src import config, env, expstate, progress, resultlog, staging, vqa_confidence

EXP = "E6C"
RESULTS_E6C = os.path.join(config.RESULTS, "E6c_vqaconf_vlm")


def out_path():
    return os.path.join(RESULTS_E6C, "vqa_predictions_vlm.parquet")


def _blip2_scorer(model_id: str, device: str):
    """Batch scorer for BLIP-2-style conditional-generation VLMs."""
    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device).eval()
    pad_id = proc.tokenizer.pad_token_id or 0

    @torch.inference_mode()
    def score(images, questions):
        prompts = [f"Question: {q} Short answer:" for q in questions]
        enc = proc(images=images, text=prompts,
                   return_tensors="pt", padding=True).to(device, torch.float16)
        out = model.generate(**enc, max_new_tokens=10,
                             output_scores=True,
                             return_dict_in_generate=True)
        trans = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True)
        gen_tokens = out.sequences[:, -trans.shape[1]:]
        mask = gen_tokens != pad_id
        tok_lp = trans.masked_fill(~mask, 0.0).float()
        n_tok = mask.sum(-1).clamp(min=1)
        confs = torch.exp(tok_lp.sum(-1) / n_tok).cpu().numpy()
        preds = proc.batch_decode(out.sequences[:, -trans.shape[1]:],
                                  skip_special_tokens=True)
        preds = [p.strip() for p in preds]
        return preds, confs

    return score


def main():
    progress.install_error_hook("E6c VLM confidence harvest")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E6C, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E6C, required=[out_path()]):
        expstate.skip_banner(EXP, RESULTS_E6C)
        return

    pbar = progress.notebook_bar("E6c VLM confidence harvest", total=5)

    if os.path.exists(out_path()) and not config.FORCE_RERUN:
        print(f"[E6c] cache hit: {out_path()}")
        df = pd.read_parquet(out_path())
        progress.step(pbar, "VLM prediction cache loaded", advance=3)
    else:
        env.check_gpu("E9")  # heaviest tier: refuse CPU, prefer A100
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = config.VQA_MODEL_ID_3
        # 2.7B decoder states are heavy; quarter of the discriminative batch.
        bs = max(2, config.batch_size_for("vqa") // 4)
        print(f"[E6c] device={device}  tier={env.gpu_tier()}  batch_size={bs}  "
              f"model={model_id}")
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
            model_id=model_id,
            device=device,
            bs=bs,
            force=config.FORCE_RERUN,
            scorer=_blip2_scorer(model_id, device),
        )
        progress.step(pbar, "VLM confidence harvest completed")

    print(f"[E6c] {len(df)} rows harvested")
    print(f"  mean VizWiz accuracy: {df['correct'].mean():.4f}")
    print(f"  mean VLM confidence:  {df['confidence'].mean():.4f}")
    progress.dataframe_summary("vqa_predictions_vlm", df)
    progress.step(pbar, "VLM summary printed")

    resultlog.log_run(EXP,
                      metrics={"n_samples": len(df),
                               "mean_accuracy": float(df["correct"].mean()),
                               "mean_confidence": float(df["confidence"].mean())},
                      params={"model": config.VQA_MODEL_ID_3},
                      results_dir=RESULTS_E6C, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E6C, artifacts=[out_path()])
    progress.step(pbar, "E6c result logged")
    pbar.close()

    if config.AUTO_DISCONNECT:
        from google.colab import runtime
        runtime.unassign()
    print("[E6c DONE] Rerun the diagnostic battery against the VLM gate next.")
