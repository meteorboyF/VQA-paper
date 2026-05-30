"""
E6 — Frozen VQA confidence harvest using ViLT.

Runs dandelin/vilt-b32-finetuned-vqa (discriminative, clean softmax logits)
over the dataset once and caches: predicted answer, max-softmax confidence,
and VizWiz VQA accuracy against the 10 ground-truth answers.

Idempotent: skips if the output parquet already exists.
Checkpoints every SHARD_ROWS rows for resume safety.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

SHARD_ROWS = 512     # flush a shard every this many rows


# ── VizWiz VQA accuracy ───────────────────────────────────────────────────────

def vqa_accuracy(pred: str, answers: list) -> float:
    """Standard VizWiz/VQA accuracy: min(#exact_matches / 3, 1)."""
    pred = pred.strip().lower()
    matches = sum(1 for a in answers if a.strip().lower() == pred)
    return min(matches / 3.0, 1.0)


# ── Main harvest ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def harvest(
    records: list,           # list of dicts with 'image_path', 'question', 'answers'
    out_parquet: str,
    model_id: str = "dandelin/vilt-b32-finetuned-vqa",
    device: str = "cuda",
    bs: int = 32,
    force: bool = False,
) -> pd.DataFrame:
    """
    records  — list of {image_path, question, answers, image, split, ...}
    Returns  — DataFrame with original fields + {pred, confidence, correct}
    Checkpoints to out_parquet + ".shards/" every SHARD_ROWS rows.
    """
    if os.path.exists(out_parquet) and not force:
        print(f"[vqa_confidence] cache hit → {out_parquet}")
        return pd.read_parquet(out_parquet)

    shard_dir  = out_parquet + ".shards"
    done_file  = out_parquet + ".done.json"
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    os.makedirs(shard_dir,  exist_ok=True)

    done_ids = set()
    if os.path.exists(done_file):
        done_ids = set(json.load(open(done_file)))

    from transformers import ViltProcessor, ViltForQuestionAnswering
    proc  = ViltProcessor.from_pretrained(model_id)
    model = ViltForQuestionAnswering.from_pretrained(model_id).to(device).eval()

    pending = [r for i, r in enumerate(records) if i not in done_ids]
    pending_idx = [i for i in range(len(records)) if i not in done_ids]

    rows_buf, shard_count = [], 0

    def _flush():
        nonlocal shard_count
        if not rows_buf:
            return
        df_shard = pd.DataFrame(rows_buf)
        shard_path = os.path.join(shard_dir, f"shard_{shard_count:06d}.parquet")
        df_shard.to_parquet(shard_path, index=False)
        done_ids.update(df_shard["_row_id"].tolist())
        json.dump(list(done_ids), open(done_file, "w"))
        shard_count += 1
        rows_buf.clear()

    for batch_start in tqdm(range(0, len(pending), bs), desc="[vqa_conf]"):
        batch = pending[batch_start: batch_start + bs]
        batch_orig_idx = pending_idx[batch_start: batch_start + bs]

        images    = []
        questions = []
        meta      = []
        for rec in batch:
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            images.append(img)
            questions.append(rec["question"])
            meta.append(rec)

        try:
            enc = proc(images, questions, return_tensors="pt",
                       padding=True, truncation=True, max_length=40).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device != "cpu")):
                logits = model(**enc).logits          # (B, n_answers)
            prob = logits.softmax(-1)
            confs, idxs = prob.max(-1)
            confs  = confs.float().cpu().numpy()
            preds  = [model.config.id2label[j] for j in idxs.cpu().tolist()]
        except Exception as e:
            print(f"[vqa_confidence] batch error: {e}; filling with NaN")
            confs = [float("nan")] * len(batch)
            preds = [""] * len(batch)

        for rec, orig_i, pred, conf in zip(meta, batch_orig_idx, preds, confs):
            answers = rec.get("answers", [])
            acc = vqa_accuracy(pred, answers) if answers else float("nan")
            row = {k: v for k, v in rec.items() if k != "answers"}
            row.update({
                "_row_id":    orig_i,
                "pred":       pred,
                "confidence": float(conf),
                "correct":    acc,
            })
            rows_buf.append(row)

        if len(rows_buf) >= SHARD_ROWS:
            _flush()

    _flush()

    # Assemble all shards
    shards = sorted(
        [os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
         if f.endswith(".parquet")]
    )
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df = df.drop(columns=["_row_id"], errors="ignore")
    df.to_parquet(out_parquet, index=False)
    print(f"[vqa_confidence] saved {len(df)} rows → {out_parquet}")
    return df
