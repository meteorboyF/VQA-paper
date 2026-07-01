"""
E6 - Frozen VQA confidence harvest using ViLT.

Runs dandelin/vilt-b32-finetuned-vqa (discriminative, clean softmax logits)
over the dataset once and caches: predicted answer, max-softmax confidence,
and VizWiz VQA accuracy against the 10 ground-truth answers.

Idempotent: skips if the output parquet already exists.
Checkpoints every SHARD_ROWS rows for resume safety.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

SHARD_ROWS = 512     # flush a shard every this many rows


def _atomic_json_dump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_parquet_write(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _quarantine(path: str, reason: str) -> None:
    bad = path + ".corrupt"
    print(f"[vqa_confidence] corrupt checkpoint ignored: {path} ({reason})")
    try:
        if os.path.isdir(path):
            if os.path.exists(bad):
                shutil.rmtree(bad, ignore_errors=True)
            shutil.move(path, bad)
        elif os.path.exists(path):
            if os.path.exists(bad):
                os.remove(bad)
            os.replace(path, bad)
    except Exception as exc:
        print(f"[vqa_confidence] warning: could not quarantine {path}: {exc}")


def _validate_final_parquet(path: str, n: int):
    required = {"image", "split", "question", "image_path", "pred", "confidence", "correct"}
    try:
        df = pd.read_parquet(path)
        if len(df) != n:
            raise ValueError(f"row count {len(df)} != expected {n}")
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        return df
    except Exception as exc:
        _quarantine(path, str(exc))
        return None


def _valid_prediction_shards(shard_dir: str, n: int):
    required = {"_row_id", "image", "split", "question", "image_path", "pred", "confidence", "correct"}
    valid = []
    seen = set()
    for shard_f in sorted(os.listdir(shard_dir)):
        if not shard_f.startswith("shard_") or not shard_f.endswith(".parquet"):
            continue
        shard_path = os.path.join(shard_dir, shard_f)
        try:
            df = pd.read_parquet(shard_path)
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"missing columns {sorted(missing)}")
            ids = [int(x) for x in df["_row_id"].tolist()]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate _row_id within shard")
            if any(i < 0 or i >= n for i in ids):
                raise ValueError("_row_id out of range")
            dup = seen.intersection(ids)
            if dup:
                raise ValueError(f"duplicate _row_id across shards, first={next(iter(dup))}")
            seen.update(ids)
            valid.append(shard_path)
        except Exception as exc:
            _quarantine(shard_path, str(exc))
    return valid, seen


# ── VizWiz VQA accuracy ───────────────────────────────────────────────────────

def vqa_accuracy(pred: str, answers: list) -> float:
    """Standard VizWiz/VQA accuracy: min(#exact_matches / 3, 1)."""
    pred = str(pred).strip().lower()
    matches = sum(1 for a in answers if str(a).strip().lower() == pred)
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
    records  - list of {image_path, question, answers, image, split, ...}
    Returns  - DataFrame with original fields + {pred, confidence, correct}
    Checkpoints to out_parquet + ".shards/" every SHARD_ROWS rows.
    """
    shard_dir  = out_parquet + ".shards"
    done_file  = out_parquet + ".done.json"
    if force:
        for p in (out_parquet, done_file):
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(shard_dir):
            shutil.rmtree(shard_dir)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    os.makedirs(shard_dir,  exist_ok=True)

    if os.path.exists(out_parquet) and not force:
        cached = _validate_final_parquet(out_parquet, len(records))
        if cached is not None:
            print(f"[vqa_confidence] cache hit -> {out_parquet}")
            return cached
        print("[vqa_confidence] final cache was invalid; resuming/rebuilding from shards")

    valid_shards, done_ids = _valid_prediction_shards(shard_dir, len(records))
    _atomic_json_dump(sorted(done_ids), done_file)
    print(f"[vqa_confidence] resume state: {len(done_ids)}/{len(records)} rows complete, "
          f"{len(records)-len(done_ids)} remaining")

    pending = [r for i, r in enumerate(records) if i not in done_ids]
    pending_idx = [i for i in range(len(records)) if i not in done_ids]
    proc = model = None

    existing = [
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(shard_dir)
        if f.startswith("shard_") and f.endswith(".parquet")
    ]
    rows_buf = []
    shard_count = (max(existing) + 1) if existing else 0

    def _flush():
        nonlocal shard_count
        if not rows_buf:
            return
        df_shard = pd.DataFrame(rows_buf)
        shard_path = os.path.join(shard_dir, f"shard_{shard_count:06d}.parquet")
        _atomic_parquet_write(df_shard, shard_path)
        done_ids.update(df_shard["_row_id"].tolist())
        _atomic_json_dump(sorted(int(i) for i in done_ids), done_file)
        shard_count += 1
        rows_buf.clear()

    if pending:
        from transformers import ViltProcessor, ViltForQuestionAnswering
        proc = ViltProcessor.from_pretrained(model_id)
        model = ViltForQuestionAnswering.from_pretrained(model_id).to(device).eval()

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
    shards, done_ids = _valid_prediction_shards(shard_dir, len(records))
    if not shards:
        raise RuntimeError("[vqa_confidence] no shards were written; check input records.")
    if len(done_ids) != len(records):
        missing = sorted(set(range(len(records))) - done_ids)[:10]
        raise RuntimeError(
            f"[vqa_confidence] cannot assemble: {len(records)-len(done_ids)} rows "
            f"still missing. First missing rows: {missing}"
        )
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df = df.sort_values("_row_id").reset_index(drop=True)
    df = df.drop(columns=["_row_id"], errors="ignore")
    _atomic_parquet_write(df, out_parquet)
    print(f"[vqa_confidence] saved {len(df)} rows -> {out_parquet}")
    return df
