"""
E9 (Phase 2) - Model-agnostic grounding interface.

Primary:  NVIDIA LocateAnything-3B (LA-3B) with its REC prompt template.
Fallback: Qwen2.5-VL-3B-Instruct - same ground() signature, different backend.

The harvest loop in E9 calls only ground() and groundability_features().
Switching the grounder does NOT require changing the harvest loop - set
config.GROUNDER = "qwen25vl" to activate the fallback.

Negative-Block detection: LA-3B emits a learned "no valid target" abstention.
We detect this and set grounded=False.
"""
import os
import json
import shutil
import numpy as np
import torch
from PIL import Image

from src import config

_MODEL_CACHE = {}   # singleton cache so we don't reload on every call


GROUNDING_FEATURE_COLS = [
    "grounded", "n_boxes", "max_conf", "box_area_frac",
    "touches_border", "centeredness",
]


def _atomic_json_dump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_parquet_write(df, path: str) -> None:
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _quarantine(path: str, reason: str) -> None:
    bad = path + ".corrupt"
    print(f"[grounding] corrupt checkpoint ignored: {path} ({reason})")
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
        print(f"[grounding] warning: could not quarantine {path}: {exc}")


def _validate_final_grounding(path: str, expected_ids):
    import pandas as pd
    required = {"global_idx", "phrase", "image"} | set(GROUNDING_FEATURE_COLS)
    expected_ids = {int(i) for i in expected_ids}
    try:
        df = pd.read_parquet(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        ids = [int(x) for x in df["global_idx"].tolist()]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate global_idx rows")
        if set(ids) != expected_ids:
            raise ValueError(f"id set mismatch: got {len(set(ids))}, expected {len(expected_ids)}")
        return df
    except Exception as exc:
        _quarantine(path, str(exc))
        return None


def _valid_grounding_shards(shard_dir: str, expected_ids):
    import pandas as pd
    required = {"global_idx", "phrase", "image"} | set(GROUNDING_FEATURE_COLS)
    expected_ids = {int(i) for i in expected_ids}
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
            ids = [int(x) for x in df["global_idx"].tolist()]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate global_idx within shard")
            bad_ids = set(ids) - expected_ids
            if bad_ids:
                raise ValueError(f"unexpected global_idx, first={next(iter(bad_ids))}")
            dup = seen.intersection(ids)
            if dup:
                raise ValueError(f"duplicate global_idx across shards, first={next(iter(dup))}")
            seen.update(ids)
            valid.append(shard_path)
        except Exception as exc:
            _quarantine(shard_path, str(exc))
    return valid, seen


# ── Ground function ──────────────────────────────────────────────────────────

@torch.inference_mode()
def ground(
    image: Image.Image,
    phrase: str,
    device: str = "cuda",
    grounder: str = None,
) -> dict:
    """
    Locate `phrase` in `image`.
    Returns:
        {
          'boxes':   [[x1,y1,x2,y2], ...],  # normalised to [0,1000]
          'conf':    float,                  # max grounding confidence
          'grounded': bool,                  # False = Negative-Block / no target
        }
    """
    grounder = grounder or config.GROUNDER
    if grounder == "locate_anything":
        return _ground_locate_anything(image, phrase, device)
    elif grounder == "qwen25vl":
        return _ground_qwen25vl(image, phrase, device)
    else:
        raise ValueError(f"Unknown grounder: {grounder}")


# ── LocateAnything-3B backend ─────────────────────────────────────────────────

def _load_locate_anything(device: str):
    if "locate_anything" in _MODEL_CACHE:
        return _MODEL_CACHE["locate_anything"]
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    model_id = "nvidia/LocateAnything-3B"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    _MODEL_CACHE["locate_anything"] = (model, processor)
    return model, processor


@torch.inference_mode()
def _ground_locate_anything(image: Image.Image, phrase: str, device: str) -> dict:
    model, processor = _load_locate_anything(device)
    prompt = (
        f"Locate a single instance that matches the following description: {phrase}."
    )
    try:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device != "cpu")):
            outputs = model.generate(**inputs, max_new_tokens=128)
        decoded = processor.decode(outputs[0], skip_special_tokens=True)

        # Detect Negative-Block abstention (LA-3B's learned refusal token)
        if any(kw in decoded.lower() for kw in
               ["no valid target", "negative block", "no target", "not found"]):
            return {"boxes": [], "conf": 0.0, "grounded": False}

        boxes = _parse_boxes_la3b(decoded)
        if not boxes:
            return {"boxes": [], "conf": 0.0, "grounded": False}
        # LA-3B does not output per-box confidence; use 1.0 as presence indicator
        return {"boxes": boxes, "conf": 1.0, "grounded": True}

    except Exception as e:
        print(f"[grounding] LA-3B error: {e}")
        return {"boxes": [], "conf": 0.0, "grounded": False}


def _parse_boxes_la3b(text: str) -> list:
    """
    Parse bounding box coordinates from LA-3B output.
    LA-3B typically outputs coordinates in [0,1000] normalised space
    as comma-separated integers: x1,y1,x2,y2.
    """
    import re
    matches = re.findall(r"\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?", text)
    boxes = [[int(x) for x in m] for m in matches]
    return boxes


# ── Qwen2.5-VL-3B fallback ───────────────────────────────────────────────────

def _load_qwen25vl(device: str):
    if "qwen25vl" in _MODEL_CACHE:
        return _MODEL_CACHE["qwen25vl"]
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device).eval()
    _MODEL_CACHE["qwen25vl"] = (model, processor)
    return model, processor


@torch.inference_mode()
def _ground_qwen25vl(image: Image.Image, phrase: str, device: str) -> dict:
    model, processor = _load_qwen25vl(device)
    prompt = (
        f"Locate the bounding box of '{phrase}' in the image. "
        f"Output the box as [x1,y1,x2,y2] in coordinates from 0 to 1000. "
        f"If it is not present, output 'not found'."
    )
    try:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device != "cpu")):
            out = model.generate(**inputs, max_new_tokens=64)
        decoded = processor.decode(out[0], skip_special_tokens=True)

        if "not found" in decoded.lower():
            return {"boxes": [], "conf": 0.0, "grounded": False}

        boxes = _parse_boxes_la3b(decoded)   # same regex works
        if not boxes:
            return {"boxes": [], "conf": 0.0, "grounded": False}
        return {"boxes": boxes, "conf": 1.0, "grounded": True}

    except Exception as e:
        print(f"[grounding] Qwen2.5-VL error: {e}")
        return {"boxes": [], "conf": 0.0, "grounded": False}


# ── Groundability features ────────────────────────────────────────────────────

def groundability_features(
    g: dict,
    img_w: int,
    img_h: int,
    eps: float = 0.02,
) -> dict:
    """
    Derive scalar features from a grounding result dict.
    Coordinates are assumed to be in [0,1000] normalised space.
    """
    if not g["grounded"] or not g["boxes"]:
        return dict(grounded=0, n_boxes=0, max_conf=0.0,
                    box_area_frac=0.0, touches_border=0, centeredness=0.0)

    boxes = g["boxes"]
    n_boxes = len(boxes)
    max_conf = float(g["conf"])

    # Use the first/best box for spatial features
    x1, y1, x2, y2 = boxes[0]
    # Normalise from [0,1000] to [0,1]
    x1n, y1n, x2n, y2n = x1/1000, y1/1000, x2/1000, y2/1000
    x1n, x2n = min(x1n, x2n), max(x1n, x2n)
    y1n, y2n = min(y1n, y2n), max(y1n, y2n)

    box_area_frac = max(0.0, (x2n - x1n) * (y2n - y1n))
    touches = int(x1n < eps or y1n < eps or x2n > (1 - eps) or y2n > (1 - eps))

    cx, cy = (x1n + x2n) / 2, (y1n + y2n) / 2
    # centeredness: 1 = box center at image center; 0 = at corner
    centeredness = 1.0 - 2 * np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / np.sqrt(0.5)
    centeredness = float(np.clip(centeredness, 0.0, 1.0))

    return dict(
        grounded=1,
        n_boxes=n_boxes,
        max_conf=max_conf,
        box_area_frac=float(box_area_frac),
        touches_border=touches,
        centeredness=centeredness,
    )


# ── Entity extraction for E9 ─────────────────────────────────────────────────

def extract_entity(question: str) -> str:
    """
    Extract the primary queried noun phrase from a VizWiz question using spaCy.
    Returns the longest noun chunk (deterministic, no model call).
    Falls back to the full question if spaCy is unavailable or finds nothing.

    The method is logged so the grounding cache is reproducible.
    """
    try:
        import spacy
        if not hasattr(extract_entity, "_nlp"):
            try:
                extract_entity._nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not downloaded - fall back to simple heuristic
                extract_entity._nlp = None

        if extract_entity._nlp is not None:
            doc = extract_entity._nlp(question.lower())
            chunks = list(doc.noun_chunks)
            if chunks:
                # Prefer the longest noun chunk
                return str(max(chunks, key=lambda c: len(c.text)))
    except Exception:
        pass

    # Simple fallback: strip wh-words from the front
    import re
    phrase = re.sub(
        r"^(what|what is|what are|what color|is there|how many|can you|"
        r"tell me|where is|do you see|does this|what kind of)\s+",
        "", question.lower().strip()
    ).strip("?. ")
    return phrase if phrase else question


def harvest_grounding(
    records: list,
    out_parquet: str,
    device: str = "cuda",
    grounder: str = None,
    force: bool = False,
    shard_rows: int = 200,
):
    """
    Resume-safe E9 grounding harvest.

    records: list of dicts with keys:
      global_idx, image, image_path, phrase

    A rerun validates final cache and all shard files, quarantines corrupt
    checkpoints, and only grounds missing global_idx rows.
    """
    import pandas as pd
    from tqdm.auto import tqdm

    expected_ids = [int(r["global_idx"]) for r in records]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("[grounding] records contain duplicate global_idx values")

    shard_dir = out_parquet + ".shards"
    done_file = out_parquet + ".done.json"
    if force:
        for p in (out_parquet, done_file):
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(shard_dir):
            shutil.rmtree(shard_dir)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)

    if os.path.exists(out_parquet) and not force:
        cached = _validate_final_grounding(out_parquet, expected_ids)
        if cached is not None:
            print(f"[grounding] cache hit -> {out_parquet}")
            return cached
        print("[grounding] final cache was invalid; resuming/rebuilding from shards")

    valid_shards, done_ids = _valid_grounding_shards(shard_dir, expected_ids)
    _atomic_json_dump(sorted(done_ids), done_file)
    pending = [r for r in records if int(r["global_idx"]) not in done_ids]
    print(f"[grounding] resume state: {len(done_ids)}/{len(records)} rows complete, "
          f"{len(pending)} remaining")

    existing = [
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(shard_dir)
        if f.startswith("shard_") and f.endswith(".parquet")
    ]
    shard_count = (max(existing) + 1) if existing else 0
    rows_buf = []

    def _flush():
        nonlocal shard_count
        if not rows_buf:
            return
        df_shard = pd.DataFrame(rows_buf)
        shard_path = os.path.join(shard_dir, f"shard_{shard_count:06d}.parquet")
        _atomic_parquet_write(df_shard, shard_path)
        done_ids.update(int(x) for x in df_shard["global_idx"].tolist())
        _atomic_json_dump(sorted(done_ids), done_file)
        shard_count += 1
        rows_buf.clear()

    for rec in tqdm(pending, desc="[grounding]", unit="img"):
        phrase = rec["phrase"]
        img_path = rec["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
        except Exception:
            img, w, h = None, 224, 224

        if img is not None:
            g = ground(img, phrase, device=device, grounder=grounder)
        else:
            g = {"boxes": [], "conf": 0.0, "grounded": False}

        feat = groundability_features(g, w, h)
        rows_buf.append({
            "global_idx": int(rec["global_idx"]),
            "phrase": phrase,
            "image": rec.get("image", ""),
            **feat,
        })
        if len(rows_buf) >= shard_rows:
            _flush()
    _flush()

    valid_shards, done_ids = _valid_grounding_shards(shard_dir, expected_ids)
    if len(done_ids) != len(expected_ids):
        missing = sorted(set(expected_ids) - done_ids)[:10]
        raise RuntimeError(
            f"[grounding] cannot assemble: {len(expected_ids)-len(done_ids)} rows "
            f"still missing. First missing global_idx values: {missing}"
        )
    df = pd.concat([pd.read_parquet(p) for p in valid_shards], ignore_index=True)
    order = {int(gid): i for i, gid in enumerate(expected_ids)}
    df["_order"] = df["global_idx"].map(order)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    _atomic_parquet_write(df, out_parquet)
    print(f"[grounding] saved {len(df)} rows -> {out_parquet}")
    return df
