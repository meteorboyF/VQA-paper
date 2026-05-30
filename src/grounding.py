"""
E9 (Phase 2) — Model-agnostic grounding interface.

Primary:  NVIDIA LocateAnything-3B (LA-3B) with its REC prompt template.
Fallback: Qwen2.5-VL-3B-Instruct — same ground() signature, different backend.

The harvest loop in E9 calls only ground() and groundability_features().
Switching the grounder does NOT require changing the harvest loop — set
config.GROUNDER = "qwen25vl" to activate the fallback.

Negative-Block detection: LA-3B emits a learned "no valid target" abstention.
We detect this and set grounded=False.
"""
import os
import json
import numpy as np
import torch
from PIL import Image

from src import config

_MODEL_CACHE = {}   # singleton cache so we don't reload on every call


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
                # Model not downloaded — fall back to simple heuristic
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
