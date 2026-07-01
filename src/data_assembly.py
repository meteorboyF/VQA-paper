"""
E1 - Master data assembly.

Joins VizWiz-VQA annotations with VizWiz-QualityIssues annotations into a
single master.parquet keyed by (image, split).

IMPORTANT: field names are verified against the E0 audit output. If E0 prints
different field names, update FIELD_MAP below before running E1.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Six quality flaws from VizWiz-QualityIssues
QUALITY_FLAWS = ["blur", "bright", "dark", "obstruction", "framing", "rotation"]

# ── Field-name map (populated from E0 audit; update if your JSON differs) ────
# Keys = canonical name used in this codebase.
# Values = actual JSON field name found in the annotation file.
# E0 will print the real names; adapt here before running E1.
FIELD_MAP_QUALITY = {
    "image":         "image",           # filename key
    "flaws":         "quality_flaws",   # dict/list of flaw names (may vary)
    "unrecognizable":"unrecognizable",  # bool / 0-1
}
FIELD_MAP_VQA = {
    "image":      "image",
    "question":   "question",
    "answerable": "answerable",
    "answers":    "answers",            # list of {"answer": str, ...}
}


def _get(obj: dict, *keys, default=None):
    """Try multiple key names; return first hit."""
    for k in keys:
        if k in obj:
            return obj[k]
    return default


def _records(obj, split: str = None):
    """
    Return a list of annotation records from either the raw VizWiz list format
    or compiled dict-of-lists formats used by VizWiz-QualityIssues helpers.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("annotations", "data", "items", "records"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if split and isinstance(obj.get(split), dict):
            return _records(obj[split], split=None)
        # Compiled format: {"image": [...], "flaws": [...], ...}
        if "image" in obj and isinstance(obj["image"], list):
            n = len(obj["image"])
            out = []
            for i in range(n):
                row = {}
                for k, v in obj.items():
                    row[k] = v[i] if isinstance(v, list) and len(v) == n else v
                out.append(row)
            return out
    raise ValueError("Unsupported annotation JSON structure; inspect E0 schema output.")


def _image_name(value) -> str:
    """Normalize image identifiers to the basename used by the image zips."""
    return os.path.basename(str(value))


def _as_binary(value) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return int(bool(value))


def _flaws_to_dict(raw_flaws) -> dict:
    """Normalize VizWiz quality flaw encodings to {canonical_flaw: 0/1}."""
    aliases = {
        "blur": "blur",
        "blurry": "blur",
        "bright": "bright",
        "overexposure": "bright",
        "overexposed": "bright",
        "dark": "dark",
        "underexposure": "dark",
        "underexposed": "dark",
        "obstruction": "obstruction",
        "obstructed": "obstruction",
        "framing": "framing",
        "improper framing": "framing",
        "rotation": "rotation",
        "rotated": "rotation",
    }
    flaws = {}
    if isinstance(raw_flaws, list):
        if all(isinstance(x, (int, float, bool, np.integer, np.floating)) for x in raw_flaws):
            return {name: _as_binary(raw_flaws[i]) for i, name in enumerate(QUALITY_FLAWS[:len(raw_flaws)])}
        for f in raw_flaws:
            key = aliases.get(str(f).strip().lower(), str(f).strip().lower())
            flaws[key] = 1
    elif isinstance(raw_flaws, dict):
        for k, v in raw_flaws.items():
            key = aliases.get(str(k).strip().lower(), str(k).strip().lower())
            flaws[key] = _as_binary(v)
    return flaws


def load_quality(split_json: str, split: str) -> pd.DataFrame:
    """
    Load one QualityIssues split JSON -> DataFrame with columns:
    image, q_blur, q_bright, q_dark, q_obstruction, q_framing, q_rotation,
    q_unrecognizable, split.
    """
    data = _records(json.load(open(split_json)), split=split)
    rows = []
    for it in data:
        image = _image_name(_get(it, FIELD_MAP_QUALITY["image"], "image", "image_id", "file_name"))
        # The flaws field may be a dict {flaw: 0/1} or a list of flaw names -
        # handle both to be robust against schema variations.
        raw_flaws = _get(it,
                         FIELD_MAP_QUALITY["flaws"],
                         "flaws", "quality_flaws", "flaw",
                         default={})
        flaws = _flaws_to_dict(raw_flaws)

        unrec_raw = _get(it,
                         FIELD_MAP_QUALITY["unrecognizable"],
                         "unrecognizable", "not_recognizable", "unrecog",
                         "recognizable",
                         default=0)
        if "recognizable" in it and FIELD_MAP_QUALITY["unrecognizable"] not in it:
            unrec_raw = 1 - _as_binary(unrec_raw)
        rows.append({
            "image": image,
            **{f"q_{flaw}": _as_binary(flaws.get(flaw, 0)) for flaw in QUALITY_FLAWS},
            "q_unrecognizable": _as_binary(unrec_raw),
            "split": split,
        })
    return pd.DataFrame(rows)


def load_vqa(split_json: str, split: str) -> pd.DataFrame:
    """
    Load one VQA split JSON -> DataFrame with columns:
    image, question, answerable, answers, split.
    """
    data = _records(json.load(open(split_json)), split=split)
    rows = []
    for it in data:
        image = _image_name(_get(it, FIELD_MAP_VQA["image"], "image", "image_id", "file_name"))
        question = _get(it, FIELD_MAP_VQA["question"], "question", default="")
        answerable = _as_binary(_get(it, FIELD_MAP_VQA["answerable"], "answerable", default=1))
        raw_ans = _get(it, FIELD_MAP_VQA["answers"], "answers", default=[])
        answers = [a["answer"] if isinstance(a, dict) else str(a) for a in raw_ans]
        rows.append({
            "image":      image,
            "question":   question,
            "answerable": answerable,
            "answers":    answers,
            "split":      split,
        })
    return pd.DataFrame(rows)


def build_master(
    vqa_paths: dict,
    quality_paths: dict,
    out_path: str,
) -> pd.DataFrame:
    """
    vqa_paths     = {"train": path, "val": path, ...}
    quality_paths = {"train": path, "val": path, ...}
    Inner-join on (image, split) so we only keep images with BOTH label sets.
    """
    vqa_frames = [load_vqa(p, s) for s, p in vqa_paths.items()]
    qua_frames = [load_quality(p, s) for s, p in quality_paths.items()]
    vqa = pd.concat(vqa_frames, ignore_index=True)
    qua = pd.concat(qua_frames, ignore_index=True)

    # Deduplicate (same image may appear multiple times in VQA with different questions)
    # Keep all rows - each row is one (image, question) pair.
    master = vqa.merge(qua, on=["image", "split"], how="inner")
    master = master.reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    master.to_parquet(out_path, index=False)
    print(f"[data_assembly] master.parquet written: {len(master)} rows -> {out_path}")
    return master


def label_stats(master: pd.DataFrame, out_path: str) -> dict:
    """
    Compute and save per-label positive rates, co-occurrence matrix,
    and answerable x defect contingency table.
    """
    import json

    flaw_cols = [f"q_{f}" for f in QUALITY_FLAWS] + ["q_unrecognizable"]
    stats = {}

    # Per-label positive rates
    rates = {}
    for col in flaw_cols + ["answerable"]:
        rates[col] = float(master[col].mean())
    stats["positive_rates"] = rates

    # Co-occurrence matrix (fraction of images that have BOTH defects)
    cooccur = {}
    for i, ci in enumerate(flaw_cols):
        for j, cj in enumerate(flaw_cols):
            if j >= i:
                key = f"{ci}x{cj}"
                cooccur[key] = float((master[ci] & master[cj]).mean())
    stats["cooccurrence"] = cooccur

    # Answerable x defect contingency
    contingency = {}
    for col in flaw_cols:
        ct = pd.crosstab(master["answerable"], master[col])
        contingency[col] = ct.to_dict()
    stats["contingency_answerable_x_defect"] = contingency

    # Split counts
    stats["split_counts"] = master["split"].value_counts().to_dict()
    stats["total_rows"] = int(len(master))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[data_assembly] label_stats.json -> {out_path}")
    return stats
