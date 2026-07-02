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

# ── Field-name map (verified against the REAL VizWiz downloads, 2026-07) ────
# QualityIssues record: {"image": "VizWiz_train_....jpg",
#                        "flaws": {"BLR": 5, "FRM": 1, "DRK": 0, "BRT": 3,
#                                  "OBS": 0, "ROT": 0, "OTH": 0, "NON": 0},
#                        "unrecognizable": 4}
# Values are VOTE COUNTS from 5 crowdworkers. Paper (arXiv:2003.12511):
# "We deemed a label as valid only if at least two crowdworkers chose that
# label." -> binarize at >= MIN_VOTES when values look like counts.
FIELD_MAP_QUALITY = {
    "image":         "image",           # filename key
    "flaws":         "flaws",           # dict abbreviation -> vote count
    "unrecognizable":"unrecognizable",  # vote count 0-5
}
# VQA record: {"image": ..., "question": ..., "answer_type": ...,
#              "answerable": 0/1, "answers": [{"answer": str,
#              "answer_confidence": str}] * 10}
FIELD_MAP_VQA = {
    "image":      "image",
    "question":   "question",
    "answerable": "answerable",
    "answers":    "answers",            # list of {"answer": str, ...}
}

# Binarization threshold when quality labels are crowd vote counts (0-5).
MIN_VOTES = 2


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
    """Normalize VizWiz quality flaw encodings to {canonical_flaw: numeric}.
    Values are kept numeric (vote counts or 0/1); binarization happens in
    load_quality once we know whether the dataset uses counts."""
    aliases = {
        # Real VizWiz-QualityIssues abbreviations (the download uses THESE):
        "blr": "blur",
        "brt": "bright",
        "drk": "dark",
        "obs": "obstruction",
        "frm": "framing",
        "rot": "rotation",
        # "oth" (other) and "non" (no flaw) are not among the 6 canonical flaws.
        # Long-form spellings, for robustness against schema variants:
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
            return {name: float(raw_flaws[i])
                    for i, name in enumerate(QUALITY_FLAWS[:len(raw_flaws)])}
        for f in raw_flaws:
            key = aliases.get(str(f).strip().lower(), str(f).strip().lower())
            flaws[key] = 1.0
    elif isinstance(raw_flaws, dict):
        for k, v in raw_flaws.items():
            key = aliases.get(str(k).strip().lower(), str(k).strip().lower())
            try:
                flaws[key] = float(v)
            except (TypeError, ValueError):
                flaws[key] = float(_as_binary(v))
    return flaws


def load_quality(split_json: str, split: str) -> pd.DataFrame:
    """
    Load one QualityIssues split JSON -> DataFrame with columns:
    image, q_blur, q_bright, q_dark, q_obstruction, q_framing, q_rotation,
    q_unrecognizable, split.

    Real VizWiz values are 5-crowdworker VOTE COUNTS; per the dataset paper a
    label is positive iff >= MIN_VOTES (2). If the values are already binary
    (max <= 1, e.g. synthetic tests), they are used as-is.
    """
    data = _records(json.load(open(split_json)), split=split)

    parsed = []
    max_flaw_val = 0.0
    max_unrec_val = 0.0
    for it in data:
        image = _image_name(_get(it, FIELD_MAP_QUALITY["image"], "image", "image_id", "file_name"))
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
        try:
            unrec_val = float(unrec_raw)
        except (TypeError, ValueError):
            unrec_val = float(_as_binary(unrec_raw))

        vals = [flaws.get(flaw, 0.0) for flaw in QUALITY_FLAWS]
        max_flaw_val = max(max_flaw_val, max(vals) if vals else 0.0)
        max_unrec_val = max(max_unrec_val, unrec_val)
        parsed.append((image, vals, unrec_val))

    # Vote counts (0-5) vs already-binary labels.
    flaw_thresh = MIN_VOTES if max_flaw_val > 1 else 1
    unrec_thresh = MIN_VOTES if max_unrec_val > 1 else 1
    print(f"[data_assembly] quality[{split}]: flaw values max={max_flaw_val:.0f} "
          f"-> positive iff >= {flaw_thresh}; unrecognizable max={max_unrec_val:.0f} "
          f"-> positive iff >= {unrec_thresh}")

    rows = []
    for image, vals, unrec_val in parsed:
        rows.append({
            "image": image,
            **{f"q_{flaw}": int(v >= flaw_thresh)
               for flaw, v in zip(QUALITY_FLAWS, vals)},
            "q_unrecognizable": int(unrec_val >= unrec_thresh),
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
