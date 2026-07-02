"""
Data staging shared by E0/E1/E2/E6/E9.

Centralizes:
  - Drive zip auto-discovery (finds the right zip even if names differ),
  - staging (copy Drive zip -> local disk -> unzip once, marker-guarded),
  - annotation JSON location (one candidate list, used by E0 and E1),
  - image path resolution (image root found ONCE per split, then joined -
    no per-row glob calls over 40k rows).

Every function is idempotent: on a warm runtime it does nothing but check
markers; on a fresh runtime it restages only what the pending cell needs.
"""
import glob
import json
import os
import zipfile

from src import config, env

ANNOTATION_KINDS = ("vqa_annot", "quality_annot")
IMAGE_KINDS = ("images_train", "images_val")


# ── Drive zip discovery ──────────────────────────────────────────────────────

def list_drive_zips():
    zips = sorted(glob.glob(os.path.join(config.DRIVE_BASE, "**", "*.zip"),
                            recursive=True))
    return zips


def _zip_names(path, limit=80):
    try:
        with zipfile.ZipFile(path) as z:
            return z.namelist()[:limit]
    except Exception:
        return []


def looks_like_zip(kind, path):
    base = os.path.basename(path).lower()
    names = [n.lower() for n in _zip_names(path)]
    sample = " ".join(names[:80])
    if kind == "images_train":
        return (("train" in base and "annotation" not in base and "annot" not in base)
                or any("train" in n and n.endswith((".jpg", ".jpeg", ".png")) for n in names))
    if kind == "images_val":
        return (("val" in base and "annotation" not in base and "annot" not in base)
                or any("val" in n and n.endswith((".jpg", ".jpeg", ".png")) for n in names))
    if kind == "vqa_annot":
        return (("annotations" in base and os.path.basename(path)[:1].isupper())
                or ("answerable" in sample and "answers" in sample))
    if kind == "quality_annot":
        return (("quality" in base or base == "annotations.zip")
                and ("quality" in sample or "quality_flaws" in sample
                     or "unrecognizable" in sample or "recognizable" in sample))
    return False


def resolve_zip(kind, configured, all_zips=None):
    """Return the Drive zip path for `kind`, trying auto-discovery on miss."""
    if os.path.exists(configured):
        return configured
    if all_zips is None:
        all_zips = list_drive_zips()
    matches = [p for p in all_zips if looks_like_zip(kind, p)]
    if matches:
        print(f"  [AUTO] {kind}: using {matches[0]}")
        if len(matches) > 1:
            print(f"         other candidates: {matches[1:5]}")
        return matches[0]
    print(f"  [MISSING] {kind}: configured path not found: {configured}")
    return None


# ── Staging ──────────────────────────────────────────────────────────────────

def is_staged(kind: str) -> bool:
    return os.path.exists(os.path.join(config.LOCAL_BASE, kind, ".unzipped_ok"))


def stage_kinds(kinds, all_zips=None):
    """Stage the given zip kinds to local disk. Raises with a clear list if
    any required zip cannot be found on Drive."""
    todo = [k for k in kinds if not is_staged(k)]
    staged = {k: os.path.join(config.LOCAL_BASE, k) for k in kinds if is_staged(k)}
    for k in staged:
        print(f"[staging] already staged -> {staged[k]}")
    if not todo:
        return staged
    if all_zips is None:
        all_zips = list_drive_zips()
        print(f"[staging] Found {len(all_zips)} zip file(s) under {config.DRIVE_BASE}")
    missing = []
    for kind in todo:
        zp = resolve_zip(kind, config.RAW_ZIPS[kind], all_zips)
        if zp is None:
            missing.append(kind)
            continue
        dest = os.path.join(config.LOCAL_BASE, kind)
        staged[kind] = env.stage_zip_to_local(zp, dest)
    if missing:
        print("\n[staging] Required dataset zip(s) are missing on Drive:")
        for k in missing:
            print(f"  - {k}: expected {config.RAW_ZIPS[k]}")
        print("[staging] Put the zips on Drive or fix VQA_DRIVE_BASE/config.RAW_ZIPS, then rerun.")
        raise FileNotFoundError(f"missing required dataset zip(s): {missing}")
    return staged


def ensure_annotations():
    return stage_kinds(ANNOTATION_KINDS)


def ensure_images():
    return stage_kinds(IMAGE_KINDS)


# ── Annotation JSON location (single source for E0 and E1) ──────────────────

def annotation_json_candidates(dataset: str, split: str):
    L, D = config.LOCAL_BASE, config.DRIVE_BASE
    if dataset == "vqa":
        return [
            f"{L}/vqa_annot/Annotations/{split}.json",
            f"{L}/vqa_annot/vqa_annotations/{split}.json",
            f"{L}/vqa_annot/annotations/vqa_annotations/{split}.json",
            f"{L}/vqa_annot/{split}.json",
            f"{D}/data_raw/Annotations/{split}.json",
        ]
    if dataset == "quality":
        return [
            f"{L}/quality_annot/quality_annotations/{split}.json",
            f"{L}/quality_annot/annotations/{split}.json",
            f"{L}/quality_annot/annotations/quality_annotations/{split}.json",
            f"{L}/quality_annot/{split}.json",
            f"{D}/data_raw/annotations/{split}.json",
        ]
    raise ValueError(f"unknown dataset: {dataset}")


def find_annotation_json(dataset: str, split: str):
    for p in annotation_json_candidates(dataset, split):
        if os.path.exists(p):
            return p
    # Last resort: any matching json under the staged dir.
    root = os.path.join(config.LOCAL_BASE,
                        "vqa_annot" if dataset == "vqa" else "quality_annot")
    hits = glob.glob(os.path.join(root, "**", f"{split}.json"), recursive=True)
    return hits[0] if hits else None


def annotation_records(obj, split=None):
    """Normalize the various VizWiz annotation JSON layouts to a list of dicts."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("annotations", "data", "items", "records"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if split and isinstance(obj.get(split), dict):
            return annotation_records(obj[split])
        if "image" in obj and isinstance(obj["image"], list):
            n = len(obj["image"])
            return [{k: (v[i] if isinstance(v, list) and len(v) == n else v)
                     for k, v in obj.items()} for i in range(n)]
    return []


# ── Image path resolution ────────────────────────────────────────────────────

_IMAGE_ROOT_CACHE = {}


def image_root(split: str) -> str:
    """Find (once) the directory that actually contains the split's images."""
    if split in _IMAGE_ROOT_CACHE:
        return _IMAGE_ROOT_CACHE[split]
    base = os.path.join(config.LOCAL_BASE, f"images_{split}")
    exts = (".jpg", ".jpeg", ".png")

    def has_images(d):
        try:
            return any(f.lower().endswith(exts) for f in os.listdir(d))
        except Exception:
            return False

    for cand in (base, os.path.join(base, split), os.path.join(base, "data")):
        if os.path.isdir(cand) and has_images(cand):
            _IMAGE_ROOT_CACHE[split] = cand
            return cand
    for root, _dirs, files in os.walk(base):
        if any(f.lower().endswith(exts) for f in files):
            _IMAGE_ROOT_CACHE[split] = root
            return root
    _IMAGE_ROOT_CACHE[split] = base
    return base


def resolve_image_path(image_name: str, split: str) -> str:
    return os.path.join(image_root(split), image_name)


def add_image_paths(master):
    """Vectorized image_path column using one memoized root per split."""
    roots = {s: image_root(s) for s in master["split"].unique()}
    master = master.copy()
    master["image_path"] = [os.path.join(roots[s], n)
                            for s, n in zip(master["split"], master["image"])]
    return master


# ── What still needs staging, given completion state ────────────────────────

def stage_for_pending(run_e9: bool = None):
    """
    Stage only what the not-yet-finished experiments will need.

    - annotations: needed while E1 has not produced master.parquet
    - images:      needed while E2 or E6 (or an unlocked E9) are not done
    Saves 20-40 min of image unzipping on runtimes that will only run
    cached/CPU experiments.
    """
    from src import expstate
    if run_e9 is None:
        run_e9 = config.RUN_E9

    need = []
    master_path = os.path.join(config.DATA_PROCESSED, "master.parquet")
    if not os.path.exists(master_path) or config.FORCE_RERUN:
        need.extend(ANNOTATION_KINDS)

    e2_done = expstate.is_done("E2", config.RESULTS_E2, required=[
        os.path.join(config.ARTIFACTS, f"emb_{bb}.npy") for bb in config.BACKBONES])
    e6_done = expstate.is_done("E6", config.RESULTS_E6, required=[
        os.path.join(config.RESULTS_E6, "vqa_predictions.parquet")])
    e9_pending = run_e9 and not expstate.is_done("E9", config.RESULTS_E9)
    if not (e2_done and e6_done) or e9_pending:
        need.extend(IMAGE_KINDS)

    if not need:
        print("[staging] All image/annotation consumers already have cached "
              "outputs on Drive - nothing to stage.")
        return {}
    return stage_kinds(sorted(set(need)))
