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
import shutil
import zipfile

from src import config, env

ANNOTATION_KINDS = ("vqa_annot", "quality_annot")
IMAGE_KINDS = ("images_train", "images_val")

# Official VizWiz download URLs (verified live 2026-07). Used to auto-download
# anything missing from Drive; the downloaded zip is also copied back to
# Drive (config.RAW_ZIPS path) so future sessions skip the download.
DOWNLOAD_URLS = {
    "images_train": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
    "images_val": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
    "vqa_annot": "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip",
    "quality_annot": "https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip",
}
APPROX_SIZE_GB = {
    "images_train": 10.6,
    "images_val": 3.3,
    "vqa_annot": 0.002,
    "quality_annot": 0.001,
}


# ── Drive zip discovery ──────────────────────────────────────────────────────

def list_drive_zips():
    """All zips under DRIVE_BASE, plus a shallow scan one level up (in case
    the zips were uploaded next to, not inside, the dataset folder)."""
    zips = sorted(glob.glob(os.path.join(config.DRIVE_BASE, "**", "*.zip"),
                            recursive=True))
    parent = os.path.dirname(config.DRIVE_BASE.rstrip("/"))
    if parent and os.path.isdir(parent):
        for pat in ("*.zip", "*/*.zip", "*/*/*.zip"):
            for p in glob.glob(os.path.join(parent, pat)):
                if p not in zips:
                    zips.append(p)
    return zips


_ZIP_PROFILE_CACHE = {}


def _zip_profile(path):
    """Cheap classification profile: member names + head of the first JSON."""
    if path in _ZIP_PROFILE_CACHE:
        return _ZIP_PROFILE_CACHE[path]
    prof = {"names": [], "json_head": ""}
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()[:400]
            prof["names"] = [n.lower() for n in names]
            # Sniff train/val JSONs first: in the real QualityIssues zip,
            # test.json sorts first but its labels are hidden (image-only
            # records), which would defeat content classification.
            jsons = sorted(
                (n for n in names if n.lower().endswith(".json")),
                key=lambda n: ("train" not in n.lower() and "val" not in n.lower(), n))
            heads = []
            for n in jsons[:3]:
                try:
                    with z.open(n) as f:
                        heads.append(f.read(8_000_000).decode("utf-8", "ignore").lower())
                except Exception:
                    pass
            prof["json_head"] = " ".join(heads)
    except Exception:
        pass
    _ZIP_PROFILE_CACHE[path] = prof
    return prof


def looks_like_zip(kind, path):
    """Classify a zip by its CONTENT first, filename second.
    (The VizWiz QualityIssues zip contains only train/val/test.json - the word
    'quality' never appears in its file listing, so name-only matching fails.)"""
    base = os.path.basename(path).lower()
    prof = _zip_profile(path)
    names, head = prof["names"], prof["json_head"]
    img_names = [n for n in names if n.endswith((".jpg", ".jpeg", ".png"))]

    if kind in ("images_train", "images_val"):
        split = kind.split("_")[1]
        if "annot" in base:
            return False
        if img_names:
            return any(split in n for n in img_names) or split in base
        return split in base
    if kind == "vqa_annot":
        if head:
            return _head_is_vqa(head)
        return "annotations" in base and os.path.basename(path)[:1].isupper()
    if kind == "quality_annot":
        if head:
            return _head_is_quality(head)
        return "quality" in base
    return False


def _head_is_vqa(head: str) -> bool:
    # Match JSON KEYS (quote immediately followed by colon), never bare words:
    # VQA questions are free text and can contain 'quality'/'flaws' etc.
    return ('"answerable":' in head
            or ('"answers":' in head and '"question":' in head))


def _head_is_quality(head: str) -> bool:
    return '"flaws":' in head or '"unrecognizable":' in head


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


# ── Auto-download of missing datasets ────────────────────────────────────────

def _download_scratch() -> str:
    if os.path.isdir("/content"):
        return "/content"
    return os.path.dirname(config.LOCAL_BASE.rstrip("/\\")) or "."


def _is_valid_zip(path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return bool(z.namelist())
    except Exception:
        return False


def _download_url(url, dest_path, desc):
    """Stream a URL to dest_path with a progress bar. Resumes a partial
    .part file via HTTP Range; retries 3 times."""
    import urllib.request
    try:
        from tqdm.auto import tqdm
    except Exception:
        from tqdm import tqdm
    part = dest_path + ".part"
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    last_exc = None
    for attempt in range(1, 4):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and getattr(resp, "status", 200) != 206:
                    have = 0  # server ignored the resume request; restart
                total = resp.headers.get("Content-Length")
                total = (int(total) + have) if total else None
                with open(part, "ab" if have else "wb") as out, tqdm(
                        total=total, initial=have, unit="B", unit_scale=True,
                        unit_divisor=1024, desc=desc) as bar:
                    while True:
                        chunk = resp.read(1 << 22)   # 4 MB
                        if not chunk:
                            break
                        out.write(chunk)
                        bar.update(len(chunk))
            os.replace(part, dest_path)
            return dest_path
        except Exception as exc:
            last_exc = exc
            print(f"[staging] download attempt {attempt}/3 failed: {exc} - "
                  f"resuming from what was already fetched")
    raise RuntimeError(f"could not download {url}: {last_exc}")


def download_kind(kind: str) -> str:
    """Download the official VizWiz zip for `kind` to local disk, validate it,
    and persist a copy to Drive (config.RAW_ZIPS). Returns the local path."""
    url = DOWNLOAD_URLS[kind]
    scratch = _download_scratch()
    os.makedirs(scratch, exist_ok=True)
    # kind-prefixed name avoids the Annotations.zip / annotations.zip clash
    local_zip = os.path.join(scratch, f"{kind}__{os.path.basename(url)}")

    if not (os.path.exists(local_zip) and _is_valid_zip(local_zip)):
        need_gb = APPROX_SIZE_GB.get(kind, 1.0)
        try:
            free_gb = shutil.disk_usage(scratch).free / 1e9
            if free_gb < need_gb * 2.2:
                print(f"[staging] WARN: only {free_gb:.1f} GB free on local disk; "
                      f"{kind} needs ~{need_gb:.1f} GB zipped + unzipped.")
        except Exception:
            pass
        print(f"[staging] downloading {kind} (~{APPROX_SIZE_GB.get(kind, '?')} GB) "
              f"from {url}")
        _download_url(url, local_zip, desc=f"download {kind}")

    if not _is_valid_zip(local_zip):
        try:
            os.remove(local_zip)
        except OSError:
            pass
        raise RuntimeError(f"downloaded file for {kind} is not a valid zip; "
                           f"deleted - rerun to retry")

    # Persist to Drive so future sessions find it without downloading again.
    drive_dest = config.RAW_ZIPS[kind]
    if not os.path.exists(drive_dest):
        try:
            os.makedirs(os.path.dirname(drive_dest), exist_ok=True)
            print(f"[staging] copying {kind} zip to Drive for future sessions: "
                  f"{drive_dest}")
            shutil.copy(local_zip, drive_dest + ".part")
            os.replace(drive_dest + ".part", drive_dest)
            print(f"[staging] persisted -> {drive_dest}")
        except Exception as exc:
            print(f"[staging] WARN: could not persist {kind} zip to Drive "
                  f"({exc}). Continuing with the local copy; the download "
                  f"will repeat next session unless you free Drive space.")
    return local_zip


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
        print(f"[staging] Found {len(all_zips)} zip file(s) under {config.DRIVE_BASE} (+1 level up):")
        for zp in all_zips[:40]:
            print(f"  [zip] {zp}")
        if len(all_zips) > 40:
            print(f"  ... {len(all_zips) - 40} more zip files not shown")
    missing = []
    for kind in todo:
        zp = resolve_zip(kind, config.RAW_ZIPS[kind], all_zips)
        if zp is None:
            missing.append(kind)
            continue
        dest = os.path.join(config.LOCAL_BASE, kind)
        staged[kind] = env.stage_zip_to_local(zp, dest)

    # Annotation kinds can fall back to unzipped JSONs already on Drive.
    for kind in list(missing):
        if kind not in ANNOTATION_KINDS:
            continue
        ds = "vqa" if kind == "vqa_annot" else "quality"
        found = {s: find_annotation_json(ds, s) for s in ("train", "val")}
        if all(found.values()):
            print(f"[staging] {kind}: no zip found, but unzipped JSONs exist on Drive - using them:")
            for s, p in found.items():
                print(f"    {s}: {p}")
            missing.remove(kind)

    # Anything still missing: download from the official VizWiz mirror,
    # stage it, and persist the zip to Drive (disable with VQA_AUTO_DOWNLOAD=0).
    if missing and os.environ.get("VQA_AUTO_DOWNLOAD", "1") != "0":
        for kind in list(missing):
            if kind not in DOWNLOAD_URLS:
                continue
            try:
                local_zip = download_kind(kind)
                dest = os.path.join(config.LOCAL_BASE, kind)
                staged[kind] = env.stage_zip_to_local(local_zip, dest)
                missing.remove(kind)
            except Exception as exc:
                print(f"[staging] auto-download failed for {kind}: {exc}")

    if missing:
        hints = {
            "images_train": "VizWiz train images (train.zip) from https://vizwiz.org/tasks-and-datasets/vqa/",
            "images_val": "VizWiz val images (val.zip) from https://vizwiz.org/tasks-and-datasets/vqa/",
            "vqa_annot": "VizWiz-VQA Annotations.zip (train.json/val.json with 'answerable' + 'answers')",
            "quality_annot": "VizWiz-QualityIssues annotations zip (train.json/val.json with 'flaws') "
                             "from https://vizwiz.org/tasks-and-datasets/image-quality-issues/",
        }
        print("\n[staging] Required dataset(s) not found on Drive "
              "(and auto-download did not succeed):")
        for k in missing:
            print(f"  - {k}: expected {config.RAW_ZIPS[k]}")
            print(f"      -> need: {hints.get(k, '')}")
        print("[staging] The [zip] list above shows everything that WAS found.")
        print("[staging] Upload the missing file(s) to Drive (any folder under "
              f"{config.DRIVE_BASE} works - discovery matches by content), then rerun.")
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


def _json_head(path, nbytes=8_000_000):
    try:
        with open(path, "rb") as f:
            return f.read(nbytes).decode("utf-8", "ignore").lower()
    except Exception:
        return ""


def _json_matches_dataset(path, dataset):
    head = _json_head(path)
    return _head_is_vqa(head) if dataset == "vqa" else _head_is_quality(head)


def find_annotation_json(dataset: str, split: str):
    # Prefer content-verified hits everywhere: the VQA and Quality zips both
    # contain files literally named train.json/val.json, so a path match
    # alone can pick the wrong dataset.
    existing = [p for p in annotation_json_candidates(dataset, split)
                if os.path.exists(p)]
    for p in existing:
        if _json_matches_dataset(p, dataset):
            return p
    # Any matching json under the staged dir.
    root = os.path.join(config.LOCAL_BASE,
                        "vqa_annot" if dataset == "vqa" else "quality_annot")
    for p in glob.glob(os.path.join(root, "**", f"{split}.json"), recursive=True):
        if _json_matches_dataset(p, dataset):
            return p
    # Shallow Drive-side search with content sniffing, so an unzipped upload
    # in a nonstandard folder still works.
    for pat in (f"{split}.json", f"*/{split}.json", f"*/*/{split}.json",
                f"*/*/*/{split}.json"):
        for p in glob.glob(os.path.join(config.DRIVE_BASE, pat)):
            if _json_matches_dataset(p, dataset):
                return p
    # Last resort: an existing candidate whose content didn't verify (unknown
    # schema variant) - better to let E0's schema audit show it than to fail.
    if existing:
        print(f"[staging] WARN: {dataset}/{split}.json found at {existing[0]} "
              f"but its content did not match the expected schema - using it anyway.")
        return existing[0]
    return None


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
