"""
E2 - Multi-backbone feature extraction.

Extracts frozen image embeddings for CLIP ViT-B/32, DINOv2 ViT-S/14, and
MobileNetV3-Large. Each backbone is extracted once and cached as float16 .npy
plus a parquet index. Every later experiment loads the cache - never re-extracts.

Idempotent: skips a backbone if its .npy already exists (FORCE_RERUN to override).
Checkpoints every SHARD_SIZE batches for resume safety.
"""
import os
import json
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

SHARD_SIZE = 500   # save a shard every N batches

KNOWN_DIMS = {
    "clip": 512,
    "dinov2": 384,
    "mobilenet": 960,
}


def _atomic_json_dump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_npy_save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _quarantine(path: str, reason: str) -> None:
    bad = path + ".corrupt"
    print(f"[features] corrupt checkpoint ignored: {path} ({reason})")
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
        print(f"[features] warning: could not quarantine {path}: {exc}")


def _validate_final_npy(path: str, n: int, dim: int):
    try:
        arr = np.load(path, allow_pickle=False)
        if arr.shape != (n, dim):
            raise ValueError(f"shape {arr.shape} != {(n, dim)}")
        if arr.dtype != np.float16:
            raise ValueError(f"dtype {arr.dtype} != float16")
        if not np.isfinite(arr).all():
            raise ValueError("contains non-finite values")
        return arr
    except Exception as exc:
        _quarantine(path, str(exc))
        return None


def _valid_feature_shards(shard_dir: str, n: int, dim: int):
    """Return sorted valid shard records and quarantine corrupt/incomplete pairs."""
    valid = []
    seen = set()
    for shard_f in sorted(os.listdir(shard_dir)):
        if not shard_f.startswith("shard_") or not shard_f.endswith(".npy"):
            continue
        shard_path = os.path.join(shard_dir, shard_f)
        idx_path = shard_path + ".idx.json"
        try:
            if not os.path.exists(idx_path):
                raise ValueError("missing idx json")
            arr = np.load(shard_path, allow_pickle=False)
            idxs = json.load(open(idx_path))
            if arr.ndim != 2 or arr.shape[1] != dim:
                raise ValueError(f"bad array shape {arr.shape}")
            if len(idxs) != arr.shape[0]:
                raise ValueError(f"idx length {len(idxs)} != rows {arr.shape[0]}")
            if any((int(i) < 0 or int(i) >= n) for i in idxs):
                raise ValueError("row id out of range")
            if not np.isfinite(arr).all():
                raise ValueError("non-finite values")
            dup = seen.intersection(int(i) for i in idxs)
            if dup:
                raise ValueError(f"duplicate row ids, first={next(iter(dup))}")
            seen.update(int(i) for i in idxs)
            valid.append((shard_path, [int(i) for i in idxs]))
        except Exception as exc:
            _quarantine(shard_path, str(exc))
            _quarantine(idx_path, "paired with corrupt shard")
    return valid, seen


# ── Dataset ──────────────────────────────────────────────────────────────────

class ImageDS(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = list(paths)
        self.pre   = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.pre(img), i
        except Exception:
            # Return a zeros tensor of the right shape on corrupt images
            dummy = Image.new("RGB", (224, 224))
            return self.pre(dummy), i


# ── Backbone loaders ─────────────────────────────────────────────────────────

def _load_clip(device: str):
    """CLIP ViT-B/32 via open_clip."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    dim = 512
    def encode(imgs):
        return model.encode_image(imgs)
    return encode, preprocess, dim


def _load_dinov2(device: str):
    """DINOv2 ViT-S/14 via torch.hub."""
    import torchvision.transforms as T
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14",
        pretrained=True, verbose=False
    ).to(device).eval()
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dim = 384
    def encode(imgs):
        return model(imgs)
    return encode, preprocess, dim


def _load_mobilenet(device: str):
    """MobileNetV3-Large, penultimate layer (before classifier)."""
    import torchvision.models as tvm
    import torchvision.transforms as T

    weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    full_model = tvm.mobilenet_v3_large(weights=weights).to(device).eval()

    # Drop the final classifier; keep everything through adaptive_avg_pool
    backbone = torch.nn.Sequential(
        full_model.features,
        full_model.avgpool,
        torch.nn.Flatten(1),
    )

    preprocess = weights.transforms()
    dim = 960
    def encode(imgs):
        return backbone(imgs)
    return encode, preprocess, dim


def load_backbone(name: str, device: str):
    loaders = {
        "clip":     _load_clip,
        "dinov2":   _load_dinov2,
        "mobilenet":_load_mobilenet,
    }
    if name not in loaders:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(loaders)}")
    return loaders[name](device)


# ── Main extraction ───────────────────────────────────────────────────────────

@torch.inference_mode()
def extract(
    backbone_name: str,
    paths,
    out_npy: str,
    device: str = "cuda",
    bs: int = 192,
    num_workers: int = 7,
    force: bool = False,
) -> np.ndarray:
    """
    Extract embeddings for all images in `paths` using `backbone_name`.
    Saves float16 .npy to `out_npy`. Resumes from shards if interrupted.
    """
    paths = list(paths)
    n = len(paths)
    from src.env import autocast_dtype, setup_cuda_perf
    setup_cuda_perf()
    ac_dtype = autocast_dtype()

    shard_dir  = out_npy + ".shards"
    done_file  = out_npy + ".done_rows.json"
    if force:
        for p in (out_npy, done_file):
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(shard_dir):
            shutil.rmtree(shard_dir)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)

    dim = KNOWN_DIMS.get(backbone_name)

    # Idempotency check: trust only a readable, correctly shaped final cache.
    if os.path.exists(out_npy) and not force and dim is not None:
        cached = _validate_final_npy(out_npy, n, dim)
        if cached is not None:
            print(f"[features] cache hit -> {out_npy}  (use FORCE_RERUN=True to re-extract)")
            return cached
        print(f"[features] final cache was invalid; resuming/rebuilding from shards")

    model = preprocess = None
    if dim is None:
        model, preprocess, dim = load_backbone(backbone_name, device)
    if os.path.exists(out_npy) and not force:
        cached = _validate_final_npy(out_npy, n, dim)
        if cached is not None:
            print(f"[features] cache hit -> {out_npy}  (use FORCE_RERUN=True to re-extract)")
            return cached
        print(f"[features] final cache was invalid; resuming/rebuilding from shards")

    # Source of truth for resume is readable shards, not the done JSON.
    valid_shards, done_rows = _valid_feature_shards(shard_dir, n, dim)
    _atomic_json_dump(sorted(done_rows), done_file)
    remaining_idx = [i for i in range(n) if i not in done_rows]
    print(f"[features] resume state for {backbone_name}: "
          f"{len(done_rows)}/{n} rows complete, {len(remaining_idx)} remaining")

    if not remaining_idx:
        print(f"[features] all shards complete for {backbone_name}; assembling...")
    else:
        if model is None or preprocess is None:
            model, preprocess, dim = load_backbone(backbone_name, device)
        dl = DataLoader(
            ImageDS([paths[i] for i in remaining_idx], preprocess),
            batch_size=bs,
            num_workers=min(num_workers, max(1, os.cpu_count() - 1)),
            pin_memory=(device != "cpu"),
            persistent_workers=(num_workers > 0),
        )
        existing = [
            int(f.split("_")[1].split(".")[0])
            for f in os.listdir(shard_dir)
            if f.startswith("shard_") and f.endswith(".npy")
        ]
        shard_count = (max(existing) + 1) if existing else 0

        def _write_shard(arr, idxs):
            nonlocal shard_count
            shard_path = os.path.join(shard_dir, f"shard_{shard_count:06d}.npy")
            shard_idx_path = shard_path + ".idx.json"
            _atomic_npy_save(shard_path, arr)
            _atomic_json_dump([int(i) for i in idxs], shard_idx_path)
            done_rows.update(int(i) for i in idxs)
            _atomic_json_dump(sorted(done_rows), done_file)
            shard_count += 1

        for (imgs_batch, batch_local_idx) in tqdm(dl, desc=f"[{backbone_name}]"):
            batch = imgs_batch.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=ac_dtype, enabled=(device != "cpu")):
                feat = model(batch)
                feat = torch.nn.functional.normalize(feat, dim=-1)
            arr = feat.float().cpu().numpy().astype(np.float16)
            idxs = [remaining_idx[int(local_i)] for local_i in batch_local_idx.numpy()]
            _write_shard(arr, idxs)

    # Assemble all shards into final array
    out = np.zeros((n, dim), dtype=np.float16)
    valid_shards, done_rows = _valid_feature_shards(shard_dir, n, dim)
    if len(done_rows) != n:
        missing = sorted(set(range(n)) - done_rows)[:10]
        raise RuntimeError(
            f"[features] cannot assemble {backbone_name}: {n-len(done_rows)} rows "
            f"still missing after extraction. First missing rows: {missing}"
        )
    for shard_path, idxs in valid_shards:
        arr = np.load(shard_path, allow_pickle=False)
        out[idxs] = arr

    _atomic_npy_save(out_npy, out)
    print(f"[features] saved {out_npy}  shape={out.shape}  dtype={out.dtype}")
    return out


def build_feature_index(image_paths, image_names, splits, out_parquet: str):
    """
    Save a parquet mapping row_idx -> image_path -> image_name -> split,
    so embeddings rows can be aligned to master.parquet rows later.
    """
    import pandas as pd
    df = pd.DataFrame({
        "row_idx":    np.arange(len(image_paths)),
        "path":       image_paths,
        "image":      image_names,
        "split":      splits,
    })
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[features] feature_index saved -> {out_parquet}")
    return df
