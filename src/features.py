"""
E2 — Multi-backbone feature extraction.

Extracts frozen image embeddings for CLIP ViT-B/32, DINOv2 ViT-S/14, and
MobileNetV3-Large. Each backbone is extracted once and cached as float16 .npy
plus a parquet index. Every later experiment loads the cache — never re-extracts.

Idempotent: skips a backbone if its .npy already exists (FORCE_RERUN to override).
Checkpoints every SHARD_SIZE batches for resume safety.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

SHARD_SIZE = 500   # save a shard every N batches


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

    # Idempotency check
    if os.path.exists(out_npy) and not force:
        print(f"[features] cache hit → {out_npy}  (use FORCE_RERUN=True to re-extract)")
        return np.load(out_npy, allow_pickle=False)

    shard_dir  = out_npy + ".shards"
    done_file  = out_npy + ".done_rows.json"
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)

    model, preprocess, dim = load_backbone(backbone_name, device)

    # Load already-finished rows from shards (resume path)
    done_rows = set()
    if os.path.exists(done_file):
        done_rows = set(json.load(open(done_file)))
    remaining_idx = [i for i in range(n) if i not in done_rows]

    if not remaining_idx:
        print(f"[features] all shards complete for {backbone_name}; assembling…")
    else:
        dl = DataLoader(
            ImageDS([paths[i] for i in remaining_idx], preprocess),
            batch_size=bs,
            num_workers=min(num_workers, max(1, os.cpu_count() - 1)),
            pin_memory=(device != "cpu"),
            persistent_workers=(num_workers > 0),
        )
        buf_imgs, buf_orig_idx = [], []
        shard_count = 0

        def _flush():
            nonlocal shard_count
            if not buf_imgs:
                return
            batch = torch.stack(buf_imgs).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device != "cpu")):
                feat = model(batch)
                feat = torch.nn.functional.normalize(feat, dim=-1)
            arr = feat.float().cpu().numpy().astype(np.float16)
            shard_path = os.path.join(shard_dir, f"shard_{shard_count:06d}.npy")
            shard_idx_path = shard_path + ".idx.json"
            np.save(shard_path, arr)
            json.dump(buf_orig_idx, open(shard_idx_path, "w"))
            done_rows.update(buf_orig_idx)
            json.dump(list(done_rows), open(done_file, "w"))
            shard_count += 1
            buf_imgs.clear()
            buf_orig_idx.clear()

        for (imgs_batch, batch_local_idx) in tqdm(dl, desc=f"[{backbone_name}]"):
            for local_i, img in zip(batch_local_idx.numpy(), imgs_batch):
                orig_i = remaining_idx[local_i]
                buf_imgs.append(img)
                buf_orig_idx.append(int(orig_i))
                if len(buf_imgs) >= SHARD_SIZE * bs:
                    _flush()
        _flush()

    # Assemble all shards into final array
    out = np.zeros((n, dim), dtype=np.float16)
    for shard_f in sorted(os.listdir(shard_dir)):
        if not shard_f.endswith(".npy"):
            continue
        idx_f = os.path.join(shard_dir, shard_f + ".idx.json")
        shard_path = os.path.join(shard_dir, shard_f)
        arr  = np.load(shard_path)
        idxs = json.load(open(idx_f))
        out[idxs] = arr

    np.save(out_npy, out)
    print(f"[features] saved {out_npy}  shape={out.shape}  dtype={out.dtype}")
    return out


def build_feature_index(image_paths, image_names, splits, out_parquet: str):
    """
    Save a parquet mapping row_idx → image_path → image_name → split,
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
    print(f"[features] feature_index saved → {out_parquet}")
    return df
