"""
Question-text feature extraction for question-conditioned triage (E10).

Uses the CLIP ViT-B/32 text tower (same open_clip checkpoint as the image
side) so that (a) no new dependency is introduced and (b) an image--text
cosine-similarity baseline comes for free. Embeddings are cached once as
float16 .npy in master-row order, mirroring src/features.py.
"""
import os

import numpy as np

from src import config

TEXT_DIM = 512


def question_emb_path() -> str:
    return os.path.join(config.ARTIFACTS, "emb_question_clip.npy")


def imgtext_cos_path() -> str:
    return os.path.join(config.ARTIFACTS, "clip_imgtext_cos.npy")


def extract_question_embeddings(questions, device: str = "cuda",
                                batch_size: int = 256,
                                force: bool = False) -> np.ndarray:
    """Encode all questions with the CLIP text tower; cache to Drive."""
    out_npy = question_emb_path()
    if os.path.exists(out_npy) and not force:
        arr = np.load(out_npy)
        if arr.shape == (len(questions), TEXT_DIM):
            print(f"[text_features] cache hit -> {out_npy}")
            return arr
        print("[text_features] cached question embeddings have wrong shape; re-extracting")

    import torch
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    out = np.zeros((len(questions), TEXT_DIM), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            chunk = [str(q) for q in questions[start:start + batch_size]]
            toks = tokenizer(chunk).to(device)
            feat = model.encode_text(toks)
            out[start:start + len(chunk)] = feat.float().cpu().numpy().astype(np.float16)
            if (start // batch_size) % 20 == 0:
                print(f"[text_features] {start + len(chunk)}/{len(questions)}")

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    tmp = out_npy + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, out_npy)
    print(f"[text_features] saved {out_npy}  shape={out.shape}")
    return out


def image_text_cosine(img_emb: np.ndarray, txt_emb: np.ndarray,
                      force: bool = False) -> np.ndarray:
    """Rowwise cosine similarity between CLIP image and text embeddings."""
    out_npy = imgtext_cos_path()
    if os.path.exists(out_npy) and not force:
        arr = np.load(out_npy)
        if arr.shape == (len(img_emb),):
            return arr
    a = img_emb.astype(np.float32)
    b = txt_emb.astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    b /= np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    cos = (a * b).sum(axis=1)
    np.save(out_npy, cos)
    return cos
