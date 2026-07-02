"""E2 - Multi-backbone feature extraction (GPU; the only expensive cell).
Extracts frozen embeddings once, caches float16 .npy on Drive, resumes from
shards after a crash."""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, features, progress, resultlog, staging

EXP = "E2"


def required_artifacts():
    arts = [os.path.join(config.ARTIFACTS, f"emb_{bb}.npy") for bb in config.BACKBONES]
    arts.append(os.path.join(config.DATA_PROCESSED, "feature_index.parquet"))
    return arts


def main():
    progress.install_error_hook("E2 feature extraction")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()

    if expstate.is_done(EXP, config.RESULTS_E2, required=required_artifacts()):
        expstate.skip_banner(EXP, config.RESULTS_E2)
        return

    env.check_gpu(EXP)  # raises on CPU runtime - do not crawl through E2 on CPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = config.batch_size_for("features")
    nw = max(1, env.nproc() - 1)
    print(f"[E2] device={device}  tier={env.gpu_tier()}  batch_size={bs}  num_workers={nw}")

    pbar = progress.notebook_bar("E2 feature extraction", total=5 + len(config.BACKBONES))
    progress.step(pbar, f"Environment checked: device={device}")

    master_path = os.path.join(config.DATA_PROCESSED, "master.parquet")
    assert os.path.exists(master_path), "Run E1 first!"
    master = pd.read_parquet(master_path)
    progress.dataframe_summary("master", master)
    progress.step(pbar, "master.parquet loaded")

    # A fresh runtime needs the image zips staged again (Drive cache persists,
    # /content/local does not).
    staging.ensure_images()
    master = staging.add_image_paths(master)

    sample_paths = master["image_path"].sample(min(5, len(master)), random_state=42)
    missing = [p for p in sample_paths if not os.path.exists(p)]
    if missing:
        print(f"[E2] WARN: {len(missing)} sample paths not found. "
              f"First missing: {missing[0]}")
        print("  Check the unzip layout under", config.LOCAL_BASE)
    else:
        print("[E2] Path resolution looks correct.")
    progress.step(pbar, "Image paths resolved and sampled")

    all_paths = master["image_path"].tolist()
    all_names = master["image"].tolist()
    all_splits = master["split"].tolist()
    os.makedirs(config.ARTIFACTS, exist_ok=True)
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    emb_results = {}
    for bb in config.BACKBONES:
        out_npy = os.path.join(config.ARTIFACTS, f"emb_{bb}.npy")
        print(f"\n[E2] backbone={bb}  output={out_npy}")
        emb = features.extract(
            backbone_name=bb,
            paths=all_paths,
            out_npy=out_npy,
            device=device,
            bs=bs,
            num_workers=nw,
            force=config.FORCE_RERUN,
        )
        emb_results[bb] = {"shape": list(emb.shape), "dtype": str(emb.dtype)}
        print(f"  [E2] {bb}: shape={emb.shape}")
        progress.step(pbar, f"{bb} embeddings cached: shape={emb.shape}")

    fi_path = os.path.join(config.DATA_PROCESSED, "feature_index.parquet")
    if not os.path.exists(fi_path) or config.FORCE_RERUN:
        features.build_feature_index(all_paths, all_names, all_splits, fi_path)
    progress.step(pbar, "feature_index.parquet ready")

    # ── Cross-check E2 rows against the E0 audit ──
    audit_path = os.path.join(config.RESULTS_E0, "audit.json")
    if os.path.exists(audit_path):
        audit = json.load(open(audit_path))
        for split in ("train", "val"):
            e0_n = audit.get("image_counts_local", {}).get(f"images_{split}")
            e2_n = int((master["split"] == split).sum())
            if e0_n and abs(e0_n - e2_n) / max(e0_n, 1) > 0.10:
                print(f"[E2] FAIL: {split} count mismatch - E0 found {e0_n} image "
                      f"files but E2 has {e2_n} rows in master. "
                      f"Check the join key in data_assembly.py.")
            else:
                print(f"[E2] {split}: E0={e0_n}  E2={e2_n} - OK")
    progress.step(pbar, "E0/E2 count cross-check completed")

    resultlog.log_run(EXP, metrics=emb_results,
                      params={"backbones": config.BACKBONES, "seed": config.SEED,
                              "batch_size": bs, "gpu_tier": env.gpu_tier()},
                      results_dir=config.RESULTS_E2, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E2, artifacts=required_artifacts())
    progress.step(pbar, "E2 result logged")
    pbar.close()

    if config.AUTO_DISCONNECT:
        from google.colab import runtime
        runtime.unassign()
    print("[E2 DONE] Embeddings cached on Drive. E3/E4 run on any GPU.")
