"""
Runtime bootstrap: mount Drive, stage zips to local SSD, seed everything,
carve the cal/rep split from VizWiz val.
"""
import os, sys, json, zipfile, shutil, random, subprocess
import numpy as np

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def nproc() -> int:
    try:
        return int(subprocess.check_output(["nproc"]).decode().strip())
    except Exception:
        return os.cpu_count() or 2


def mount_drive() -> None:
    """Mount Google Drive (Colab only; no-op outside Colab)."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("[env] Drive mounted at /content/drive")
    except ImportError:
        print("[env] Not in Colab — skipping Drive mount.")


def stage_zip_to_local(zip_path: str, dest_dir: str) -> str:
    """
    Copy a zip from Drive → local SSD and unzip there.
    Idempotent: skips if the '.unzipped_ok' marker already exists.
    """
    os.makedirs(dest_dir, exist_ok=True)
    marker = os.path.join(dest_dir, ".unzipped_ok")
    if os.path.exists(marker):
        print(f"[env] already staged → {dest_dir}")
        return dest_dir
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"[env] zip not found on Drive: {zip_path}")
    local_zip = os.path.join("/content", os.path.basename(zip_path))
    if not os.path.exists(local_zip):
        print(f"[env] copying {os.path.basename(zip_path)} Drive→local …")
        shutil.copy(zip_path, local_zip)
    print(f"[env] unzipping → {dest_dir} …")
    with zipfile.ZipFile(local_zip, "r") as z:
        z.extractall(dest_dir)
    open(marker, "w").close()
    print(f"[env] staged: {dest_dir}")
    return dest_dir


def count_images(directory: str, exts=(".jpg", ".jpeg", ".png", ".JPEG", ".JPG")) -> int:
    """Recursively count image files in a directory."""
    n = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(exts):
                n += 1
    return n


def check_gpu(exp_id: str) -> None:
    """Print current GPU and warn (not crash) if it doesn't match the recommended GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            print(f"[env] GPU: {gpu_name}")
        else:
            gpu_name = "CPU"
            print("[env] GPU: none (CPU runtime)")
    except ImportError:
        gpu_name = "CPU"
        print("[env] torch not available — CPU only")

    from src.config import GPU_HINTS
    expected = GPU_HINTS.get(exp_id, "any")
    if expected == "CPU":
        return
    if expected not in gpu_name and gpu_name != "CPU":
        # Warn but don't crash — the experiment can still run
        pass
    elif gpu_name == "CPU" and expected != "CPU":
        print(f"[env] WARNING: {exp_id} recommends a {expected} GPU but running on CPU."
              f" Computation will be very slow.")


def make_cal_rep_split(val_indices, cal_frac: float = 0.30, seed: int = SEED,
                       stratify_labels=None):
    """
    Carve val into cal (~30%) and rep (~70%) with stratification.
    Returns (cal_idx, rep_idx) as numpy arrays of integer positional indices
    into val_indices.

    IMPORTANT: This split is deterministic (SEED-fixed) and is the ONLY place
    where the cal/rep boundary is defined. Save the returned index arrays to
    results/EX/.../split_ids.json immediately after calling.
    """
    from sklearn.model_selection import train_test_split
    n = len(val_indices)
    pos_idx = np.arange(n)
    cal_pos, rep_pos = train_test_split(
        pos_idx,
        test_size=(1.0 - cal_frac),
        random_state=seed,
        stratify=stratify_labels,
    )
    return cal_pos, rep_pos


def save_split_ids(cal_idx, rep_idx, out_path: str) -> None:
    """Persist the cal/rep split index arrays so every downstream experiment
    uses the exact same partition."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    obj = {"cal": cal_idx.tolist(), "rep": rep_idx.tolist(), "seed": SEED}
    with open(out_path, "w") as f:
        json.dump(obj, f)
    print(f"[env] split ids saved → {out_path}")


def load_split_ids(path: str):
    with open(path) as f:
        obj = json.load(f)
    return np.array(obj["cal"]), np.array(obj["rep"])


def assert_no_rep_leakage(split_name: str) -> None:
    """
    Call this at the top of any function that selects a threshold or temperature.
    Raises ValueError if split_name is 'rep' or 'test' — the frozen-knob rule.
    """
    if split_name in ("rep", "test", "report"):
        raise ValueError(
            f"FROZEN-KNOB VIOLATION: threshold/temperature selection called with "
            f"split='{split_name}'. Thresholds must only be chosen on the 'cal' split."
        )
