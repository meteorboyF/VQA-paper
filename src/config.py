"""
Central configuration - all paths, seeds, GPU hints, and flags live here.
Edit this file (or override in the notebook cell header) before running any experiment.
"""
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
SEEDS = [0, 1, 2, 3, 4]          # 5-seed multi-run for every trainable head
QUAL_SEED = 7                      # fixed seed for qualitative figure sampling
N_BOOT = 2000                      # bootstrap resamples for CIs

# ── Paths ────────────────────────────────────────────────────────────────────
DRIVE_BASE   = os.environ.get("VQA_DRIVE_BASE", "/content/drive/MyDrive/VQA_ML/AVA_VizWiz")
LOCAL_BASE   = os.environ.get("VQA_LOCAL_BASE", "/content/local/AVA_VizWiz")
REPO_ROOT    = os.environ.get("VQA_REPO_ROOT", "/content/VQA-paper")          # where you git-cloned the repo

# Persistent experiment outputs.
# The notebook/code are refreshed from GitHub in /content/VQA-paper, but all
# expensive or reportable outputs live on Drive so a Colab restart can resume.
PERSIST_OUTPUTS_TO_DRIVE = os.environ.get("VQA_PERSIST_OUTPUTS_TO_DRIVE", "1") != "0"
DRIVE_WORK_DIR = os.environ.get("VQA_DRIVE_WORK_DIR", os.path.join(DRIVE_BASE, "reliable_vqa_outputs"))
OUTPUT_BASE = DRIVE_WORK_DIR if PERSIST_OUTPUTS_TO_DRIVE else REPO_ROOT

DATA_PROCESSED = os.path.join(OUTPUT_BASE, "data_processed")
ARTIFACTS      = os.path.join(OUTPUT_BASE, "artifacts")
RESULTS        = os.path.join(OUTPUT_BASE, "results")

# Sub-dirs for each experiment
RESULTS_E0 = os.path.join(RESULTS, "E0_audit")
RESULTS_E1 = os.path.join(RESULTS, "E1_assembly")
RESULTS_E2 = os.path.join(RESULTS, "E2_features")
RESULTS_E3 = os.path.join(RESULTS, "E3_triage")
RESULTS_E4 = os.path.join(RESULTS, "E4_defect")
RESULTS_E5 = os.path.join(RESULTS, "E5_actionable")
RESULTS_E6 = os.path.join(RESULTS, "E6_vqaconf")
RESULTS_E7 = os.path.join(RESULTS, "E7_calib")
RESULTS_E8 = os.path.join(RESULTS, "E8_ablation")
RESULTS_E9 = os.path.join(RESULTS, "E9_grounding")
FIGURES_DIR = os.path.join(RESULTS, "figures")

# ── Raw data zip names on Drive (adapt if your zip names differ) ─────────────
# These are only the CANONICAL locations: staging auto-discovers zips anywhere
# under DRIVE_BASE by content, and auto-downloads (then persists here) any
# that are missing. quality_annotations.zip is renamed from VizWiz's official
# annotations.zip to avoid a case-only clash with the VQA Annotations.zip.
RAW_ZIPS = {
    "images_train": f"{DRIVE_BASE}/data_raw/zips/train.zip",
    "images_val":   f"{DRIVE_BASE}/data_raw/zips/val.zip",
    "vqa_annot":    f"{DRIVE_BASE}/data_raw/zips/Annotations.zip",
    "quality_annot":f"{DRIVE_BASE}/data_raw/zips/quality_annotations.zip",
}

# ── Backbone selection ────────────────────────────────────────────────────────
# All three loaders are implemented in src/features.py.
# DINOv2 is opt-in by default to save E2 compute budget (~2x cost of CLIP).
# On an A100 the full three-backbone table is cheap: set
#   os.environ["VQA_BACKBONES"] = "clip,mobilenet,dinov2"  before importing src.
_bb_env = os.environ.get("VQA_BACKBONES", "").strip()
BACKBONES = ([b.strip() for b in _bb_env.split(",") if b.strip()]
             if _bb_env else ["clip", "mobilenet"])

BACKBONE_DIM = {
    "clip":     512,
    "dinov2":   384,
    "mobilenet": 960,
}

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE  = 256
LR          = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS  = 50
PATIENCE    = 7                    # early stopping patience on val AUROC
MLP_HIDDEN  = 256
MLP_DROPOUT = 0.3

# ── Data split ────────────────────────────────────────────────────────────────
CAL_FRAC = 0.30                    # fraction of val carved into the calibration split

# ── Calibration / gating ─────────────────────────────────────────────────────
N_TEMP_SCALE_ITERS = 50            # LBFGS max_iter for temperature scaling
ECE_BINS = 15

# ── Selective prediction ──────────────────────────────────────────────────────
COVERAGE_GRID = 50                 # #points for risk-coverage curve

# ── Phase flags (overridable via env vars so the GitHub notebook never needs
#    a local file edit: set os.environ[...] in a cell BEFORE importing src) ───
FORCE_RERUN    = os.environ.get("VQA_FORCE_RERUN", "0") == "1"   # ignore all caches/DONE markers
AUTO_DISCONNECT = os.environ.get("VQA_AUTO_DISCONNECT", "0") == "1"  # fire-and-forget runs (E2, E6, E9)
RUN_E9         = os.environ.get("VQA_RUN_E9", "0") == "1"        # Phase 2 gate - flip only after E0-E8 committed

# ── E9 (Phase 2) grounder ─────────────────────────────────────────────────────
GROUNDER = "locate_anything"       # "locate_anything" | "qwen25vl"
E9_SUBSAMPLE_N = 4000              # number of images to subsample for grounding harvest
E9_BATCH_SIZE  = 4                 # VLM inference batch size (memory-bound)

# ── VQA model (E6) ───────────────────────────────────────────────────────────
VQA_MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
VQA_BATCH_SIZE = 32                # fallback; E6 uses batch_size_for("vqa")

# ── GPU-tier-aware batch sizes ───────────────────────────────────────────────
# Any GPU works for any experiment; batches scale with the card so an A100
# finishes E2/E6 several times faster without OOM risk on a T4.
BATCH_SIZES = {
    #            a100  l4   t4   other-gpu  cpu
    "features": (512,  256, 128, 96,        32),   # E2 backbone extraction
    "vqa":      (128,  64,  32,  32,        8),    # E6 ViLT harvest
    "grounder": (16,   8,   4,   4,         1),    # E9 VLM inference
}


def batch_size_for(task: str) -> int:
    from src.env import gpu_tier
    tiers = {"a100": 0, "l4": 1, "t4": 2, "gpu": 3, "cpu": 4}
    a100, l4, t4, other, cpu = BATCH_SIZES[task]
    return (a100, l4, t4, other, cpu)[tiers.get(gpu_tier(), 3)]


# ── Recommended (not required) GPU per experiment ────────────────────────────
# Purely advisory: every GPU cell runs on any CUDA GPU (A100 > L4 > T4).
GPU_HINTS = {
    "E0": "CPU",
    "E1": "CPU",
    "E2": "GPU (A100 or L4 recommended)",
    "E3": "GPU (any)",
    "E4": "GPU (any)",
    "E5": "CPU",
    "E6": "GPU (A100 or L4 recommended)",
    "E7": "CPU",
    "E8": "CPU",
    "E9": "GPU (A100 or L4 recommended)",
}

# Experiments that must not silently crawl on CPU (raise unless VQA_ALLOW_CPU=1).
GPU_REQUIRED = {"E2", "E6", "E9"}
ALLOW_CPU = os.environ.get("VQA_ALLOW_CPU", "0") == "1"


def ensure_output_dirs() -> None:
    """Create persistent output directories used by all experiments."""
    for d in (
        DATA_PROCESSED, ARTIFACTS, RESULTS, FIGURES_DIR,
        RESULTS_E0, RESULTS_E1, RESULTS_E2, RESULTS_E3, RESULTS_E4,
        RESULTS_E5, RESULTS_E6, RESULTS_E7, RESULTS_E8, RESULTS_E9,
    ):
        os.makedirs(d, exist_ok=True)


def print_output_locations() -> None:
    """Print the active persistence layout for Colab sanity checks."""
    print("[config] Output persistence:")
    print(f"  PERSIST_OUTPUTS_TO_DRIVE={PERSIST_OUTPUTS_TO_DRIVE}")
    print(f"  OUTPUT_BASE={OUTPUT_BASE}")
    print(f"  DATA_PROCESSED={DATA_PROCESSED}")
    print(f"  ARTIFACTS={ARTIFACTS}")
    print(f"  RESULTS={RESULTS}")
