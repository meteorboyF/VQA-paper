"""
Central configuration — all paths, seeds, GPU hints, and flags live here.
Edit this file (or override in the notebook cell header) before running any experiment.
"""
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
SEEDS = [0, 1, 2, 3, 4]          # 5-seed multi-run for every trainable head
QUAL_SEED = 7                      # fixed seed for qualitative figure sampling
N_BOOT = 2000                      # bootstrap resamples for CIs

# ── Paths ────────────────────────────────────────────────────────────────────
DRIVE_BASE   = "/content/drive/MyDrive/VQA_ML/AVA_VizWiz"
LOCAL_BASE   = "/content/local/AVA_VizWiz"
REPO_ROOT    = "/content/VQA-paper"          # where you git-cloned the repo

DATA_PROCESSED = os.path.join(REPO_ROOT, "data_processed")
ARTIFACTS      = os.path.join(REPO_ROOT, "artifacts")
RESULTS        = os.path.join(REPO_ROOT, "results")

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
RAW_ZIPS = {
    "images_train": f"{DRIVE_BASE}/data_raw/zips/train.zip",
    "images_val":   f"{DRIVE_BASE}/data_raw/zips/val.zip",
    "vqa_annot":    f"{DRIVE_BASE}/data_raw/zips/Annotations.zip",
    "quality_annot":f"{DRIVE_BASE}/data_raw/zips/annotations.zip",
}

# ── Backbone selection ────────────────────────────────────────────────────────
# All three loaders are implemented in src/features.py.
# DINOv2 is commented out by default to save E2 compute budget (~2× cost of CLIP).
# Flip to True to include it for the full reviewer-grade three-backbone table.
BACKBONES = ["clip", "mobilenet"]   # "dinov2" opt-in
# BACKBONES = ["clip", "mobilenet", "dinov2"]   # full table

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

# ── Phase flags ───────────────────────────────────────────────────────────────
FORCE_RERUN    = False             # set True to re-compute even if artifact exists
AUTO_DISCONNECT = False            # set True for fire-and-forget runs (E2, E6, E9)
RUN_E9         = False             # Phase 2 gate — flip only after E0–E8 committed

# ── E9 (Phase 2) grounder ─────────────────────────────────────────────────────
GROUNDER = "locate_anything"       # "locate_anything" | "qwen25vl"
E9_SUBSAMPLE_N = 4000              # number of images to subsample for grounding harvest
E9_BATCH_SIZE  = 4                 # VLM inference batch size (memory-bound)

# ── VQA model (E6) ───────────────────────────────────────────────────────────
VQA_MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
VQA_BATCH_SIZE = 32

# ── Expected GPU per experiment (for the assertion check) ────────────────────
GPU_HINTS = {
    "E0": "CPU",
    "E1": "CPU",
    "E2": "L4",
    "E3": "T4",
    "E4": "T4",
    "E5": "CPU",
    "E6": "L4",
    "E7": "CPU",
    "E8": "CPU",
    "E9": "L4",
}
