# Reliable Assistive VQA — Master Pipeline Specification

**Repo:** `https://github.com/meteorboyF/VQA-paper.git`
**Target venue:** IEEE Access
**Runtime:** Google Colab Pro+ (GPU choice specified per experiment)
**This document is the single source of truth.** Claude Code consumes it to scaffold the notebook + utilities; you run the notebook cell-by-cell in Colab.

---

## 0. The reframed paper

### 0.1 Title

**Primary (recommended):**
*Knowing When, Why, and Where to Refuse: A Defect-Aware Reliability Layer for Assistive Visual Question Answering*

> The "Where" is earned only if the LocateAnything pillar (Phase 2, E9/C5) lands. If you ship the core paper without E9, drop "Where" → *Knowing When and Why to Refuse: A Defect-Aware Reliability Layer for Assistive Visual Question Answering*. Keep the title honest to what the experiments actually deliver.

Alternative framings, if you prefer a more conventional IEEE Access tone:
- *A Defect-Aware Reliability Layer for Trustworthy Assistive Visual Question Answering*
- *Triage, Diagnose, and Guide: Selective Prediction and Actionable Recovery for Assistive VQA*

### 0.2 Research Questions
The paper is organized around three RQs. Every experiment maps to exactly one.

- **RQ1 — Triage & diagnosis (When/Why).** Can a lightweight head, operating on frozen image embeddings, jointly predict (a) whether a blind user's photo is *answerable* and (b) *which* quality defect (blur, dark, bright, obstruction, framing, rotation, unrecognizable) is responsible — and does a *unified* multi-task head outperform a *cascade* by reducing error propagation? *(Answered by E3, E4, E8/C3; backbones by E2/C4.)*

- **RQ2 — Defect-aware selective prediction (Trust).** Does the optimal abstention policy for a frozen downstream VQA model depend on the *diagnosed defect*? Specifically, does a defect-conditioned confidence gate achieve a lower risk–coverage curve (lower AURC at equal coverage) than a single global confidence threshold, and does this hold when using *predicted* (not ground-truth) defects? *(Answered by E6, E7/C1; calibration via ECE + temperature scaling.)*

- **RQ3 — Actionable & spatial recovery (Where/How).** Beyond classifying defects, can the system produce *correct corrective guidance*, measured by a new Actionable Recovery Rate (ARR) and a False-Refilm Rate (FRR)? And — Phase 2 — does grounding the *queried entity* with a frontier grounding VLM (LocateAnything) yield (a) a useful *groundability* signal for triage and (b) *spatially-localized* guidance ("the item is cut off — pan left") that improves over whole-image advice? *(Answered by E5/C2 for the core; E9/C5 for the grounding extension.)*

### 0.3a One-paragraph thesis
Off-the-shelf VQA models fail silently for blind users: they return confident answers on photos that physically cannot answer the question (too dark, finger over lens, subject cut off). We do **not** build a better VQA model. We build a *reliability layer* that wraps any frozen VQA model and decides, before trusting an answer, (1) **triage** — is this answerable? (2) **diagnose** — which quality defect is responsible? (3) **guide** — what corrective action should the user take, and *where*? and (4) **calibrate & abstain** — is the VQA model's confidence trustworthy *given the diagnosed defect*, and should we answer or ask for a retake? All labels are real VizWiz annotations; no heuristic labels anywhere.

### 0.3 Why this is publishable (and the old draft was not)
- **Old draft's fatal flaw:** intent labels were keyword-matched from the question, then predicted from the image alone — an ill-posed task with fabricated ground truth. Reviewers open the repo and reject.
- **This design:** every label (`answerable`, the six quality flaws, `unrecognizable`) is human-annotated in VizWiz-VQA + VizWiz-QualityIssues. The task is well-posed: image quality genuinely determines answerability.
- **Beyond existing baselines:** the official VizWiz-QualityIssues code already does joint *answerability + recognizability* prediction. We do **not** claim that as novel. Our four genuine contributions sit on top of it.

### 0.4 Contributions (these are the novelty — protect them)
1. **C1 — Defect-aware selective prediction.** The central new result: the optimal abstention threshold and the degree of miscalibration of a downstream VQA model **depend on which quality defect is present**. We show a *defect-conditioned* gating policy dominates a single global confidence threshold on risk–coverage (lower AURC at equal coverage). To our knowledge this link between IQA defect type and VQA selective prediction has not been made.
2. **C2 — Actionable Recovery protocol + metric.** We formalize evaluation of *corrective guidance*: a new metric (Actionable Recovery Rate, ARR) measuring, among quality-unanswerable images, the fraction where the recommended action matches a ground-truth defect — plus a False-Refilm Rate (FRR) penalizing telling users to refilm a perfectly good photo. Existing work classifies defects; nobody scores whether the resulting *advice* is right.
3. **C3 — Unified reliability head vs. cascade.** We quantify error propagation between a joint multi-task head (triage+diagnose) and a cascade, an ablation absent from prior IQA work.
4. **C4 — Modern, reproducible benchmark.** Prior baselines use ResNet152/Detectron features. We benchmark CLIP ViT-B/32, DINOv2-ViT-S/14, and MobileNetV3 features under identical heads with full calibration metrics (ECE, AURC, reliability diagrams), all released.
5. **C5 — Groundability-aware reliability & spatial guidance (Phase 2, optional).** We test whether grounding the *queried entity* with a frozen frontier grounding VLM (NVIDIA LocateAnything-3B, arXiv 2026) supplies (a) a *groundability* reliability signal — does the referred object localize at all, with what confidence, does its box hit the image border, does the model emit its learned "no valid target" abstention — that improves triage beyond appearance features alone; and (b) *spatially-localized* corrective guidance (e.g., "the item is cut off — pan left") that upgrades whole-image advice. This is the only role for a grounding model that serves the reliability thesis rather than stacking an unrelated SOTA component. **This pillar is strictly downstream: C1–C4 form a complete, submittable paper without it.** Their learned Negative-Block abstention is also a clean related-work parallel to our defect-aware selective prediction.

### 0.5 Datasets used (all real labels, all already partially on your Drive)
| Dataset | What we use from it | Label type |
|---|---|---|
| VizWiz-VQA (2018) | `answerable` flag, 10 answers per Q (for VQA correctness), question text | human |
| VizWiz-QualityIssues (2020) | 6 quality flaws + `unrecognizable` flag, train/val/test json | human |
| VizWiz-Captions (2020) *(optional, C-stretch)* | captions as an "information-sufficiency" oracle | human |

> You already downloaded VQA Annotations + image_quality annotations. The QualityIssues annotations live as `quality_annotations/{train,val,test}.json`. Captions are optional; only pull them if we add the stretch analysis.

---

## 1. Repository structure (Claude Code creates this)

```
VQA-paper/
├── PIPELINE.md                  # this file (committed for provenance)
├── README.md                    # how to run, in order
├── reproduce.sh                 # one command: rebuild every table+figure from cached metrics/logits
├── requirements.txt             # pinned versions
├── notebooks/
│   └── reliable_vqa_master.ipynb  # THE notebook you run, E0..E8 as cells
├── src/
│   ├── config.py                # all paths, seeds, GPU hints, constants
│   ├── env.py                   # mount, copy-to-local, unzip, nproc, seeding
│   ├── data_assembly.py         # join VQA + QualityIssues -> master parquet
│   ├── features.py              # multi-backbone extraction (CLIP/DINOv2/MobileNet)
│   ├── heads.py                 # triage / defect / joint head definitions
│   ├── train_eval.py            # training loops, metrics (F1/AUROC/ECE/AURC)
│   ├── stats.py                 # multi-seed, bootstrap CI, paired-bootstrap, BH-FDR, DeLong
│   ├── calibration.py           # temp scaling, defect-aware calibration
│   ├── selective.py             # risk-coverage, AURC, gating policies
│   ├── vqa_confidence.py        # frozen VQA model -> per-sample conf+correct
│   ├── actionable.py            # ARR / FRR metric + defect->action map
│   ├── grounding.py             # (Phase 2) model-agnostic ground() + groundability features
│   ├── figures.py               # every paper figure, one function each
│   └── resultlog.py             # versioned JSON + manifest + RESULTS.md updater
├── results/
│   ├── E0_audit/  E1_assembly/  E2_features/  E3_triage/
│   ├── E4_defect/ E5_actionable/ E6_vqaconf/  E7_calib/ E8_ablation/
│   ├── E9_grounding/            # (Phase 2) groundability cache, triage delta, spatial examples
│   ├── figures/                 # *.pdf + *.png, paper-ready
│   ├── manifest.jsonl           # one line per completed run (provenance)
│   └── RESULTS.md               # human-readable rolling summary (auto-appended)
└── artifacts/                   # cached embeddings, trained heads, temp scalers
```

---

## 2. Global engineering rules (NON-NEGOTIABLE — Claude Code enforces on every cell)

These directly implement your CU-saving tips and fix the inefficiencies in the old notebook.

1. **Extract features exactly once, then never touch the GPU for them again.** Embeddings are cached to Drive as `.npy` + a parquet index. Every later experiment loads the cache. Re-running E2 is a no-op if the cache exists (idempotent guard).
2. **Copy data to local disk before any loop.** Never read thousands of small images straight from `/content/drive`. Stage zips → `/content/local/` → unzip locally → read from there. (Drive network latency starves the GPU.)
3. **Mixed precision everywhere on GPU.** All forward passes use `torch.autocast`. Feature extraction and VQA inference are inference-only (`torch.inference_mode()` + autocast, no grad).
4. **DataLoader discipline.** `num_workers = max(1, nproc-1)`, `pin_memory=True`, sensible `batch_size`, `persistent_workers=True`. The old MLP did full-batch single-step "training" — banned. Real minibatching with a `DataLoader`.
5. **Checkpoint + resume.** Long cells (E2, E6) write progress shards so a disconnect resumes from the last shard, not from zero.
6. **Idempotent cells.** Each experiment checks for its output artifact and skips compute if present (`FORCE_RERUN` flag to override).
7. **Kill idle runtime.** The bottom of any long-running cell calls `google.colab.runtime.unassign()` **only if** `AUTO_DISCONNECT=True` (default False while developing, True for fire-and-forget runs). Never auto-disconnect before results are flushed to Drive.
8. **Everything is logged.** Every experiment ends by calling `resultlog.log_run(...)` which writes a versioned JSON to its `results/EX/` dir, appends a line to `manifest.jsonl` (git commit hash, GPU name from `nvidia-smi`, timestamp, seed, lib versions, key metrics), and appends a summary block to `RESULTS.md`. **No result exists unless it's in `results/`.**
9. **One seed to rule them all.** `SEED=42` set for python/numpy/torch/cuda in `env.seed_everything()`. Splits are deterministic.
10. **No silent GPU.** Each cell prints `torch.cuda.get_device_name()` at the top and asserts it matches the GPU this experiment is supposed to run on (warn, don't crash).

---

## 3. GPU / Compute-Unit budget

Principle: **this paper is cheap.** It is mostly feature extraction (one pass) + tiny heads (seconds) + one frozen-VQA inference pass. Nothing here needs an A100/H100. Reserve premium GPUs only if you later fine-tune a VLM (not in this plan).

| Exp | Job | GPU to select | Why this GPU | Wall-clock (rough) | CU est. |
|---|---|---|---|---|---|
| E0 | Mount, copy-to-local, unzip, schema audit | **CPU (High-RAM)** | zero GPU work; don't burn GPU CU on I/O | 15–30 min | ~0 |
| E1 | Join VQA+Quality → master parquet | **CPU (High-RAM)** | pure pandas | 5–10 min | ~0 |
| E2 | Feature extraction, 3 backbones × images | **L4** | best throughput/CU for ViT inference; ~2× T4 speed at ~1.6× cost → finishes sooner, similar total CU, far less wall-clock. AMP-friendly. | 40–90 min | ~3–4.5 |
| E3 | Triage head (binary) | **T4** | trivial MLP; seconds of compute | <10 min | <0.3 |
| E4 | Defect head (multi-label) | **T4** | same | <10 min | <0.3 |
| E5 | Actionable Recovery metric | **CPU/T4** | numpy only | <5 min | ~0 |
| E6 | Frozen VQA confidence harvest (ViLT) | **L4** | one inference pass over val/train; ViLT is small, L4 keeps wall-clock low | 30–60 min | ~1.5–3 |
| E7 | Calibration + selective prediction | **CPU/T4** | all post-hoc on cached logits | <10 min | <0.3 |
| E8 | Ablations + all figures | **CPU/T4** | matplotlib + cached metrics | <15 min | <0.3 |
| E9 | LocateAnything groundability harvest (Phase 2) | **L4** (subsample) / **A100** (full, optional) | 3B VLM, inference-only on a 3–5k subsample; L4 is enough for the hypothesis test. A100 only if you want the full val set fast. | 45–90 min (L4, subsample) | ~2.5–4.5 |

**Phase 2 note (E9):** LA-3B is far heavier than ViLT, so E9 runs on a **subsample** (3–5k images with both VQA + quality labels) — plenty to test the groundability hypothesis and produce spatial-guidance examples. Inference-only, cached, resume-safe. If LA-3B proves impractical on your runtime (OOM, broken brand-new code), fall back to a lighter open REC model (Qwen2.5-VL-3B or Grounding DINO-T) without changing the experiment's logic. Do not run E9 on A100 unless the L4 subsample run has already validated the pipeline end-to-end — don't debug on premium GPUs.

**Total realistic budget: ~8–12 CU for the core paper (E0–E8)**, +~2.5–4.5 CU if you add the E9 Phase-2 subsample run on L4 → **~11–16 CU all-in**. This holds only if you obey rule #1 (extract once). If you re-extract features because a later cell crashed, you triple the only expensive line item — so cache religiously.

**GPU selection cheat-sheet for you in Colab:** Runtime → Change runtime type. Use **CPU+High-RAM** for E0/E1/E5/E7/E8, **L4** for E2/E6/E9, **T4** for E3/E4. Only reach for A100 if (a) you want the full-set E9 after the L4 subsample validates, or (b) we later decide to fine-tune the VQA backbone (out of scope for v1).

---

## 4. Experiment specifications

Each spec gives purpose → GPU → inputs → outputs → efficiency notes → reference code. Code is reference-quality; Claude Code organizes it into `src/` modules and notebook cells, tightens it, and adds the idempotency/logging wrappers.

### E0 — Environment & schema audit  *(CPU, High-RAM)*
**Purpose.** Fix the data mystery from the old run (E0 saw 39k images, E2 globbed only 11.7k → silent data loss). Stage data locally and *print the real schema* of every annotation file so downstream code targets real field names, not assumptions.

**Outputs.** `results/E0_audit/audit.json` (image counts per split on local disk, annotation keys, label distributions, VQA↔Quality image-ID overlap), staged data under `/content/local/`.

**Efficiency notes.** Copy zips Drive→local, unzip locally. Verify counts match expectation; if E2 later sees fewer images than E0, the join key is wrong — fail loudly here instead.

```python
# src/env.py (excerpt)
import os, sys, json, zipfile, shutil, random, subprocess
import numpy as np

SEED = 42
DRIVE_BASE = '/content/drive/MyDrive/VQA_ML/AVA_VizWiz'
LOCAL_BASE = '/content/local/AVA_VizWiz'

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def nproc():
    return int(subprocess.check_output(['nproc']).decode().strip())

def stage_zip_to_local(zip_path, dest_dir):
    """Copy a zip from Drive to local SSD and unzip there. Idempotent."""
    os.makedirs(dest_dir, exist_ok=True)
    marker = os.path.join(dest_dir, '.unzipped_ok')
    if os.path.exists(marker):
        print(f'[skip] already staged: {dest_dir}'); return dest_dir
    local_zip = os.path.join('/content', os.path.basename(zip_path))
    if not os.path.exists(local_zip):
        shutil.copy(zip_path, local_zip)
    with zipfile.ZipFile(local_zip, 'r') as z:
        z.extractall(dest_dir)
    open(marker, 'w').close()
    return dest_dir
```

```python
# E0 cell logic
import glob, json, os
from collections import Counter
from src import env

# 1. stage everything local (edit zip names to match your Drive)
zips = {
  'images_train': f'{env.DRIVE_BASE}/data_raw/zips/train.zip',
  'images_val':   f'{env.DRIVE_BASE}/data_raw/zips/val.zip',
  'vqa_annot':    f'{env.DRIVE_BASE}/data_raw/zips/Annotations.zip',
  'quality_annot':f'{env.DRIVE_BASE}/data_raw/zips/annotations.zip',  # quality issues
}
for name, zp in zips.items():
    if os.path.exists(zp):
        env.stage_zip_to_local(zp, f'{env.LOCAL_BASE}/{name}')

# 2. AUDIT: print real schema — do not assume field names
def peek_json(path, n=2):
    obj = json.load(open(path))
    sample = obj[:n] if isinstance(obj, list) else obj
    print(f'\n=== {path} ===\ntype={type(obj).__name__} len={len(obj) if hasattr(obj,"__len__") else "?"}')
    print('keys of first item:', list(sample[0].keys()) if isinstance(obj, list) else list(obj.keys()))
    print('sample:', json.dumps(sample[0] if isinstance(obj, list) else sample, indent=2)[:800])

# walk and report counts + overlap; write audit.json (Claude Code completes this)
```

---

### E1 — Master data assembly  *(CPU, High-RAM)*
**Purpose.** Build ONE table keyed by image filename with: `answerable` (from VQA), the six quality flaws + `unrecognizable` as multi-hot (from QualityIssues), `question`, `answers`, official `split`. This table drives every later experiment.

**Outputs.** `data_processed/master.parquet`, plus `results/E1_assembly/label_stats.json` (per-label positive rates, co-occurrence matrix, answerable×defect contingency — this contingency table is the empirical seed of contribution C1).

**Efficiency notes.** Pure pandas; vectorized merges; no per-row Python loops over 40k items where avoidable.

```python
# src/data_assembly.py (excerpt) — verify exact keys against E0 audit first!
import json, pandas as pd, numpy as np

QUALITY_FLAWS = ['blur','bright','dark','obstruction','framing','rotation']  # 6 flaws
# NOTE: QualityIssues json field names confirmed via E0 audit; adapt if they differ.

def load_quality(split_json):
    rows = []
    for it in json.load(open(split_json)):
        # typical fields: 'image', 'flaws'/'quality_flaws' dict, 'unrecognizable'
        flaws = it.get('flaws', it)  # adapt to real schema
        rows.append({
            'image': it['image'],
            **{f'q_{k}': int(bool(flaws.get(k, 0))) for k in QUALITY_FLAWS},
            'q_unrecognizable': int(bool(it.get('unrecognizable', 0))),
        })
    return pd.DataFrame(rows)

def load_vqa(split_json):
    rows = []
    for it in json.load(open(split_json)):
        rows.append({
            'image': it['image'],
            'question': it.get('question',''),
            'answerable': int(it.get('answerable', 1)),
            'answers': [a['answer'] for a in it.get('answers',[])],
        })
    return pd.DataFrame(rows)

def build_master(vqa_paths, quality_paths):
    vqa = pd.concat([load_vqa(p).assign(split=s) for s,p in vqa_paths.items()])
    qua = pd.concat([load_quality(p).assign(split=s) for s,p in quality_paths.items()])
    master = vqa.merge(qua, on=['image','split'], how='inner')  # inner = images with BOTH labels
    return master
```

---

### E2 — Multi-backbone feature extraction  *(L4)*  ← only expensive cell, cache hard
**Purpose.** Produce frozen embeddings for every image under three backbones so heads train in seconds and the backbone comparison (C4) is fair.

**Outputs.** `artifacts/emb_{backbone}.npy` (float16), `data_processed/feature_index.parquet` (row_idx ↔ image ↔ split). Shards saved every N batches for resume.

**Efficiency notes.** This is where your tips pay off: local-disk reads, `DataLoader(num_workers=nproc-1, pin_memory=True, persistent_workers=True)`, `autocast`, `inference_mode`, batch 128–256 on L4. **Idempotent:** skip a backbone if its `.npy` exists. Store as float16 (halves Drive footprint, fine for linear probes).

```python
# src/features.py (excerpt)
import torch, numpy as np, pandas as pd, os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDS(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.pre = paths, preprocess
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.pre(Image.open(self.paths[i]).convert('RGB')), i

@torch.inference_mode()
def extract(backbone, paths, device='cuda', bs=192, num_workers=7):
    model, preprocess, dim = load_backbone(backbone, device)  # CLIP / dinov2 / mobilenet
    dl = DataLoader(ImageDS(paths, preprocess), batch_size=bs, num_workers=num_workers,
                    pin_memory=True, persistent_workers=True)
    out = np.zeros((len(paths), dim), dtype=np.float16)
    for imgs, idx in dl:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            feat = model(imgs)                      # (B, dim)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        out[idx.numpy()] = feat.float().cpu().numpy().astype(np.float16)
    return out
# load_backbone: CLIP ViT-B/32 (HF), DINOv2 ViT-S/14 (torch.hub 'facebookresearch/dinov2'),
# MobileNetV3-Large (torchvision, penultimate layer). Claude Code writes these three loaders.
```

---

### E3 — Answerability triage head  *(T4)*
**Purpose.** Binary answerable vs not, per backbone. This reproduces/strengthens the known baseline and gives the *confidence signal* the gating policy will use.

**Outputs.** `results/E3_triage/metrics_{backbone}.json` (F1, AUROC, AUPRC, ECE), `artifacts/triage_{backbone}.pt`, cached val logits for E7.

**Efficiency notes.** Embeddings already on GPU-friendly arrays; minibatch DataLoader over tensors, AMP, AdamW (import from `torch.optim`, NOT `transformers` — that was the old crash), `class_weight`/`pos_weight` for the imbalance, early stop on val AUROC.

**Validation (per §4.5).** 5 seeds → mean±std; AUROC/AUPRC lead, 2×2 confusion matrix + F1/P/R at the `cal`-frozen `τ`; bootstrap CI on report set; baselines = majority-class + linear probe; threshold chosen on `cal` only.

```python
# src/heads.py
import torch.nn as nn
class MLPHead(nn.Module):
    def __init__(self, d_in, d_out, hidden=256, p=0.3, multilabel=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in,hidden), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(hidden, d_out))
        self.multilabel = multilabel
    def forward(self,x): return self.net(x)
# LinearHead = single nn.Linear for the "linear probe" comparison.
```

---

### E4 — Defect diagnosis head  *(T4)*
**Purpose.** Multi-label prediction of the 6 flaws + unrecognizable. Feeds C2 (actionable advice) and C1 (defect conditioning).

**Outputs.** `results/E4_defect/per_defect_auroc_{backbone}.json`, `artifacts/defect_{backbone}.pt`, cached per-sample defect predictions.

**Efficiency notes.** `BCEWithLogitsLoss(pos_weight=...)` computed from E1 label rates; report per-defect AUROC + mAP (accuracy is meaningless under heavy imbalance — do not report bare accuracy like the old draft did).

**Validation (per §4.5).** 5 seeds → mean±std; per-defect AUROC/AUPRC + macro/micro-F1 + mAP; **one-vs-rest 2×2 per defect** (NOT a 7×7 confusion matrix) + a separate defect **co-occurrence heatmap**; imbalance ablation (BCE ↔ pos_weight ↔ focal) all multi-seed; per-defect thresholds frozen on `cal`; BH-FDR correction across the per-defect tests.

---

### E5 — Actionable Recovery metric  *(CPU/T4)*  ← contribution C2
**Purpose.** Turn defect predictions into user advice and *score the advice*.

**Outputs.** `results/E5_actionable/arr_frr.json`, the defect→action table for the paper.

**Validation (per §4.5).** Report **ARR and FRR together** (never ARR alone) with per-defect breakdown; bootstrap CI on both; computed from E4's frozen predictions so it inherits the same report split.

```python
# src/actionable.py
DEFECT_TO_ACTION = {
  'blur':'Hold the camera steady and refocus',
  'dark':'Add light or move to a brighter area',
  'bright':'Reduce glare / avoid direct light',
  'obstruction':'Move your finger or object off the lens',
  'framing':'Step back so the whole item is in frame',
  'rotation':'Rotate the phone upright',
}
def actionable_recovery_rate(pred_defects, gt_flaws, answerable):
    """ARR: among quality-unanswerable images, fraction where top predicted
       defect is an actual GT flaw (=> advice would be correct).
       FRR: among answerable images, fraction wrongly told to refilm."""
    # Claude Code implements; returns {'ARR':..., 'FRR':..., 'per_defect':{...}}
```

---

### E6 — Frozen VQA confidence harvest  *(L4)*
**Purpose.** Run a frozen VQA model once over the data and cache, per sample: predicted answer, **confidence** (softmax max over answer logits), and **correctness** (VizWiz VQA accuracy vs the 10 answers). We calibrate/ gate this black box — we do not train it.

**Why ViLT (primary).** `dandelin/vilt-b32-finetuned-vqa` is a *discriminative* VQA model: it outputs a categorical distribution over ~3k answers → clean logits, ideal for temperature scaling and ECE. (A generative model like BLIP gives messy sequence-probabilities.) Small + fast on L4. Optional BLIP-VQA robustness check later.

**Outputs.** `results/E6_vqaconf/vqa_predictions.parquet` (image, question, pred, confidence, correct, + joined GT defects), one inference pass, cached.

**Efficiency notes.** `inference_mode` + autocast, batch the processor, num_workers loader for image decode. This is the second (and last) GPU-heavy cell — checkpoint shards, resume-safe.

```python
# src/vqa_confidence.py (excerpt)
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

@torch.inference_mode()
def harvest(images, questions, device='cuda', bs=32):
    proc = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
    model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa').to(device).eval()
    confs, preds = [], []
    for i in range(0, len(images), bs):
        enc = proc(images[i:i+bs], questions[i:i+bs], return_tensors='pt',
                   padding=True, truncation=True).to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(**enc).logits           # (B, n_answers)
        prob = logits.softmax(-1)
        c, idx = prob.max(-1)
        confs += c.float().cpu().tolist()
        preds += [model.config.id2label[j] for j in idx.cpu().tolist()]
    return preds, confs
# vqa_accuracy(pred, answers): standard VizWiz min(#matches/3, 1) metric -> Claude Code adds.
```

---

### E7 — Calibration & selective prediction  *(CPU/T4)*  ← contribution C1 (the headline)
**Purpose.** The core result. Compare global vs **defect-aware** confidence gating.

**Steps.**
1. Temperature-scale VQA confidences on a calibration split; report ECE before/after.
2. Build risk–coverage curves; compute **AURC** for: (a) global threshold, (b) per-defect thresholds, (c) "diagnosed-defect-conditioned" policy that uses E4's predicted defect to pick the threshold.
3. Show (c) ≈ (b) and both beat (a) → defect awareness helps *even using predicted (not GT) defects* = practical.

**Outputs.** `results/E7_calib/{ece.json, risk_coverage.json, aurc_comparison.json}`, reliability diagrams + risk–coverage figure data.

**Validation (per §4.5 — the headline).** `T` and all gate thresholds chosen on `cal`, applied to `rep`. Primary claim "defect-aware < global AURC" tested by **paired bootstrap on the AURC delta** (report Δ, 95% CI, p); if the CI crosses 0 we reframe honestly. Also report ΔECE (raw↔temp↔defect-aware) with bootstrap CI, Brier, and a random-confidence selective baseline as the floor.

```python
# src/calibration.py + src/selective.py (excerpts)
import numpy as np, torch
def temperature_scale(logits, labels):           # 1-param LBFGS on NLL
    T = torch.nn.Parameter(torch.ones(1)); opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits/T, labels); loss.backward(); return loss
    opt.step(closure); return T.item()

def ece(confs, correct, n_bins=15):
    bins = np.linspace(0,1,n_bins+1); e=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(confs>lo)&(confs<=hi)
        if m.sum(): e += m.mean()*abs(correct[m].mean()-confs[m].mean())
    return float(e)

def risk_coverage(confs, correct):
    order=np.argsort(-confs); c=correct[order]
    cov=np.arange(1,len(c)+1)/len(c); risk=1-np.cumsum(c)/np.arange(1,len(c)+1)
    aurc=np.trapz(risk,cov); return cov,risk,float(aurc)
```

---

### E8 — Ablations & figures  *(CPU/T4)*
**Purpose.** C3 (joint vs cascade), C4 (backbone table), and every paper figure from cached metrics.

**Figures (all to `results/figures/` as PDF+PNG, consistent palette; every data figure shows error bars or CI bands per §4.5).**
- F1: pipeline schematic (drawn, not data).
- F2: defect **co-occurrence** heatmap + answerable×defect contingency (from E1) — motivates C1. *(labeled co-occurrence, not confusion)*
- F3: per-defect AUROC + AUPRC bar chart across backbones, with error bars (E4).
- F4: reliability diagram before/after temperature scaling, with ECE annotated (E7).
- F5: **risk–coverage curves with CI bands, global vs defect-aware + random baseline (E7)** — the money figure; annotate AURC delta + p.
- F6: ARR **and** FRR per defect, error bars (E5).
- F7: backbone comparison table → rendered figure with mean±std (E3/E4).
- F8: triage ROC (2×2 confusion matrix inset at frozen τ) + per-defect one-vs-rest panels (E3/E4).
- F9: **qualitative grid** — sampled TP/FP/FN per defect + the high-confidence-wrong "danger panel" (E3/E4/E5), seed-documented.
- F10: *(Phase 2, if E9 ran)* triage ROC with vs. without groundability (ΔAUROC + p) + spatial-guidance panels incl. misfires.

**Efficiency notes.** Pure matplotlib over cached JSON/logits. One function per figure in `figures.py`, each returns the saved path and logs it. `reproduce.sh` calls them all → regenerates the entire figure set without touching a GPU.

---

### E9 — Groundability-aware reliability & spatial guidance  *(L4 subsample / A100 optional)*  ← contribution C5, **Phase 2, optional**
**Purpose.** Test RQ3's grounding extension. Run frozen LocateAnything-3B over a subsample, ground the *queried entity* from each question, and derive (a) a groundability feature for triage and (b) spatial corrective guidance. **Do not start E9 until E0–E8 are complete and committed** — the core paper must stand alone first.

**Steps.**
1. *Entity extraction.* From each question, extract the referred noun phrase (spaCy noun-chunk or a tiny prompt to the VQA tokenizer's vocabulary; keep it simple and deterministic — log the method). This phrase is the grounding query.
2. *Grounding harvest.* Prompt LA-3B with the image + phrase using its REC prompt template ("Locate a single instance that matches the following description: [PHRASE]."). Cache per sample: returned box(es) or Negative-Block (no target), grounding/format confidence signals, and box geometry.
3. *Groundability features.* Derive: `grounded` (bool), `n_boxes`, `max_conf`, `box_area_frac`, `touches_border` (any edge within ε), `centeredness`. These become extra triage features.
4. *RQ3a — does groundability help triage?* Re-train the E3 triage head with vs. without these features on the **same subsample split**; report ROC/AUPRC delta. Hypothesis: ungroundable entity ⇒ more likely unanswerable.
5. *RQ3b — spatial guidance.* Combine the grounded box with the E4 defect prediction into directional advice (`touches_border` + framing ⇒ "pan toward the object"; low grounding conf under `dark` ⇒ "the area you're asking about is too dark"). Produce a qualitative panel + a small human-readable rubric score (no new heavyweight metric needed for v1).

**Outputs.** `results/E9_grounding/{grounding_cache.parquet, triage_delta.json, spatial_examples.json, subsample_ids.json}`, qualitative panels for F8.

**Validation (per §4.5).** RQ3a tested on the **identical subsample split** (saved `subsample_ids.json`) for both triage heads: report ΔAUROC/ΔAUPRC with **paired bootstrap + DeLong's test**; appearance-only triage is the control to beat. Subsample size caps statistical power — report the achieved CI width as the honest limit. Spatial-guidance panels sampled by rule (`QUAL_SEED`), including grounding misfires.

**Efficiency notes.** Inference-only, `inference_mode` + autocast, batch the processor, **subsample only** (3–5k). Shard + resume. **Fallback:** if LA-3B is impractical, swap in Qwen2.5-VL-3B or Grounding DINO-T behind the same `ground(image, phrase) -> boxes+conf` interface in `src/grounding.py`; the rest of E9 is model-agnostic. Cache the entity-extraction output separately so you never re-run NLP when only the grounder changes.

```python
# src/grounding.py (excerpt) — model-agnostic interface; LA-3B primary, fallbacks behind same signature
@torch.inference_mode()
def ground(image, phrase, model, processor, device='cuda'):
    """Return {'boxes':[[x1,y1,x2,y2]...], 'conf':float, 'grounded':bool} in [0,1000] norm space.
       LA-3B uses its REC template; Negative-Block -> grounded=False. Fallbacks adapt their own output."""
    # Claude Code implements per-backend; the harvest loop only ever calls this function.

def groundability_features(g, img_w, img_h, eps=0.02):
    if not g['grounded']: return dict(grounded=0, n_boxes=0, max_conf=0.0,
                                      box_area_frac=0.0, touches_border=0, centeredness=0.0)
    # compute area frac, border touch, centeredness from g['boxes']; Claude Code completes.
```

---

## 4.5 Validation & Statistical Protocol (read before writing any training code)

This section is the antidote to how the old draft died: it reported per-class **accuracy** on an imbalanced **multi-label** problem (meaningless), with single-run point estimates, no error bars, no significance test, and cherry-picked qualitative examples. Every rule below is enforced by Claude Code inside `train_eval.py`, `calibration.py`, `selective.py`, and `figures.py`. **No headline number ships without an error bar; no comparison claim ships without a significance test; no threshold is chosen on the data it's reported on.**

### 4.5.1 Data splits & leakage discipline
- **Three disjoint partitions, deterministic (SEED=42):**
  - **train** — VizWiz train: fit head weights only.
  - **calibration (cal)** — a fixed slice carved from VizWiz val (≈30% of val, stratified by `answerable`): the ONLY place temperature `T`, decision thresholds `τ`, and defect-gate thresholds are selected.
  - **report (rep)** — the remaining val: every number in the paper comes from here, untouched until the end.
  - **test** — VizWiz test has hidden VQA answers, so it is eval-only for triage/defect tasks where labels exist; never used to pick anything.
- **Frozen-knob rule.** Select `T` and all `τ` on `cal`, freeze them, apply to `rep`. Selecting a threshold on the report set silently inflates selective-prediction results and invalidates the whole RQ2 claim. Claude Code asserts that no threshold-selection function is ever called with `rep` data.
- **Subsample provenance (E9).** The Phase-2 subsample is drawn once with a logged seed and the index list saved to `results/E9_grounding/subsample_ids.json`, so E9's triage-with-groundability delta is computed on a split that exactly matches its no-groundability control.

### 4.5.2 Metrics per task (and what NOT to report)
| Task | Report these | Never report |
|---|---|---|
| Triage (binary, E3) | **AUROC, AUPRC** (lead with these), F1/precision/recall at the frozen `τ`, Balanced Accuracy, 2×2 confusion matrix at `τ` | bare accuracy as the headline (imbalanced) |
| Diagnosis (multi-label, E4) | **per-defect AUROC + AUPRC**, macro/micro-F1, **mAP**, per-defect P/R at frozen per-defect `τ` | a single multi-class confusion matrix (undefined for multi-label); per-class accuracy |
| Selective prediction (E7) | **AURC** (primary), risk–coverage curve, coverage@risk≤k, **ECE + reliability diagram**, Brier score | accuracy without coverage context |
| Actionable recovery (E5) | **ARR, FRR**, per-defect breakdown | ARR without FRR (you can game one alone) |
| Groundability (E9) | triage **ΔAUROC / ΔAUPRC** with vs. without groundability features; qualitative spatial panels | grounding IoU as if detection were our task |

**Confusion-matrix policy (resolves the old bug):** confusion matrices are valid ONLY for single-label problems. Triage → one 2×2 at the frozen threshold. Diagnosis → **one-vs-rest 2×2 per defect**, never a 7×7. The 7×7 "matrix" that *is* meaningful is the **defect co-occurrence heatmap** (how often defects appear together) — label it as co-occurrence, not confusion, so reviewers aren't misled.

### 4.5.3 Uncertainty: error bars on everything
- **Multi-seed training.** Every trainable head (triage, defect, cascade, unified, each backbone, each loss variant) is trained with **`SEEDS=[0,1,2,3,4]`** (5 runs). Report **mean ± std** and a **95% CI** for every metric. `train_eval.py` returns a per-seed array, never a scalar.
- **Metric-level CIs on frozen predictions.** For metrics computed on cached logits (AUROC, AURC, ECE, ARR), also compute a **bootstrap CI over the report samples** (`N_BOOT=2000`, resample with replacement, percentile interval). This separates *model-init variance* (seeds) from *evaluation-set variance* (bootstrap) — report both.
- Tables show `mean ± std`; figures show shaded CI bands (risk–coverage, ROC) or error bars (bar charts).

### 4.5.4 Significance testing for every comparison claim
Each paper claim of the form "A beats B" gets a real test, not eyeballing:
- **Unified vs. cascade (C3, RQ1):** paired bootstrap on the **AUROC/F1 difference** over report samples → report Δ, 95% CI, p-value. Paired = same samples scored by both models.
- **Defect-aware vs. global gating (C1, RQ2 — the headline):** paired bootstrap on the **AURC difference**. This is the single most important significance test in the paper; if its CI crosses 0, the central claim is unsupported and we reframe honestly.
- **Groundability feature helps triage (C5, RQ3a):** paired bootstrap on **ΔAUROC** between the two triage heads on the identical subsample split; **DeLong's test** as a parametric cross-check for AUROC specifically.
- **Calibration improvement (RQ2):** bootstrap CI on **ΔECE** (raw vs. temperature-scaled vs. defect-aware).
- Multiple-comparison note: when reporting many per-defect tests at once, apply **Benjamini–Hochberg** FDR control and say so.

### 4.5.5 Baselines (the floors every result must clear)
- **Majority-class** predictor (triage) and **per-class base-rate** predictor (diagnosis) — the "did we learn anything" floor.
- **Random-confidence selective baseline** (RQ2) — a flat risk–coverage line; defect-aware gating must beat it AND the global-threshold baseline.
- **Linear probe** vs. **MLP head** — cheap-vs-expressive floor for RQ1.
- **Appearance-only triage** — the control the groundability feature (C5) must beat.
- **Plain-BCE diagnosis head** — the control the `pos_weight`/focal variants must beat.

### 4.5.6 Qualitative analysis: sampled, not curated
Qualitative figures are evidence only if the sampling rule is stated and reproducible (`QUAL_SEED=7`, documented selection):
- **Per defect:** N true positives, N false positives, N false negatives — drawn by rule, not hand-picked.
- **The danger panel (most important):** highest-confidence **wrong** triage predictions — model says "answerable & confident" but the photo is genuinely unanswerable. These failures are the paper's most informative figure.
- **RQ2 face:** concrete cases where global threshold answers a doomed photo but the defect-aware gate correctly abstains.
- **RQ3 (Phase 2):** spatial-guidance panels (grounded box + defect + directional advice), *including* grounding misfires.
- Every qualitative figure caption states the sampling rule and seed.

### 4.5.7 Ablation matrix (one variable changed at a time, all multi-seed)
| Ablation | Varies | Holds fixed | Validates |
|---|---|---|---|
| Architecture (C3) | unified multi-task head ↔ cascade | data, backbone, seeds | RQ1, error-propagation claim |
| Backbone (C4) | CLIP ↔ DINOv2 ↔ MobileNet | head, data | RQ1 robustness + benchmark |
| Head capacity | linear probe ↔ MLP | backbone, data | feature-vs-model contribution |
| Imbalance handling | BCE ↔ pos_weight ↔ focal | head, data | fixes the old paper's collapse |
| Calibration (RQ2) | raw ↔ temp-scaled ↔ defect-aware | logits | each calibration step's value |
| Groundability (C5) | appearance ↔ appearance+grounding | split, head | RQ3a delta |
| Grounder choice (C5) | LA-3B ↔ fallback | E9 pipeline | robustness of the grounding claim |

### 4.5.8 Reproducibility ledger
Beyond per-run logging (Section 5), the paper needs: pinned `requirements.txt` with exact versions; the SEED list and split index files committed; `nvidia-smi` GPU string per run; git commit hash per result (already in `manifest.jsonl`); and a one-command `reproduce.sh` that regenerates every table/figure from cached logits + metrics JSON (so a reviewer — or you, in three months — can rebuild the paper without re-running GPUs). IEEE Access reviewers reward this concretely.

### 4.5.9 Threats to validity (write this paragraph in the paper)
State plainly: VizWiz quality labels are crowd-sourced (annotator noise); answerability is partly subjective; the frozen VQA model (ViLT) bounds the selective-prediction ceiling (calibrating a weak model ≠ making it correct); groundability depends on entity extraction from the question (error source); subsample size (E9) limits the power of the RQ3a test — report the achieved CI width as the honest measure of that limit.

---
# src/resultlog.py (excerpt)
import json, os, time, subprocess, torch, sys
def _git_hash():
    try: return subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
    except: return 'nogit'
def log_run(exp_id, metrics:dict, params:dict, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    rec = {'exp':exp_id,'time':time.strftime('%Y-%m-%d %H:%M:%S'),
           'git':_git_hash(),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
           'python':sys.version.split()[0],'torch':torch.__version__,
           'params':params,'metrics':metrics}
    json.dump(rec, open(os.path.join(results_dir,f'{exp_id}_{rec["git"]}.json'),'w'), indent=2)
    with open('results/manifest.jsonl','a') as f: f.write(json.dumps(rec)+'\n')
    with open('results/RESULTS.md','a') as f:
        f.write(f'\n### {exp_id} ({rec["time"]}, {rec["gpu"]}, {rec["git"]})\n')
        f.write('```json\n'+json.dumps(metrics,indent=2)+'\n```\n')
```

**Rule:** if `log_run` wasn't called, the experiment didn't happen. The paper is written *only* from `results/RESULTS.md` + the JSONs + figures.

### 5.1 Statistics helpers (`src/stats.py`) — shared by all experiments
The validation protocol (§4.5) is implemented once here and called everywhere, so seeds, bootstrap, and significance are consistent across experiments.

```python
# src/stats.py (excerpt)
import numpy as np

def multi_seed(train_fn, seeds=(0,1,2,3,4), **kw):
    """Run train_fn(seed=s, **kw) -> metrics dict for each seed; return per-metric arrays + mean/std/ci."""
    runs = [train_fn(seed=s, **kw) for s in seeds]
    keys = runs[0].keys()
    agg = {}
    for k in keys:
        v = np.array([r[k] for r in runs], float)
        agg[k] = dict(mean=v.mean(), std=v.std(ddof=1), ci95=1.96*v.std(ddof=1)/np.sqrt(len(v)),
                      seeds=v.tolist())
    return agg

def bootstrap_ci(metric_fn, *arrays, n_boot=2000, seed=42):
    """Percentile 95% CI of metric_fn(*arrays) by resampling sample indices with replacement."""
    rng = np.random.default_rng(seed); n = len(arrays[0]); stats=[]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(*[a[idx] for a in arrays]))
    lo, hi = np.percentile(stats, [2.5, 97.5]); return float(np.mean(stats)), float(lo), float(hi)

def paired_bootstrap_delta(metric_fn, y, score_a, score_b, n_boot=2000, seed=42):
    """Paired test of metric(A) - metric(B) on the SAME resampled indices. Returns delta, CI, p(two-sided)."""
    rng = np.random.default_rng(seed); n=len(y); deltas=[]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(metric_fn(y[idx], score_a[idx]) - metric_fn(y[idx], score_b[idx]))
    deltas=np.array(deltas); lo,hi=np.percentile(deltas,[2.5,97.5])
    p = 2*min((deltas<=0).mean(), (deltas>=0).mean())
    return float(deltas.mean()), float(lo), float(hi), float(p)

def benjamini_hochberg(pvals, alpha=0.05):
    """Return boolean reject array under BH-FDR for the per-defect multiple tests."""
    p=np.asarray(pvals); order=np.argsort(p); m=len(p); thresh=alpha*(np.arange(1,m+1))/m
    passed=p[order]<=thresh; k=np.where(passed)[0]
    cut = order[:k.max()+1] if len(k) else []
    out=np.zeros(m,bool); out[cut]=True; return out
# delong_auroc(y, a, b): DeLong's test for correlated AUROCs -> Claude Code adds (RQ3a cross-check).
```

---

## 6. Order of operations (what you actually do)

**Phase 1 — core paper (E0–E8), this is the submittable unit:**
1. Run the Claude Code prompt (separate file) → it scaffolds the repo + notebook and pushes.
2. `git pull` in Colab (or open the notebook from the repo).
3. Set runtime per Section 3, run cells **in order E0→E8**, one at a time, reading each cell's printed summary before moving on.
4. After E8, commit `results/` back to the repo.

**Phase 2 — grounding extension (E9), only after Phase 1 is committed:**
5. Switch to L4, run E9 on the subsample. If LA-3B misbehaves, flip the `GROUNDER` flag to the fallback model and re-run — the harvest loop is unchanged.
6. Commit `results/E9_grounding/` + F8.

**Then:**
7. Hand `RESULTS.md` + figures back to me → we write the paper section by section. If E9 didn't pan out, we write the When/Why paper (RQ1+RQ2 fully, RQ3 = ARR/FRR only) and the title drops "Where". Nothing is wasted.

## 7. Open decisions for you (answer before Claude Code runs)
- **Backbones:** all three (CLIP+DINOv2+MobileNet) or start CLIP-only to save the E2 budget, add others if a reviewer-grade table is needed? *(Recommend: CLIP+MobileNet first; DINOv2 only if time.)*
- **Captions stretch (C-stretch):** include the information-sufficiency analysis (4th contribution, +1 dataset, +modest compute) or hold it as future work? *(Recommend: hold for v1.)*
- **Data scope:** train+val (both have answerable + quality labels) — test has hidden VQA labels so it's eval-only for triage. Confirm we train on train, calibrate on a val slice, report on the rest of val.
- **Phase 2 (E9 / LocateAnything):** commit to it now so Claude Code scaffolds `grounding.py` + E9 cell (run later), or leave it out entirely for v1? *(Recommend: scaffold it now but leave the cell unrun until E0–E8 are done — zero cost to have it ready, and you decide based on how the core results look.)*
- **Grounder choice if E9 runs:** LA-3B primary with Qwen2.5-VL-3B fallback (both VLMs, REC-capable) — or do you want Grounding DINO-T as the fallback instead (lighter, detection-style, but not instruction-driven)? *(Recommend: Qwen2.5-VL-3B fallback — keeps the REC framing consistent.)*
