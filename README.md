# Reliable Assistive VQA

*Defect-Aware Refusal and Retake Guidance for Assistive Visual Question Answering: An Offline Reliability Study*

**Target venue:** IEEE Access  
**Runtime:** Google Colab Pro+ (cell-by-cell, GPU per experiment)  
**Data:** VizWiz-VQA + VizWiz-QualityIssues (human-annotated labels only)

---

## How to run

Open `notebooks/reliable_vqa_master.ipynb` in Colab (from GitHub). Run the
**SETUP** cell first on every fresh runtime, then either run cells in order or
just **Runtime → Run all**: every experiment checks its Drive `DONE.json`
marker + artifacts and **skips itself instantly if it already completed**.

| Cell | Experiment | Colab runtime | ~Wall-clock (first run) |
|------|-----------|--------------|-------------------------|
| SETUP | Clone/update repo, install missing deps only | any | 1–3 min |
| E0   | Environment & schema audit | CPU + High-RAM (or the GPU you'll use next) | 15–30 min (staging) |
| E1   | Master data assembly | CPU | 5–10 min |
| E2   | Feature extraction (CLIP + MobileNet) | **GPU — A100 fastest, L4/T4 fine** | A100 ~20–30 min, L4 40–90 min |
| E3   | Triage head (5-seed) | any GPU | < 10 min |
| E4   | Defect diagnosis head (5-seed) | any GPU | < 10 min |
| E5   | Actionable Recovery (ARR/FRR) | CPU | < 5 min |
| E6   | Frozen ViLT confidence harvest | **GPU — A100 fastest** | A100 ~15–25 min, L4 30–60 min |
| E7   | Calibration + selective prediction | CPU/any | < 10 min |
| E8   | Ablations + all figures (F1–F9) | CPU/any | < 15 min |
| E9   | Groundability (Phase 2, **GATED**) | GPU | 45–90 min |

**Cheapest workflow (2 sessions):** (1) CPU High-RAM: SETUP → E0 → E1.
(2) A100 or L4: SETUP → Run all — E0/E1 skip, E2–E8 run.

Batch sizes auto-scale to the GPU tier (A100 / L4 / T4), with bf16 autocast +
TF32 on A100/L4. E2/E6/E9 refuse to run on a CPU runtime instead of silently
crawling for hours (`VQA_ALLOW_CPU=1` overrides for debugging).

For the full reviewer-grade three-backbone table on an A100, set
`os.environ['VQA_BACKBONES'] = 'clip,mobilenet,dinov2'` in the SETUP cell.

---

## Where outputs are saved

Open the notebook from GitHub, but keep generated work on Google Drive.
By default `src/config.py` stores all resumable outputs under:

```text
/content/drive/MyDrive/VQA_ML/AVA_VizWiz/reliable_vqa_outputs/
  data_processed/
  artifacts/
  results/
```

After a Colab restart, rerunning cells will find completed Drive caches and
skip or resume instead of recomputing. Each experiment writes a `DONE.json`
marker in its results folder when it finishes; delete that file (or set
`VQA_FORCE_RERUN=1`) to force a rerun. Set `VQA_PERSIST_OUTPUTS_TO_DRIVE=0`
only for local debugging.

---

## Resuming a crashed cell

Every long cell (E2, E6, E9) checkpoints in shards. Simply re-run the cell —
idempotency guards detect partial work and resume from the last shard.

To force a full re-run:  set `FORCE_RERUN = True` in `src/config.py`.

---

## Reproduce all figures from cached results (no GPU)

```bash
bash reproduce.sh
```

This regenerates F1–F10 from `results/**/*.json` without touching any GPU.
Requires Python environment with requirements.txt installed.

---

## Repository structure

```
src/
  experiments/      — e0_audit.py … e9_grounding.py: ALL experiment logic
                      (notebook cells are thin wrappers; fixes ship via git pull)
  config.py         — all paths, seeds, GPU-tier batch sizes, env-var flags
  expstate.py       — DONE markers: skip-if-done for Run All
  staging.py        — Drive zip discovery, local staging, image path resolution
  env.py            — mount Drive, seed, GPU tier/bf16/TF32, cal/rep split
  data_assembly.py  — join VQA + QualityIssues → master.parquet
  features.py       — CLIP / DINOv2 / MobileNet extraction (cache once)
  heads.py          — LinearHead, MLPHead, JointHead
  train_eval.py     — training loop (torch.optim.AdamW), metrics, threshold selection
  stats.py          — multi_seed, bootstrap_ci, paired_bootstrap_delta, BH-FDR, DeLong
  calibration.py    — temperature scaling, ECE, defect-aware calibration
  selective.py      — risk-coverage, AURC, gating policies
  vqa_confidence.py — frozen ViLT/BLIP harvest (discriminative + generative)
  text_features.py  — CLIP text-tower question embeddings (E10)
  actionable.py     — GDMR, AIRB (legacy ARR/FRR keys), defect→action map
  grounding.py      — Phase 2: LA-3B / Qwen2.5-VL, groundability features
  figures.py        — F1–F10, one function each, PDF+PNG output
  resultlog.py      — versioned JSON + manifest.jsonl + RESULTS.md

notebooks/
  reliable_vqa_master.ipynb   — THE notebook (E0 → E9)
  revision_experiments.ipynb  — review-response pack (E4b/E5c/E5d/E10/E7e/E7f/E8f/E6c)

results/
  E*/               — per-experiment JSON metrics
  figures/          — F1–F10 PDF + PNG
  manifest.jsonl    — one line per completed run (git hash, GPU, seed, metrics)
  RESULTS.md        — human-readable rolling summary

artifacts/
  emb_{backbone}.npy       — float16 embeddings (cache; never re-extract)
  triage_{backbone}.pt     — saved triage head weights
  defect_{backbone}.pt     — saved defect head weights
```

---

## Contributions (matched to the manuscript — do not restate stronger claims here)

| # | Label | Description |
|---|-------|-------------|
| C1 | Model-specific selective prediction | Auxiliary triage/defect signals do **not** significantly improve the ViLT confidence ranking after BH-FDR correction, but do improve the weaker BLIP sequence-probability ranking (up to +0.0179 AURC) |
| C2 | Offline guidance proxies | GDMR (Guidance–Defect Match Rate) + AIRB (Answerable-Image Retake Burden), with explicit-denominator refusal-gated variants; offline proxies, not user outcomes |
| C3 | Unified vs. cascade | Small observed AUROC differences between joint and cascade heads |
| C4 | Frozen-backbone benchmark | CLIP / DINOv2 / MobileNetV3 under identical heads with calibration diagnostics |
| C5 | Groundability (Phase 2, exploratory) | Grounding-signal experiments; not part of the submitted paper's claims |

---

## Key engineering rules (non-negotiable)

1. Features extracted **once** and cached; every later cell loads the cache.
2. Data staged to `/content/local/` before any loop — never read from Drive in a loop.
3. All forward passes use `torch.autocast`; inference cells use `torch.inference_mode()`.
4. `DataLoader` with `num_workers`, `pin_memory`, `persistent_workers` everywhere.
5. Every experiment calls `resultlog.log_run()` — if it didn't log, it didn't happen.
6. Thresholds / temperature selected **only on the cal split** (enforced by assertion).
7. Every headline number = mean±std over 5 seeds + bootstrap CI.
8. Every "A beats B" claim = paired-bootstrap p-value.

---

## Data requirements

Nothing to do manually: **E0 auto-downloads anything missing** from the
official VizWiz mirror (vizwiz.cs.colorado.edu) — train images ~10.6 GB, val
images ~3.3 GB, both annotation zips a few MB — stages it locally, and copies
the zip back to Drive (`.../data_raw/zips/`) so future sessions skip the
download. Disable with `VQA_AUTO_DOWNLOAD=0`.

If you already have the files on Drive, any folder under
`VQA_ML/AVA_VizWiz` works: discovery classifies zips by *content* (VQA JSONs
have `"answerable"`, QualityIssues have `"flaws"`), not by filename. Unzipped
annotation JSONs on Drive are also accepted.

Schemas were verified against the real downloads (2026-07): quality flaw keys
are `BLR/BRT/DRK/OBS/FRM/ROT` vote counts from 5 crowdworkers, binarized at
>=2 votes per the dataset paper (`data_assembly.MIN_VOTES`).
