# Reliable Assistive VQA

*Knowing When, Why, and Where to Refuse: A Defect-Aware Reliability Layer for Assistive Visual Question Answering*

**Target venue:** IEEE Access  
**Runtime:** Google Colab Pro+ (cell-by-cell, GPU per experiment)  
**Data:** VizWiz-VQA + VizWiz-QualityIssues (human-annotated labels only)

---

## How to run (order matters)

Open `notebooks/reliable_vqa_master.ipynb` in Colab. Run cells **one at a time**.
Read each cell's printed summary before proceeding.

| Cell | Experiment | Colab runtime | ~Wall-clock |
|------|-----------|--------------|-------------|
| E0   | Environment & schema audit | **CPU + High-RAM** | 15–30 min |
| E1   | Master data assembly | CPU + High-RAM | 5–10 min |
| E2   | Feature extraction (CLIP + MobileNet) | **L4** | 40–90 min |
| E3   | Triage head (5-seed) | **T4** | < 10 min |
| E4   | Defect diagnosis head (5-seed) | **T4** | < 10 min |
| E5   | Actionable Recovery (ARR/FRR) | CPU | < 5 min |
| E6   | Frozen ViLT confidence harvest | **L4** | 30–60 min |
| E7   | Calibration + selective prediction | CPU/T4 | < 10 min |
| E8   | Ablations + all figures (F1–F9) | CPU/T4 | < 15 min |
| E9   | Groundability (Phase 2, **GATED**) | L4 | 45–90 min |

**GPU selection:** Runtime → Change runtime type.

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
  config.py         — all paths, seeds, GPU hints, flags
  env.py            — mount Drive, stage zips, seed, cal/rep split
  data_assembly.py  — join VQA + QualityIssues → master.parquet
  features.py       — CLIP / DINOv2 / MobileNet extraction (cache once)
  heads.py          — LinearHead, MLPHead, JointHead
  train_eval.py     — training loop (torch.optim.AdamW), metrics, threshold selection
  stats.py          — multi_seed, bootstrap_ci, paired_bootstrap_delta, BH-FDR, DeLong
  calibration.py    — temperature scaling, ECE, defect-aware calibration
  selective.py      — risk-coverage, AURC, gating policies
  vqa_confidence.py — frozen ViLT harvest (discriminative VQA → clean softmax)
  actionable.py     — ARR, FRR, defect→action map
  grounding.py      — Phase 2: LA-3B / Qwen2.5-VL, groundability features
  figures.py        — F1–F10, one function each, PDF+PNG output
  resultlog.py      — versioned JSON + manifest.jsonl + RESULTS.md

notebooks/
  reliable_vqa_master.ipynb   — THE notebook (E0 → E9)

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

## Contributions

| # | Label | Description |
|---|-------|-------------|
| C1 | Defect-aware selective prediction | Defect-conditioned gating beats a global threshold on AURC |
| C2 | Actionable Recovery | ARR + FRR metric for scoring corrective guidance |
| C3 | Unified vs. cascade | Quantifies error propagation between joint and cascade heads |
| C4 | Modern benchmark | CLIP / DINOv2 / MobileNet with full calibration metrics |
| C5 | Groundability (Phase 2) | Grounding signal improves triage; spatial guidance |

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

- VizWiz-VQA annotations (train.json, val.json) → Drive at `VQA_ML/AVA_VizWiz/data_raw/zips/Annotations.zip`
- VizWiz-QualityIssues annotations → Drive at `.../data_raw/zips/annotations.zip`
- VizWiz images (train, val) → Drive at `.../data_raw/zips/train.zip`, `val.zip`

Update `config.RAW_ZIPS` if your Drive layout differs. E0 will print the real
JSON schemas; update `data_assembly.py FIELD_MAP_*` if field names differ.
