# Agent Handoff — Reliable Assistive VQA

> **Purpose of this file:** a cold-start briefing so a *new* AI chat (or collaborator) can help
> immediately without re-reading the whole history. If anything here conflicts with
> [`PIPELINE.md`](PIPELINE.md), **PIPELINE.md wins** — it is the single source of truth.

---

## 1. What this project is

We are building a **reproducible ML research codebase** for an IEEE Access paper:

> *Knowing When, Why, and Where to Refuse: A Defect-Aware Reliability Layer for
> Assistive Visual Question Answering*

**We are NOT building a better VQA model.** We build a **reliability layer** that wraps any
frozen VQA model and, before trusting an answer, decides:

1. **Triage** — is this photo answerable at all?
2. **Diagnose** — which quality defect is responsible (blur, dark, bright, obstruction,
   framing, rotation, unrecognizable)?
3. **Calibrate & abstain** — is the VQA model's confidence trustworthy *given the diagnosed
   defect*, and should we answer or ask for a retake?
4. **Guide** — what corrective action should the blind user take, and (Phase 2) *where*?

**Every label is a real human VizWiz annotation. No heuristic/keyword labels anywhere** —
that was the fatal flaw of the previous draft (it keyword-matched "intent" from the question
and predicted it from the image alone → ill-posed, auto-reject).

### Datasets (all real labels)
- **VizWiz-VQA (2018):** `answerable` flag, 10 answers/question (VQA accuracy), question text.
- **VizWiz-QualityIssues (2020):** 6 quality flaws + `unrecognizable` flag, train/val/test json.
- User already downloaded both to Google Drive.

---

## 2. Research questions → experiments → contributions

**RQ1 (When/Why):** Can a light head on frozen embeddings jointly predict answerability +
defect, and does a *unified* head beat a *cascade*? → **E3, E4, E8** · contributions **C3, C4**

**RQ2 (Trust):** Does the optimal abstention policy depend on the diagnosed defect? Does a
defect-conditioned confidence gate beat a single global threshold on AURC — *even with
predicted (not GT) defects*? → **E6, E7** · contribution **C1 (the headline)**

**RQ3 (Where/How):** Can we produce correct corrective guidance (ARR/FRR), and — Phase 2 —
does grounding the queried entity give a groundability signal + spatial guidance?
→ **E5** (core) · **E9** (Phase 2) · contributions **C2, C5**

| Contribution | One line |
|---|---|
| **C1** | Defect-conditioned gating dominates a global threshold on risk–coverage (lower AURC). |
| **C2** | Actionable Recovery Rate (ARR) + False-Refilm Rate (FRR) — scoring corrective *advice*. |
| **C3** | Unified multi-task head vs. cascade — quantifies error propagation. |
| **C4** | Modern benchmark: CLIP / DINOv2 / MobileNet under identical heads + full calibration. |
| **C5** | Groundability-aware reliability + spatial guidance (Phase 2, optional; paper stands without it). |

**C1–C4 form a complete, submittable paper. C5/E9 is strictly downstream.**

---

## 3. The experiments (E0–E9) and their GPU

| Exp | Job | Colab runtime | Notes |
|-----|-----|---------------|-------|
| **E0** | Environment & schema audit | **CPU + High-RAM** | Prints real JSON field names; catches the old silent data-loss bug. |
| **E1** | Master data assembly → `master.parquet` | **CPU + High-RAM** | Pure pandas inner-join on (image, split). |
| **E2** | Feature extraction (CLIP + MobileNet) | **L4** | Only expensive cell. Cache hard (float16 `.npy`). |
| **E3** | Triage head (binary, 5-seed) | **T4** | AUROC/AUPRC + 2×2 CM at frozen τ. |
| **E4** | Defect head (multi-label, 5-seed) | **T4** | per-defect AUROC/AUPRC + mAP + one-vs-rest 2×2. |
| **E5** | Actionable Recovery (ARR/FRR) | **CPU** | numpy only. |
| **E6** | Frozen VQA confidence harvest (ViLT) | **L4** | 2nd/last GPU-heavy cell; cache. |
| **E7** | Calibration + selective prediction | **CPU/T4** | **C1 headline**: paired-bootstrap AURC delta. |
| **E8** | Ablations (C3, C4) + figures F1–F9 | **CPU/T4** | matplotlib over cached JSON. |
| **E9** | Groundability (Phase 2) | **L4** | **GATED behind `RUN_E9=False`.** Do not run until E0–E8 committed. |

**Total budget ~8–12 CU for E0–E8**, +~2.5–4.5 CU for the E9 subsample. This holds only if
features are extracted **once** (rule #1 below) — re-extraction triples the only expensive line.

---

## 4. Repository layout

```
VQA-paper/  (https://github.com/meteorboyF/VQA-paper.git, branch main)
├── PIPELINE.md              # SINGLE SOURCE OF TRUTH (read this for any deep question)
├── AGENT_HANDOFF.md         # this file
├── README.md                # run order, GPU per cell, resume story
├── reproduce.sh             # rebuild all figures from cached JSON, no GPU
├── requirements.txt         # pinned for Colab CUDA 12.x / Python 3.10
├── notebooks/
│   ├── reliable_vqa_master.ipynb   # THE pipeline: one cell per E0..E9 (banners on top of each)
│   └── smoketest.ipynb             # CPU-only offline test of all src/ modules (run FIRST)
├── src/
│   ├── config.py            # all paths, seeds, flags, BACKBONES, RUN_E9, GROUNDER
│   ├── env.py               # mount Drive, stage zips, seed, cal/rep split, frozen-knob assert
│   ├── data_assembly.py     # join VQA+Quality → master.parquet (FIELD_MAP_* adapt to E0 audit)
│   ├── features.py          # CLIP/DINOv2/MobileNet extraction, shard+cache+resume
│   ├── heads.py             # LinearHead, MLPHead, JointHead
│   ├── train_eval.py        # training (torch.optim.AdamW!), metrics, threshold selection
│   ├── stats.py             # multi_seed, bootstrap_ci, paired_bootstrap_delta, BH-FDR, DeLong
│   ├── calibration.py       # temperature scaling, ECE, defect-aware calibration
│   ├── selective.py         # risk-coverage, AURC (NumPy 2.x-safe trapz shim), gating policies
│   ├── vqa_confidence.py    # frozen ViLT harvest + VizWiz VQA accuracy min(#/3,1)
│   ├── actionable.py        # ARR / FRR + defect→action map
│   ├── grounding.py         # Phase 2: LA-3B / Qwen2.5-VL behind one ground() interface
│   ├── figures.py           # F1–F10, one function each, PDF+PNG
│   └── resultlog.py         # versioned JSON + manifest.jsonl + RESULTS.md
├── results/E0..E9/ figures/ manifest.jsonl RESULTS.md
└── artifacts/               # .gitkeep; .npy/.pt gitignored (too large — live on Drive/local)
```

---

## 5. Non-negotiable engineering rules (enforced in code)

1. **Extract features once**, cache float16 `.npy` + parquet index; later cells load cache.
   Re-running E2 is a no-op if cache exists (`FORCE_RERUN` overrides).
2. **Stage data to `/content/local/` and unzip there** before any loop — never stream small
   files from Drive.
3. **Mixed precision** (`torch.autocast`) on GPU; inference cells use `torch.inference_mode()`.
4. **Real minibatch `DataLoader`** (`num_workers`, `pin_memory`, `persistent_workers`) — never
   the old full-batch single-step "training."
5. **Idempotent + resumable:** every cell checks its output artifact and skips; E2/E6/E9
   checkpoint in shards and resume.
6. **Everything logged:** every experiment ends with `resultlog.log_run(...)` → JSON +
   `manifest.jsonl` line (git hash, GPU, seed, versions, metrics) + `RESULTS.md` block.
   **If it wasn't logged, it didn't happen.**
7. **Frozen-knob rule:** temperature `T` and ALL thresholds `τ` are selected **only on the `cal`
   split**, frozen, then applied to `rep`. `env.assert_no_rep_leakage()` raises if a
   threshold/temperature function is ever called with `rep`/`test`.
8. **Print GPU** at cell top; warn (don't crash) on mismatch.

### Validation protocol (§4.5 — what makes it publishable)
- **Splits:** train (fit weights) · **cal** (~30% of val, the ONLY place τ/T are chosen) ·
  **rep** (remaining val, every reported number) · test (eval-only, hidden VQA answers).
- **5 seeds** `[0,1,2,3,4]` → mean±std; **bootstrap CI** `N_BOOT=2000` on report samples.
- **Every "A beats B" = paired-bootstrap p-value.** AURC-delta is the E7 headline test;
  ΔAUROC + DeLong is the E9 RQ3a test. **BH-FDR** across per-defect tests.
- **Metrics:** triage → AUROC/AUPRC lead + 2×2 CM; diagnosis → per-defect AUROC/AUPRC + mAP +
  **one-vs-rest 2×2 per defect** (NEVER a 7×7) + a **co-occurrence heatmap**; selective → AURC +
  risk–coverage + ECE + reliability diagram. **Never report bare accuracy as a headline.**
- **Baselines:** majority-class, per-class base-rate, random-confidence selective, linear-probe,
  plain-BCE.
- **Qualitative** sampled by rule with `QUAL_SEED=7` (incl. the high-confidence-WRONG danger panel).

### Correctness landmines (do not repeat old mistakes)
- `AdamW` from **`torch.optim`**, not `transformers` (that import crashed the old run).
- Intent/keyword labels **banned** — all labels from real VizWiz annotations.
- VQA confidence uses **ViLT** (`dandelin/vilt-b32-finetuned-vqa`, discriminative → clean
  softmax for calibration), not a generative model.
- VizWiz VQA accuracy = `min(#matches/3, 1)` over the 10 answers.

---

## 6. How the user runs it in Colab (the actual workflow)

The user runs **cell-by-cell, one at a time, in Google Colab Pro+**, opening the notebooks
**from GitHub** (so any fix pushed to `main` reaches them after a runtime restart + reopen).
They are NOT running locally and NOT headless — every cell prints a human-readable summary.

### Step 0 — Smoke test FIRST (cheap, CPU)
Open **`notebooks/smoketest.ipynb`** on a plain **CPU** runtime → Run All.
It exercises every `src/` module with synthetic data + a monkeypatched backbone (no GPU, no
Drive, no dataset). 26 checks, prints `[PASS]`/`[FAIL]`, final cell asserts all passed.
This catches Python/logic errors **before** spending any compute units.
→ If a check fails, the user pastes the traceback; we fix the `src/` module, push, they re-pull.

### Step 1 — Core pipeline (E0→E8)
Open **`notebooks/reliable_vqa_master.ipynb`**. Each cell has a banner stating its runtime.
Run in order, switching runtime per banner:
- **CPU + High-RAM:** E0, E1, E5, E7, E8
- **L4:** E2, E6
- **T4:** E3, E4

**Resume story:** re-run a crashed cell — idempotency guards detect partial work and resume
from the last shard. `FORCE_RERUN=True` in `config.py` forces recompute.
After E8, commit `results/` back to the repo.

### Step 2 — Phase 2 (E9), only after E0–E8 committed
Flip `RUN_E9=True` in `config.py`, switch to **L4**, run the E9 cell. If LA-3B misbehaves,
set `GROUNDER="qwen25vl"` — the harvest loop is model-agnostic.

### Reproduce figures without GPU
`bash reproduce.sh` regenerates F1–F10 from cached JSON.

---

## 7. Defaults & open decisions already made

- **Backbones:** `BACKBONES = ["clip", "mobilenet"]` active by default (saves E2 budget);
  DINOv2 loader implemented but opt-in (add `"dinov2"` to the list for the full 3-backbone table).
- **Phase 2 (E9):** fully scaffolded but **gated** (`RUN_E9=False`). Zero cost to have ready.
- **Grounder:** primary `nvidia/LocateAnything-3B`; fallback `Qwen2.5-VL-3B-Instruct`; both
  behind `ground(image, phrase, ...)`; selected by `config.GROUNDER`.
- **Data scope:** train on VizWiz train; carve stratified **cal (~30%)** from val; report on the
  **rep** remainder; test is eval-only.
- **Captions stretch (C-stretch):** held for future work (not in v1).

---

## 8. What has been done so far (state as of last commit)

1. **Full repo scaffolded and pushed** to `main`: all 14 `src/` modules, master notebook
   (E0–E9, one cell each, runtime banners), `requirements.txt` (pinned), `README.md`,
   `reproduce.sh`, `PIPELINE.md` (committed for provenance), results/ tree with `.gitkeep`.
2. **Smoke-test notebook added** (`notebooks/smoketest.ipynb`) — 26 checks across all modules,
   verified to run cleanly end-to-end on CPU locally.
3. **Two real bugs caught by the smoke test and fixed:**
   - `selective.aurc` used `np.trapz`, **removed in NumPy 2.x** (Colab ships 2.x) →
     replaced with a version-safe `np.trapezoid`/`np.trapz` shim.
   - Non-ASCII characters in `print()` statements → replaced with ASCII to avoid
     locale-dependent `UnicodeEncodeError`.
4. **Runtime banners** added to every master-notebook cell.

### NOT yet done / next actions for the user
- Run the **smoke test** in Colab (Step 0) and confirm all pass in the real environment.
- Run **E0** and read the printed schema. **If VizWiz JSON field names differ from the defaults,
  update `FIELD_MAP_VQA` / `FIELD_MAP_QUALITY` in `src/data_assembly.py`.** The `RAW_ZIPS` paths
  in `config.py` may also need editing to match the user's Drive layout.
- Then run E1→E8, commit `results/`, and only then consider E9.

---

## 9. How a new chat should help

- **For deep questions on methodology/metrics/validation:** read [`PIPELINE.md`](PIPELINE.md)
  (esp. §4 experiment specs and §4.5 validation protocol).
- **When the user pastes a Colab error:** identify the failing cell/module, fix the `src/` file
  (not the notebook glue unless it's genuinely notebook-level), keep the fix minimal, then
  **re-run `smoketest.ipynb` logic** before pushing. The user re-pulls from GitHub.
- **Never** introduce heuristic labels, `transformers.AdamW`, bare-accuracy headlines, a 7×7
  defect confusion matrix, or threshold selection on `rep`/`test`. These are the exact things
  that would sink the paper or trip the assertions.
- **Preserve idempotency/caching/logging** in any new cell — every experiment must end with
  `resultlog.log_run(...)`.

**Repo:** https://github.com/meteorboyF/VQA-paper.git · branch `main` · Colab Pro+.
