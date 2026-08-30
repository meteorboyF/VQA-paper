# Revision Plan — response to internal review (2026-07-16)

Source: `ML_research_paper_review (1).pdf` (teammate review of `access.pdf`).
Every review point maps to an action below. Status codes:
- **[TEX]** manuscript-only fix (done locally, no new compute)
- **[CODE]** new/changed code in `src/` + notebook, ready to run
- **[COLAB]** requires a Colab GPU/CPU session on the Drive-cached artifacts
- **[AUTHOR]** needs input/approval from a named author
- **[DONE]** completed and committed

## 1. Submission-blocking methodological problems

| # | Issue | Action | Status |
|---|-------|--------|--------|
| 1.1 | Answerability head does not use the question (image-only ≠ pair answerability) | Reframe current heads as **image-level visual sufficiency triage** everywhere in the text; add question-conditioned experiment (image-only linear probe / image-only MLP / question-only / concat / gated fusion / + VQA confidence) as new experiment E10 | TEX done; CODE ready; COLAB pending |
| 1.2 | "ARR" is not recovery — it is diagnostic agreement | Rename ARR → **Guidance–Defect Match Rate (GDMR)**, FRR → **Answerable-Image Retake Burden (AIRB)**; state explicitly both are *offline proxies for potential actionability*; genuine Retake Recovery Rate deferred to user study (Limitations + Future Work) | TEX |
| 1.3 | Gated metrics need explicit denominators | Add the four explicit quantities with formulas: conditional guidance precision, end-to-end unanswerable recovery coverage, answerable retake burden, refusal precision — each with raw numerator/denominator counts and CIs; computed by new E5c | TEX formulas done; COLAB for counts |
| 1.4 | FRR is not strictly a "false" rate | Renamed to burden proxy (AIRB); text spells out that answerable images may contain true defects and retakes may still help | TEX |
| 1.5 | No trivial recovery baselines | New E5d: always-framing, most-prevalent-among-unanswerable, prevalence-sampled, uniform-random, question-independent prior, oracle-defect policies — same GDMR/AIRB protocol | CODE ready; COLAB pending |
| 1.6 | 0.5 defect threshold arbitrary | New E4b: per-label thresholds selected on calibration split (F1 and cost-sensitive objectives) + threshold sweep reported; guidance policy re-evaluated at selected thresholds | CODE ready; COLAB pending |
| 1.7 | No BLV user validation | Cannot be fixed offline. All "recovery"/"actionable" claims weakened to offline-proxy language; explicit user-study protocol added to Future Work; venue strategy notes TACCESS/ASSETS after user study | TEX |
| 1.8 | Only ViLT + BLIP answerers | Findings presented as model-specific case studies (not discriminative-vs-generative in general); harvest code prepared for one modern VLM answerer (E6c) to run when GPU available | TEX done; CODE ready |
| 1.9 | Methods not reproducible | New **Implementation Details** subsection with all architecture/optimization/calibration/bootstrap specifics extracted from `src/`; discloses that answer matching is lowercase/strip exact match, *not* the official VizWiz normalization | TEX |
| 1.10 | "Deployment-aligned" too strong (27.4% answered accuracy at 90% coverage) | Replaced with "refusal-gated offline evaluation"; risk-targeted operating points discussed | TEX |
| 1.11 | Internal AURC disagreement (0.5341 vs 0.5357) | One locked results manifest (E8d) that emits `results/paper_numbers.tex` macros; every prose/table/figure number regenerated from it; discrepancy traced and resolved on rerun | CODE ready; COLAB pending |

## 2. Selective-prediction and calibration

- [TEX] Temperature scaling is monotonic → presented strictly as calibration diagnostic (NLL/Brier/ECE), never as an AURC competitor.
- [TEX] BLIP framed as "weaker correctness-ranking signal on the evaluated report split", not "less calibrated"; BLIP calibration metrics to be added from E7d rerun.
- [TEX] Oracle interpretation hedged: continuous-score gain may reflect embedding-derived difficulty, label uncertainty, prevalence, answerability correlation — not necessarily "richer defect description"; control experiments listed (E7e: permuted-defect targets, no-defect-supervision embedding features, identical-capacity risk models). [CODE ready]

## 3. Missing experimental detail
- [TEX] Implementation Details subsection (see 1.9). Includes: model IDs, preprocessing, dims, MLP spec, losses/pos-weighting, AdamW/LR/decay/epochs/patience/batch, seeds, threshold objective, BLIP greedy decoding + max_new_tokens=10 + length-normalized sequence probability equation, risk-model spec, LBFGS temperature fitting, ECE 15 equal-width bins, 2000 bootstrap resamples (unstratified, seed 42), paired-bootstrap p construction, BH families, cal/report split construction, leakage guards, hardware.

## 4. Baselines
- [CODE ready; COLAB pending] Answerability: majority, question-only, image-only linear probe, CLIP image–question cosine similarity, VQA confidence alone, confidence+question features (E10).
- [CODE ready] Defect: per-label prevalence, linear probe (exists as head option), with/without pos-weighting (E4b).
- [CODE ready] Selective prediction: entropy, top-two margin, answer-length & token stats for BLIP (E7f).
- [CODE ready] Guidance: trivial policies (E5d, see 1.5).

## 5. Results/reporting consistency
- [COLAB] Single manifest → tex macros (1.11); five-seed aggregation policy stated per metric; split-accuracy difference table (train/cal/report accuracy + answerable proportion) added (E8e).
- [TEX] "All numbers on report subset" → "all final model-performance comparisons and statistical tests are on the held-out report subset"; joint-vs-cascade → "small observed gains" + parameter-matching note.

## 6. Claims weakened  — applied globally [TEX]
Title → *Defect-Aware Refusal and Retake Guidance for Assistive Visual Question Answering*. All Section-6 replacement phrasings applied (offline guidance–defect agreement; refusal-gated offline recovery proxy; modular offline evaluation; identifies an annotated visual defect that may have contributed to failure; etc.).

## 7. Author biographies [AUTHOR]
Duplicated/incorrect bios replaced with corrected drafts (Himu = Lecturer, Jahangir = Assistant Professor, per UIU faculty pages); pronouns fixed; **every author must personally approve their bio before submission**; ORCIDs to be collected.

## 8. Repository hygiene
- [DONE] Manuscript synced into repo; single source of truth.
- [TEX/README] README claims matched to manuscript; "Where" dropped from title line; gating-beats-global claim removed.
- [CODE] `actionable.py` docstring fixed ("answerable images", not "clean photo").
- Pending: pinned lockfile, split-index publication, release tag + DOI at submission time.

## 9. Figures & accessibility
- Fig. 2 (pipeline): simplified three-block redraw. Fig. 3 (workflows): larger fonts, grayscale-safe. Fig. 9: moved before Discussion; larger cells. [TEX + figure regeneration]
- Alt text/captions carry the result; tagged-PDF pass at camera-ready.

## 10. Editorial
All applied [TEX]: DINOv2/ViLT casing; refilm→retake; acronyms defined at first use; "Selective risk among answered examples"; repetition reduced; full 7-row defect→guidance table added; below-threshold and conflicting-defect behavior explained; data-ethics statement added; qualitative grid checked for private content; reference audit pass at submission.

## Colab run queue (single session, in order)
1. E4b per-label thresholds → E5c explicit gated counts → E5d guidance baselines (CPU, minutes each)
2. E10 question-conditioned triage (GPU: text-feature extraction + head training)
3. E7e oracle controls + E7f selective baselines (CPU)
4. E8d locked manifest + paper_numbers.tex + E8e split table (CPU)
5. Optional: E6c modern-VLM harvest (GPU, longest)
