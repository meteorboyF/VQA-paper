# Manuscript Draft

This folder contains a LaTeX base draft for the revised paper framing.

## Structure

- `main.tex`: top-level manuscript file.
- `sections/`: one file per manuscript section.
- `references.bib`: separate bibliography file.
- `figures/`: place exported paper figures here if you want local LaTeX builds.
- `tables/`: optional place for generated tables.

## Build

From this folder:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

If you use Overleaf, upload the whole `manuscript/` folder and set `main.tex`
as the root document.

## Deep Research Workflow

Use each file in `sections/` as the base prompt/context for improving that
section. Keep the core framing intact:

> Confidence remains strongest for risk ranking; defect diagnosis supports
> interpretable refusal and actionable recovery.

Avoid reintroducing the unsupported claim that defect-aware calibration or
gating beats global VQA confidence.

## Citation Note

Most bibliography entries are ready as base citations. The entry
`selectivevq2024` is intentionally marked "Authors to verify" because the exact
metadata should be checked before submission.
