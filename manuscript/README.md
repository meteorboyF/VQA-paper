# Manuscript Draft (superseded)

> **The submission-ready paper now lives in `../IEEE Access template/`**
> (`access.tex` + `references.bib` + `figures/`), written in the official
> IEEE Access class with the review fixes applied. The figure PDFs were
> MOVED there, so `main.tex` in this folder no longer builds. This folder
> is kept as the section-by-section drafting workspace.

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

The `selectivevq2024` placeholder was resolved in the IEEE Access version:
arXiv:2406.00980 is "Selectively Answering Visual Questions" by Eisenschlos,
Maina, Ivetta, and Benotti (2024), cited there as `eisenschlos2024selectively`.
