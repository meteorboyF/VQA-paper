#!/usr/bin/env python3
"""Regenerate the hand-drawn method figures (paper Fig. 2 and Fig. 3 panels).

Review Section 9 fixes:
  - Fig. 2 (F1_pipeline_schematic): simplified into three large blocks
    (answer+confidence / refusal decision / defect identification+guidance),
    legible at page scale, no icons, fixes the "Recovery matrics" typo.
  - Fig. 3 panels (F10 workflow, F11 policy comparison): larger fonts,
    grayscale-safe (fills distinguish by lightness, arrows by line style,
    never by hue alone), renamed metrics (GDMR/AIRB), no "deployment-
    aligned" wording.

Pure matplotlib; no data needed. Output: PDF into the manuscript figures dir.

    python3 scripts/make_method_figures.py [outdir]
"""
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "IEEE Access template", "figures")

# Grayscale-safe palette: distinguish by lightness, not hue.
FILL_DARK = "#c8c8c8"
FILL_MID = "#e0e0e0"
FILL_LIGHT = "#f2f2f2"
EDGE = "black"


def box(ax, x, y, w, h, text, fill=FILL_LIGHT, fs=11, lw=1.4, ls="-",
        weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=fill, edgecolor=EDGE,
                                linewidth=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, ls="-", lw=1.6, label=None, label_dx=0.0,
          fs=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=lw, linestyle=ls, color="black",
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2, label,
                ha="center", va="center", fontsize=fs,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


def fig2_pipeline():
    """Three large blocks, vertical flow, single-column friendly."""
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.4)
    ax.axis("off")

    box(ax, 0.4, 10.3, 9.2, 1.7,
        "Frozen VQA model\nanswer + confidence score",
        fill=FILL_DARK, fs=12.5, weight="bold")
    box(ax, 0.4, 6.1, 9.2, 2.6,
        "Refusal decision\nconfidence below calibration-selected\n"
        "threshold $\\Rightarrow$ refuse instead of answer",
        fill=FILL_MID, fs=12.5, weight="bold")
    box(ax, 0.4, 0.6, 9.2, 3.6,
        "Defect identification + retake guidance\n"
        "frozen image embedding $\\rightarrow$ defect head\n"
        "(blur, bright, dark, obstruction,\nframing, rotation, unrecognizable)\n"
        "$\\rightarrow$ deterministic retake instruction",
        fill=FILL_LIGHT, fs=12.5, weight="bold")

    arrow(ax, 5.0, 10.3, 5.0, 8.8)
    arrow(ax, 5.0, 6.1, 5.0, 4.35, label="only refused examples",
          label_dx=0.0, fs=10.5)

    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, "F1_pipeline_schematic.pdf")
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def fig3_left_workflow():
    """F10: selective-prediction workflow, bigger fonts, grayscale-safe."""
    fig, ax = plt.subplots(figsize=(5.4, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    box(ax, 0.3, 11.6, 4.4, 1.1, "User image + question", FILL_DARK, fs=10.5)
    box(ax, 5.3, 11.6, 4.4, 1.1, "Frozen VQA model", FILL_DARK, fs=10.5)
    arrow(ax, 4.7, 12.15, 5.3, 12.15)

    box(ax, 0.3, 9.7, 4.4, 1.1, "Answer prediction", FILL_LIGHT, fs=10.5)
    box(ax, 5.3, 9.7, 4.4, 1.1, "Confidence score", FILL_LIGHT, fs=10.5)
    arrow(ax, 6.6, 11.6, 2.5, 10.8)
    arrow(ax, 7.5, 11.6, 7.5, 10.8)

    box(ax, 0.3, 7.6, 9.4, 1.3,
        "Calibration subset: fit temperatures, refusal\n"
        "thresholds, learned risk models (then frozen)", FILL_MID, fs=10.5)
    arrow(ax, 7.5, 9.7, 6.5, 8.9)

    box(ax, 0.3, 5.6, 9.4, 1.2,
        "Rank held-out report subset by\nconfidence or learned risk",
        FILL_MID, fs=10.5)
    arrow(ax, 5.0, 7.6, 5.0, 6.8)

    box(ax, 0.3, 3.5, 4.4, 1.3, "Answered subset\nrisk-coverage, AURC",
        FILL_LIGHT, fs=10.5)
    box(ax, 5.3, 3.5, 4.4, 1.3, "Refused subset", FILL_LIGHT, fs=10.5)
    arrow(ax, 3.2, 5.6, 2.5, 4.8, label="above\nthreshold", label_dx=-1.2,
          fs=9.5)
    arrow(ax, 6.8, 5.6, 7.5, 4.8, label="below\nthreshold", label_dx=1.2,
          fs=9.5)

    box(ax, 5.3, 1.4, 4.4, 1.4,
        "Predicted defect\n$\\rightarrow$ retake guidance\n(refused-only analysis)",
        FILL_DARK, fs=10.5)
    arrow(ax, 7.5, 3.5, 7.5, 2.8)

    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, "F10_selective_prediction_refusal_workflow.pdf")
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def fig3_right_policies():
    """F11: ungated vs refusal-gated policy comparison, GDMR/AIRB names."""
    fig, ax = plt.subplots(figsize=(5.4, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    box(ax, 2.8, 11.5, 4.4, 1.2, "Predicted defect\nprobabilities",
        FILL_DARK, fs=10.5)

    box(ax, 0.3, 8.9, 4.2, 1.2, "Ungated diagnostic\npolicy", FILL_MID, fs=10.5)
    box(ax, 5.5, 8.9, 4.2, 1.2, "Refusal-gated policy\n(confidence gate first)",
        FILL_MID, fs=10.5)
    arrow(ax, 4.0, 11.5, 2.4, 10.1)
    arrow(ax, 6.0, 11.5, 7.6, 10.1)

    box(ax, 0.3, 6.2, 4.2, 1.5,
        "GDMR on all\nunanswerable images\nAIRB on all answerable",
        FILL_LIGHT, fs=10.5)
    box(ax, 5.5, 6.2, 4.2, 1.5,
        "Guidance issued only\nfor refused examples", FILL_LIGHT, fs=10.5)
    arrow(ax, 2.4, 8.9, 2.4, 7.7)
    arrow(ax, 7.6, 8.9, 7.6, 7.7)

    box(ax, 0.3, 3.6, 4.2, 1.5,
        "Diagnostic question:\n“what is wrong\nwith this image?”",
        FILL_LIGHT, fs=10.5, ls="--", lw=1.2)
    box(ax, 5.5, 3.2, 4.2, 1.9,
        "Gated GDMR / gated AIRB\nwith explicit denominators\n"
        "(conditional precision,\nend-to-end coverage)",
        FILL_DARK, fs=10.5)
    arrow(ax, 2.4, 6.2, 2.4, 5.1, ls="--", lw=1.2)
    arrow(ax, 7.6, 6.2, 7.6, 5.1)

    box(ax, 2.8, 0.6, 4.4, 1.4,
        "Offline proxies:\nno user retake is\nevaluated in either policy",
        FILL_LIGHT, fs=10.5, ls=":", lw=1.2)
    arrow(ax, 2.4, 3.6, 4.0, 2.0, ls=":", lw=1.2)
    arrow(ax, 7.6, 3.2, 6.0, 2.0, ls=":", lw=1.2)

    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, "F11_recovery_policy_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig2_pipeline()
    fig3_left_workflow()
    fig3_right_policies()
    print("done")
