"""
Paper figures F1–F10 (§4.8).

Each function saves PDF+PNG to results/figures/ and returns the saved path.
reproduce.sh calls all of them from cached metrics JSONs — no GPU required.

Palette is consistent across all figures. All data figures include error bars
or CI bands per §4.5.3.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PALETTE = {
    "clip":      "#2196F3",   # blue
    "dinov2":    "#4CAF50",   # green
    "mobilenet": "#FF9800",   # orange
    "global":    "#9C27B0",   # purple
    "defect":    "#F44336",   # red
    "random":    "#9E9E9E",   # grey
    "positive":  "#4CAF50",
    "negative":  "#F44336",
}
DEFECT_COLORS = {
    "blur":        "#1565C0",
    "bright":      "#F9A825",
    "dark":        "#37474F",
    "obstruction": "#880E4F",
    "framing":     "#1B5E20",
    "rotation":    "#BF360C",
    "unrecognizable": "#4A148C",
}

FIG_DIR = None  # set by set_fig_dir() before calling any figure function


def set_fig_dir(d: str) -> None:
    global FIG_DIR
    FIG_DIR = d
    os.makedirs(d, exist_ok=True)


def _save(fig, name: str) -> str:
    assert FIG_DIR is not None, "Call set_fig_dir() first"
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[figures] saved {pdf} + {png}")
    return pdf


# ── F1: Pipeline schematic (drawn, not data) ─────────────────────────────────

def f1_pipeline_schematic() -> str:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    boxes = [
        (0.05, "Image\n(user photo)"),
        (0.22, "Triage\n(answerable?)"),
        (0.40, "Diagnose\n(which defect?)"),
        (0.58, "Calibrate &\nAbstain"),
        (0.76, "Guide\n(corrective action)"),
    ]
    arrow_kw = dict(arrowstyle="->", color="black", lw=1.5)
    for i, (x, label) in enumerate(boxes):
        col = list(PALETTE.values())[i % len(PALETTE)]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.25), 0.14, 0.50, boxstyle="round,pad=0.02",
            facecolor=col, edgecolor="white", alpha=0.85,
        ))
        ax.text(x + 0.07, 0.50, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i + 1][0] + 0.01, 0.50),
                        xytext=(x + 0.14, 0.50),
                        arrowprops=arrow_kw)
    ax.set_title("Reliability Layer Pipeline", fontsize=12, pad=8)
    return _save(fig, "F1_pipeline_schematic")


# ── F2: Defect co-occurrence heatmap + answerable × defect contingency ────────

def f2_cooccurrence(label_stats_path: str) -> str:
    with open(label_stats_path) as f:
        stats = json.load(f)

    defects = ["blur", "bright", "dark", "obstruction", "framing",
               "rotation", "unrecognizable"]
    n = len(defects)
    mat = np.zeros((n, n))
    cooccur = stats.get("cooccurrence", {})
    for i, di in enumerate(defects):
        for j, dj in enumerate(defects):
            key = f"q_{di}xq_{dj}" if j >= i else f"q_{dj}xq_{di}"
            mat[i, j] = cooccur.get(key, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: co-occurrence heatmap
    ax = axes[0]
    im = ax.imshow(mat, vmin=0, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(defects, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(defects, fontsize=8)
    ax.set_title("Defect Co-occurrence (fraction of images)", fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if mat[i, j] < 0.3 else "white")
    fig.colorbar(im, ax=ax)

    # Right: answerable × defect contingency (rates)
    ax2 = axes[1]
    rates = stats.get("positive_rates", {})
    defect_keys = [f"q_{d}" for d in defects]
    rate_vals = [rates.get(k, 0) for k in defect_keys]
    colors = [DEFECT_COLORS.get(d, "#888") for d in defects]
    ax2.barh(defects, rate_vals, color=colors, alpha=0.8)
    ax2.axvline(rates.get("answerable", 0.5), color="black",
                linestyle="--", label=f"Answerable rate={rates.get('answerable',0):.2f}")
    ax2.set_xlabel("Positive rate")
    ax2.set_title("Per-defect positive rate vs. answerability", fontsize=10)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, "F2_cooccurrence")


# ── F3: Per-defect AUROC/AUPRC bar chart across backbones ────────────────────

def f3_per_defect_auroc(metrics_by_backbone: dict) -> str:
    defects = ["blur", "bright", "dark", "obstruction", "framing",
               "rotation", "unrecognizable"]
    backbones = list(metrics_by_backbone.keys())
    n_d = len(defects)
    x = np.arange(n_d)
    width = 0.8 / max(len(backbones), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric_key, ylabel in [
        (axes[0], "per_defect_auroc", "AUROC"),
        (axes[1], "per_defect_auprc", "AUPRC"),
    ]:
        for i, bb in enumerate(backbones):
            m = metrics_by_backbone[bb]
            vals  = [m.get(metric_key, {}).get(d, {}).get("mean", 0) for d in defects]
            stds  = [m.get(metric_key, {}).get(d, {}).get("std",  0) for d in defects]
            offset = (i - len(backbones) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=bb,
                   color=PALETTE.get(bb, "#888"), alpha=0.8,
                   yerr=stds, capsize=3, ecolor="black", error_kw={"lw": 1})
        ax.set_xticks(x)
        ax.set_xticklabels(defects, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="grey", linestyle="--", lw=0.8, label="random")
        ax.legend(fontsize=8)
        ax.set_title(f"Per-defect {ylabel} by backbone (mean±std, 5 seeds)")

    fig.tight_layout()
    return _save(fig, "F3_per_defect_auroc_auprc")


# ── F4: Reliability diagram ───────────────────────────────────────────────────

def f4_reliability_diagram(calib_json_path: str) -> str:
    with open(calib_json_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [
        (axes[0], "raw",  "Before temperature scaling"),
        (axes[1], "temp", "After temperature scaling"),
    ]:
        d = data.get(key, {})
        bin_conf = d.get("bin_conf", [])
        bin_acc  = d.get("bin_acc",  [])
        bin_frac = d.get("bin_frac", [])
        ece_val  = d.get("ece", float("nan"))
        if bin_conf:
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
            valid = [(c, a) for c, a in zip(bin_conf, bin_acc)
                     if not np.isnan(a)]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, "o-", color=PALETTE["global"], lw=2)
            ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE={ece_val:.4f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    fig.suptitle("Reliability Diagrams (VQA Confidence)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "F4_reliability_diagram")


# ── F5: Risk-coverage curves (the money figure) ──────────────────────────────

def f5_risk_coverage(rc_json_path: str) -> str:
    with open(rc_json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    style_map = {
        "random":       dict(color=PALETTE["random"],  linestyle="--", lw=1.5),
        "global_raw":   dict(color=PALETTE["global"],  linestyle="-",  lw=2),
        "global_temp":  dict(color=PALETTE["clip"],    linestyle="-.", lw=2),
        "defect_aware": dict(color=PALETTE["defect"],  linestyle="-",  lw=2.5),
    }

    for key, style in style_map.items():
        curve = data.get(key, {})
        cov   = curve.get("coverage", [])
        risk  = curve.get("risk",     [])
        aurc  = curve.get("aurc",     float("nan"))
        ci_lo = curve.get("ci_lo",    [])
        ci_hi = curve.get("ci_hi",    [])
        if not cov:
            continue
        cov = np.array(cov); risk = np.array(risk)
        label = f"{key.replace('_',' ')} (AURC={aurc:.4f})"
        ax.plot(cov, risk, label=label, **style)
        if ci_lo and ci_hi:
            ax.fill_between(cov, ci_lo, ci_hi, alpha=0.15,
                            color=style["color"])

    # Annotate delta
    ga = data.get("global_raw",   {}).get("aurc", None)
    da = data.get("defect_aware", {}).get("aurc", None)
    p  = data.get("delta_p", None)
    if ga is not None and da is not None:
        delta = ga - da
        p_str = f", p={p:.3f}" if p is not None else ""
        ax.text(0.05, 0.95,
                f"ΔAURC={delta:.4f}{p_str} (global−defect-aware)",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk (1 − accuracy)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9); ax.set_title("Risk-Coverage Curves with CI Bands", fontsize=11)
    fig.tight_layout()
    return _save(fig, "F5_risk_coverage")


# ── F6: ARR and FRR per defect ────────────────────────────────────────────────

def f6_arr_frr(arr_frr_json_path: str) -> str:
    with open(arr_frr_json_path) as f:
        data = json.load(f)

    per_defect = data.get("per_defect", {})
    defects = list(per_defect.keys())
    arr_vals = [per_defect[d].get("ARR", 0) for d in defects]
    overall_arr = data.get("ARR", float("nan"))
    overall_frr = data.get("FRR", float("nan"))
    arr_ci = data.get("ARR_ci95", (0, 0))
    frr_ci = data.get("FRR_ci95", (0, 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-defect ARR bars
    ax = axes[0]
    colors = [DEFECT_COLORS.get(d, "#888") for d in defects]
    ax.bar(defects, arr_vals, color=colors, alpha=0.8)
    ax.axhline(overall_arr, color="black", linestyle="--",
               label=f"Overall ARR={overall_arr:.3f} [{arr_ci[0]:.3f},{arr_ci[1]:.3f}]")
    ax.set_ylim(0, 1.1); ax.set_ylabel("ARR")
    ax.set_xticklabels(defects, rotation=30, ha="right", fontsize=9)
    ax.set_title("Actionable Recovery Rate per Defect")
    ax.legend(fontsize=8)

    # Right: FRR summary with CI
    ax2 = axes[1]
    ax2.barh(["FRR"], [overall_frr], xerr=[[overall_frr - frr_ci[0]],
                                            [frr_ci[1] - overall_frr]],
             capsize=5, color=PALETTE["negative"], alpha=0.8)
    ax2.barh(["ARR (overall)"], [overall_arr],
             xerr=[[overall_arr - arr_ci[0]], [arr_ci[1] - overall_arr]],
             capsize=5, color=PALETTE["positive"], alpha=0.8)
    ax2.set_xlim(0, 1); ax2.set_xlabel("Rate")
    ax2.set_title("Overall ARR vs FRR (with 95% CI)\n"
                  "(ARR↑ is good, FRR↓ is good)", fontsize=9)

    fig.tight_layout()
    return _save(fig, "F6_arr_frr")


# ── F7: Backbone comparison table → figure ───────────────────────────────────

def f7_backbone_comparison(metrics_by_backbone: dict) -> str:
    backbones = list(metrics_by_backbone.keys())
    metric_keys = ["AUROC", "AUPRC", "F1", "balanced_acc", "mAP"]
    n_m = len(metric_keys)
    x = np.arange(n_m)
    width = 0.8 / max(len(backbones), 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, bb in enumerate(backbones):
        m = metrics_by_backbone[bb]
        vals = [m.get(k, {}).get("mean", 0) for k in metric_keys]
        stds = [m.get(k, {}).get("std",  0) for k in metric_keys]
        offset = (i - len(backbones) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=bb,
               color=PALETTE.get(bb, "#888"), alpha=0.8,
               yerr=stds, capsize=3, ecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(metric_keys, fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    ax.set_title("Backbone Comparison — Triage + Defect (mean±std, 5 seeds)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "F7_backbone_comparison")


# ── F8: Triage ROC + per-defect one-vs-rest panels ───────────────────────────

def f8_roc_panels(triage_roc_data: dict, defect_roc_data: dict) -> str:
    defects = list(defect_roc_data.keys())
    n_panels = 1 + len(defects)
    cols = min(4, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).flatten()

    # Triage ROC (first panel)
    ax = axes[0]
    for bb, d in triage_roc_data.items():
        fpr = d.get("fpr", [])
        tpr = d.get("tpr", [])
        auc = d.get("AUROC", {}).get("mean", 0)
        std = d.get("AUROC", {}).get("std", 0)
        if fpr:
            ax.plot(fpr, tpr, label=f"{bb} AUC={auc:.3f}±{std:.3f}",
                    color=PALETTE.get(bb, "#888"), lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("Triage ROC", fontsize=9)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7)

    # Per-defect one-vs-rest panels
    for i, d_name in enumerate(defects):
        ax = axes[i + 1]
        d = defect_roc_data.get(d_name, {})
        fpr = d.get("fpr", [])
        tpr = d.get("tpr", [])
        auc = d.get("AUROC", float("nan"))
        if fpr:
            ax.plot(fpr, tpr,
                    color=DEFECT_COLORS.get(d_name, "#888"), lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{d_name} AUC={auc:.3f}", fontsize=8)
        ax.set_xlabel("FPR", fontsize=7); ax.set_ylabel("TPR", fontsize=7)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle("ROC Curves: Triage (top-left) + Per-defect One-vs-Rest", fontsize=10)
    fig.tight_layout()
    return _save(fig, "F8_roc_panels")


# ── F9: Qualitative grid (sampled by rule, QUAL_SEED=7) ──────────────────────

def f9_qualitative_grid(
    qual_data: list,   # list of {image_path, defect, split_type, label}
    title: str = "Qualitative Examples",
    fig_name: str = "F9_qualitative_grid",
) -> str:
    from PIL import Image
    n = len(qual_data)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No examples", ha="center")
        return _save(fig, fig_name)

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).flatten()

    for i, item in enumerate(qual_data):
        ax = axes[i]
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "load error", ha="center", transform=ax.transAxes)
        ax.set_title(
            f"{item.get('label','?')} [{item.get('defect','?')}]",
            fontsize=7, pad=2,
            color=("green" if "TP" in item.get("split_type","") else
                   "red"   if "FP" in item.get("split_type","") else
                   "orange")
        )
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title}\n(QUAL_SEED=7, sampled by rule — not hand-picked)", fontsize=9)
    fig.tight_layout()
    return _save(fig, fig_name)


# ── F10: Groundability (Phase 2) ─────────────────────────────────────────────

def f10_groundability(triage_delta_json: str, spatial_examples: list = None) -> str:
    with open(triage_delta_json) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: ΔAUROC with CI (appearance-only vs. +groundability)
    ax = axes[0]
    metrics = ["AUROC", "AUPRC"]
    delta_vals  = [data.get(f"delta_{m}", 0)  for m in metrics]
    ci_lo       = [data.get(f"delta_{m}_ci_lo", 0) for m in metrics]
    ci_hi       = [data.get(f"delta_{m}_ci_hi", 0) for m in metrics]
    colors      = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in delta_vals]
    errs_lo     = [v - lo for v, lo in zip(delta_vals, ci_lo)]
    errs_hi     = [hi - v for v, hi in zip(delta_vals, ci_hi)]
    ax.barh(metrics, delta_vals, color=colors, alpha=0.8,
            xerr=[errs_lo, errs_hi], capsize=5)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Δ (appearance+groundability − appearance-only)")
    ax.set_title("Triage ΔAUROC/ΔAUPRC with Groundability Feature\n"
                 "(bootstrap 95% CI, DeLong cross-check)", fontsize=9)

    # Right: sample count info
    ax2 = axes[1]
    ax2.axis("off")
    summary_lines = [
        f"Subsample N = {data.get('subsample_n','?')}",
        f"Grounder: {data.get('grounder','?')}",
        f"ΔAUROC = {data.get('delta_AUROC',0):.4f}  p={data.get('delta_AUROC_p','?')}",
        f"DeLong z = {data.get('delong_z',0):.3f}  p={data.get('delong_p','?')}",
        f"CI width (AUROC) = {data.get('delta_AUROC_ci_hi',0)-data.get('delta_AUROC_ci_lo',0):.4f}",
    ]
    ax2.text(0.05, 0.9, "\n".join(summary_lines), transform=ax2.transAxes,
             fontsize=9, va="top", family="monospace")

    fig.suptitle("Groundability-Aware Triage (Phase 2 — E9)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "F10_groundability")
