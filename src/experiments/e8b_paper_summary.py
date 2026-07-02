"""E8b - CPU-only paper summary and reframing notes.

Reads completed E0-E8 plus E7b diagnostics from Drive-backed results and writes
a compact manuscript-facing summary. This is intentionally analysis-only: no
model training, no GPU, no data staging.
"""
import json
import os

from src import config, env, expstate, progress, resultlog

EXP = "E8B"
RESULTS_E8B = os.path.join(config.RESULTS, "paper_summary")
E7B_DIR = os.path.join(config.RESULTS, "E7b_diagnostics")


def required_artifacts():
    return [
        os.path.join(RESULTS_E8B, "paper_summary.json"),
        os.path.join(RESULTS_E8B, "PAPER_SUMMARY.md"),
    ]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _mean_std(obj, key):
    metric = obj.get(key, {})
    return metric.get("mean"), metric.get("std")


def _fmt(x, digits=4):
    if x is None:
        return "NA"
    return f"{float(x):.{digits}f}"


def _metric_path(exp_dir, filename):
    return os.path.join(config.RESULTS, exp_dir, filename)


def _collect():
    backbones = list(config.BACKBONES)

    e0 = _load(_metric_path("E0_audit", "audit.json"))
    e1 = _load(_metric_path("E1_assembly", "label_stats.json"))
    e6_candidates = [
        p for p in os.listdir(_metric_path("E6_vqaconf", ""))
        if p.startswith("E6_") and p.endswith(".json")
    ]
    e6 = _load(_metric_path("E6_vqaconf", sorted(e6_candidates)[-1])) if e6_candidates else {}
    e7b = _load(os.path.join(E7B_DIR, "summary.json"))
    c3 = _load(_metric_path("E8_ablation", "c3_ablation.json"))

    triage = {}
    defect = {}
    actionable = {}
    selective = {}
    ablation = {}
    for bb in backbones:
        e3 = _load(_metric_path("E3_triage", f"metrics_{bb}.json"))
        e4 = _load(_metric_path("E4_defect", f"per_defect_auroc_{bb}.json"))
        e5 = _load(_metric_path("E5_actionable", f"arr_frr_{bb}.json"))
        e7 = _load(_metric_path("E7_calib", f"aurc_comparison_{bb}.json"))

        triage[bb] = {
            "auroc_mean": e3["AUROC"]["mean"],
            "auroc_std": e3["AUROC"]["std"],
            "auprc_mean": e3["AUPRC"]["mean"],
            "f1_mean": e3["F1"]["mean"],
            "baseline_majority_f1": e3["baseline_majority_f1"],
        }
        per_def_auroc = {
            d: v["mean"] for d, v in e4["per_defect_auroc"].items()
        }
        defect[bb] = {
            "map_mean": e4["mAP"]["mean"],
            "map_std": e4["mAP"]["std"],
            "macro_f1_mean": e4["macro_F1"]["mean"],
            "best_defect_auroc": max(per_def_auroc.items(), key=lambda kv: kv[1]),
            "worst_defect_auroc": min(per_def_auroc.items(), key=lambda kv: kv[1]),
        }
        actionable[bb] = {
            "arr": e5["ARR"],
            "arr_ci95": e5["ARR_ci95"],
            "frr": e5["FRR"],
            "frr_ci95": e5["FRR_ci95"],
        }
        selective[bb] = {
            "aurc_global": e7["aurc_global"],
            "aurc_defect": e7["aurc_defect"],
            "delta_aurc": e7["delta_aurc"],
            "delta_aurc_ci": [e7["delta_aurc_ci_lo"], e7["delta_aurc_ci_hi"]],
            "delta_aurc_p": e7["delta_aurc_p"],
            "e7b_predicted_risk": e7b["backbones"][bb]["predicted_defect_risk_model"],
            "e7b_oracle_risk": e7b["backbones"][bb]["oracle_gt_defect_risk_model"],
        }
        ablation[bb] = c3[bb]

    return {
        "backbones": backbones,
        "data": {
            "image_counts_local": e0.get("image_counts_local", {}),
            "annotation_overlap": e0.get("annotation_overlap", {}),
            "split_counts": e1.get("split_counts", {}),
            "total_rows": e1.get("total_rows"),
            "positive_rates": e1.get("positive_rates", {}),
            "vqa_accuracy": e6.get("metrics", {}).get("mean_accuracy"),
            "vqa_confidence": e6.get("metrics", {}).get("mean_confidence"),
        },
        "triage": triage,
        "defect": defect,
        "actionable": actionable,
        "selective": selective,
        "ablation": ablation,
        "recommended_framing": {
            "old_c1_status": "Not supported by E7/E7b: defect-aware selective prediction does not significantly improve AURC over global VQA confidence.",
            "new_thesis": (
                "Frozen VQA confidence remains the strongest risk-ranking signal; "
                "visual defect diagnosis adds interpretable refusal reasons and actionable recovery guidance."
            ),
            "keep": [
                "Answerability triage from frozen visual embeddings.",
                "Multi-label defect diagnosis using real VizWiz-QualityIssues labels.",
                "ARR/FRR actionable recovery evaluation.",
                "Joint-vs-cascade and backbone benchmark.",
            ],
            "deemphasize": [
                "Do not claim defect-conditioned calibration/gating beats global confidence.",
                "Report E7/E7b as an important negative result/diagnostic, not the headline win.",
            ],
            "e9_recommendation": (
                "Run E9 only if the paper needs a new Phase 2 contribution around "
                "groundability/spatial guidance; do not run it merely to rescue C1."
            ),
        },
    }


def _markdown(summary):
    lines = []
    lines.append("# Paper Summary After E0-E8 + E7b")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append(summary["recommended_framing"]["new_thesis"])
    lines.append("")
    lines.append("The original C1 claim is **not supported** by the current run: defect-aware selective prediction does not significantly improve AURC over global VQA confidence.")
    lines.append("")

    data = summary["data"]
    lines.append("## Data Sanity")
    lines.append("")
    lines.append(f"- Rows: {data['total_rows']} total; splits={data['split_counts']}")
    lines.append(f"- Image counts: {data['image_counts_local']}")
    lines.append(f"- VQA mean accuracy={_fmt(data['vqa_accuracy'])}; mean confidence={_fmt(data['vqa_confidence'])}")
    lines.append("")

    lines.append("## Core Metrics")
    lines.append("")
    lines.append("| Backbone | Triage AUROC | Defect mAP | ARR | FRR | Joint-Cascade dAUROC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for bb in summary["backbones"]:
        t = summary["triage"][bb]
        d = summary["defect"][bb]
        a = summary["actionable"][bb]
        c3 = summary["ablation"][bb]
        lines.append(
            f"| {bb} | {_fmt(t['auroc_mean'])} | {_fmt(d['map_mean'])} | "
            f"{_fmt(a['arr'])} | {_fmt(a['frr'])} | "
            f"{_fmt(c3['delta_auroc_joint_minus_cascade'])} |"
        )
    lines.append("")

    lines.append("## Selective Prediction Diagnostic")
    lines.append("")
    lines.append("| Backbone | E7 defect AURC delta | p | E7b predicted-risk improvement | p | E7b oracle improvement | p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for bb in summary["backbones"]:
        s = summary["selective"][bb]
        pred = s["e7b_predicted_risk"]
        oracle = s["e7b_oracle_risk"]
        lines.append(
            f"| {bb} | {_fmt(s['delta_aurc'], 6)} | {_fmt(s['delta_aurc_p'], 3)} | "
            f"{_fmt(pred['improvement_vs_global'], 6)} | {_fmt(pred['p'], 3)} | "
            f"{_fmt(oracle['improvement_vs_global'], 6)} | {_fmt(oracle['p'], 3)} |"
        )
    lines.append("")
    lines.append("Interpretation: positive improvement means lower AURC than global confidence. All predicted-defect improvements are statistically unsupported; oracle GT defects are worse than global confidence in this setup.")
    lines.append("")

    lines.append("## Recommended Contribution Reframe")
    lines.append("")
    lines.append("Keep:")
    for item in summary["recommended_framing"]["keep"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Deemphasize:")
    for item in summary["recommended_framing"]["deemphasize"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("E9 decision:")
    lines.append(f"- {summary['recommended_framing']['e9_recommendation']}")
    lines.append("")
    return "\n".join(lines)


def main():
    progress.install_error_hook("E8b paper summary")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E8B, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E8B, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E8B)
        return

    pbar = progress.notebook_bar("E8b paper summary", total=4)
    progress.step(pbar, "Environment checked: CPU-only summary")
    summary = _collect()
    progress.step(pbar, "E0-E8/E7b JSON loaded")

    json_path = os.path.join(RESULTS_E8B, "paper_summary.json")
    md_path = os.path.join(RESULTS_E8B, "PAPER_SUMMARY.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(md_path, "w") as f:
        f.write(_markdown(summary))
    progress.step(pbar, "paper summary written")

    resultlog.log_run(EXP, metrics=summary["recommended_framing"],
                      params={"backbones": config.BACKBONES},
                      results_dir=RESULTS_E8B, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E8B, artifacts=required_artifacts())
    progress.step(pbar, "E8b result logged")
    pbar.close()
    print(f"[E8b DONE] {md_path}")
