#!/usr/bin/env bash
# reproduce.sh — Regenerate ALL tables and figures from cached metrics + logits.
# No GPU required. Run after E0–E8 (or E9) are complete.
# Usage: bash reproduce.sh [--fig-dir results/figures]

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

FIG_DIR="${1:-results/figures}"

echo "=== Reliable Assistive VQA — Reproduce ==="
echo "Repo root : $REPO_ROOT"
echo "Figure dir: $FIG_DIR"
echo ""

python - <<'PYEOF'
import sys, os, json
sys.path.insert(0, os.getcwd())

from src import config, figures

# Override figure dir to the one passed from CLI (or default)
import sys
fig_dir = sys.argv[1] if len(sys.argv) > 1 else "results/figures"
figures.set_fig_dir(fig_dir)

saved = []

# F1: pipeline schematic (no data needed)
saved.append(figures.f1_pipeline_schematic())

# F2: co-occurrence heatmap
p = os.path.join(config.RESULTS_E1, "label_stats.json")
if os.path.exists(p):
    saved.append(figures.f2_cooccurrence(p))

# F3: per-defect AUROC/AUPRC by backbone
e4_by_bb = {}
for bb in config.BACKBONES:
    p = os.path.join(config.RESULTS_E4, f"per_defect_auroc_{bb}.json")
    if os.path.exists(p):
        with open(p) as f: e4_by_bb[bb] = json.load(f)
if e4_by_bb:
    saved.append(figures.f3_per_defect_auroc(e4_by_bb))

# F4 + F5: calibration + risk-coverage
for bb in config.BACKBONES:
    p = os.path.join(config.RESULTS_E7, f"aurc_comparison_{bb}.json")
    if os.path.exists(p):
        with open(p) as f: e7 = json.load(f)
        # F4
        calib_tmp = os.path.join(config.RESULTS_E7, "calib_diag.json")
        with open(calib_tmp, "w") as f:
            json.dump({"raw": e7.get("reliability",{}).get("raw",{}),
                       "temp": e7.get("reliability",{}).get("temp",{})}, f)
        saved.append(figures.f4_reliability_diagram(calib_tmp))
        # F5
        rc_tmp = os.path.join(config.RESULTS_E7, "rc_data.json")
        with open(rc_tmp, "w") as f:
            json.dump(e7.get("risk_coverage", {}), f)
        saved.append(figures.f5_risk_coverage(rc_tmp))
        break

# F6: ARR/FRR
for bb in config.BACKBONES:
    p = os.path.join(config.RESULTS_E5, f"arr_frr_{bb}.json")
    if os.path.exists(p):
        saved.append(figures.f6_arr_frr(p))
        break

# F7: backbone comparison
e3_by_bb = {}
for bb in config.BACKBONES:
    p = os.path.join(config.RESULTS_E3, f"metrics_{bb}.json")
    if os.path.exists(p):
        with open(p) as f: e3_by_bb[bb] = json.load(f)
combined = {bb: {**e3_by_bb.get(bb,{}), **e4_by_bb.get(bb,{})} for bb in config.BACKBONES}
if combined:
    saved.append(figures.f7_backbone_comparison(combined))

# F8: ROC panels
from src.data_assembly import QUALITY_FLAWS
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
saved.append(figures.f8_roc_panels(
    triage_roc_data={bb: e3_by_bb.get(bb,{}) for bb in config.BACKBONES},
    defect_roc_data={d: {} for d in DEFECT_NAMES},
))

# F10: groundability (Phase 2, if E9 ran)
p = os.path.join(config.RESULTS_E9, "triage_delta.json")
if os.path.exists(p) and config.RUN_E9:
    saved.append(figures.f10_groundability(p))

print(f"\nReproduced {len(saved)} figures:")
for s in saved:
    print(f"  {s}")
PYEOF

echo ""
echo "Done. All figures written to $FIG_DIR"
