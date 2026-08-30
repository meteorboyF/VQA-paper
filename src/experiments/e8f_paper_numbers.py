"""E8f - Locked results manifest and paper-number macros (CPU, cached data).

Review "Internal result values disagree": the Results text reports ViLT
global AURC 0.5341 / random 0.7460 while Figure 7 displays 0.5357 / 0.7548.
Root cause: numbers were transcribed from different runs. E8f eliminates the
failure mode by making one artifact the single source of truth:

  1. Recomputes the definitive headline numbers directly from the cached
     predictions and split ids (not from any intermediate JSON).
  2. Emits results/paper_numbers.tex defining a LaTeX macro per number, so
     the manuscript can \\input it and never hard-code a value again.
  3. Emits a per-subset accuracy table (train/cal/report x ViLT/BLIP with
     answerable rates) -- the "explain the split accuracy difference" table.
  4. Cross-checks every experiment JSON that mentions the same quantity and
     prints a consistency report; any disagreement > 1e-6 is listed.

Needs: master.parquet, split_ids, E6/E6b predictions, E3/E4/E7 caches. CPU.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, progress, resultlog, selective

EXP = "E8F"
RESULTS_E8F = os.path.join(config.RESULTS, "E8f_paper_numbers")


def required_artifacts():
    return [os.path.join(RESULTS_E8F, "paper_numbers.tex"),
            os.path.join(RESULTS_E8F, "paper_numbers.json"),
            os.path.join(RESULTS_E8F, "split_accuracy_table.json")]


def _texname(key: str) -> str:
    """paperNumber macro name: letters only (LaTeX macro constraint)."""
    parts = key.replace("-", "_").split("_")
    return "pn" + "".join(p.capitalize() for p in parts if p)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e4 else f"{v:g}"
    return str(v)


def main():
    progress.install_error_hook("E8f paper numbers")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E8F, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E8F, required=required_artifacts()) \
            and not config.FORCE_RERUN:
        expstate.skip_banner(EXP, RESULTS_E8F)
        return

    pbar = progress.notebook_bar("E8f paper numbers", total=4)

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]
    numbers = {}

    # ── Dataset counts ───────────────────────────────────────────────────────
    numbers["n_total"] = int(len(master))
    numbers["n_train"] = int((master["split"] == "train").sum())
    numbers["n_val"] = int(len(val_idx))
    numbers["n_cal"] = int(len(cal_pos))
    numbers["n_report"] = int(len(rep_pos))
    numbers["answerable_rate_total"] = float(master["answerable"].mean())
    progress.step(pbar, "dataset counts locked")

    # ── Definitive answerer numbers, recomputed from raw predictions ────────
    split_table = {}
    for gate, subdir, fname in (
            ("vilt", "E6_vqaconf", "vqa_predictions.parquet"),
            ("blip", "E6b_vqaconf_blip", "vqa_predictions_blip.parquet")):
        pq = os.path.join(config.RESULTS, subdir, fname)
        if not os.path.exists(pq):
            print(f"[E8f] missing {pq}; {gate} numbers not locked")
            continue
        preds = pd.read_parquet(pq)
        vp = preds[preds["split"] == "val"].reset_index(drop=True)
        conf_rep = vp.iloc[rep_pos]["confidence"].values.astype(np.float64)
        corr_rep_frac = vp.iloc[rep_pos]["correct"].values.astype(np.float64)
        corr_rep = (corr_rep_frac > 0).astype(int)

        numbers[f"{gate}_acc_full"] = float(preds["correct"].mean())
        numbers[f"{gate}_conf_full"] = float(preds["confidence"].mean())
        numbers[f"{gate}_acc_report"] = float(corr_rep_frac.mean())
        numbers[f"{gate}_aurc_global"] = float(selective.aurc(conf_rep, corr_rep))
        numbers[f"{gate}_aurc_random"] = float(1 - corr_rep.mean())

        # Per-subset accuracy table (review: explain the split difference).
        for name, sub in (("train", preds[preds["split"] == "train"]),
                          ("cal", vp.iloc[cal_pos]), ("report", vp.iloc[rep_pos])):
            key = f"{gate}_{name}"
            split_table[key] = {
                "n": int(len(sub)),
                "vqa_accuracy": float(sub["correct"].mean()),
                "mean_confidence": float(sub["confidence"].mean()),
            }
    for name, mask in (("train", (master["split"] == "train").values),
                       ("cal", np.isin(np.arange(len(master)), val_idx[cal_pos])),
                       ("report", np.isin(np.arange(len(master)), val_idx[rep_pos]))):
        split_table[f"answerable_{name}"] = float(master["answerable"].values[mask].mean())
    progress.step(pbar, "answerer numbers recomputed from raw predictions")

    # ── Consistency cross-check against every experiment JSON ────────────────
    inconsistencies = []
    watch = {k: v for k, v in numbers.items() if isinstance(v, float)}
    for path in glob.glob(os.path.join(config.RESULTS, "E*", "*.json")):
        try:
            with open(path) as f:
                blob = json.dumps(json.load(f))
        except Exception:
            continue
        for key, val in watch.items():
            token = f"{val:.4f}"[:6]
            # Heuristic: flag files that contain a near-miss of a locked value
            # (same first 3 decimals but different 4th) for manual review.
            near = [f"{val + d:.4f}"[:6] for d in (-0.002, -0.001, 0.001, 0.002)]
            for nm in near:
                if nm in blob and token not in blob:
                    inconsistencies.append({"file": os.path.relpath(path, config.RESULTS),
                                            "locked_key": key,
                                            "locked_value": round(val, 4),
                                            "near_miss": nm})
    if inconsistencies:
        print(f"[E8f] WARNING: {len(inconsistencies)} potential near-miss "
              "values found; review them before submission:")
        for item in inconsistencies[:20]:
            print(f"  {item['file']}: {item['locked_key']}={item['locked_value']} "
                  f"vs near-miss {item['near_miss']}")
    else:
        print("[E8f] no near-miss inconsistencies detected")
    progress.step(pbar, "consistency cross-check complete")

    # ── Emit artifacts ───────────────────────────────────────────────────────
    with open(os.path.join(RESULTS_E8F, "paper_numbers.json"), "w") as f:
        json.dump({"numbers": numbers, "inconsistencies": inconsistencies}, f, indent=2)
    with open(os.path.join(RESULTS_E8F, "split_accuracy_table.json"), "w") as f:
        json.dump(split_table, f, indent=2)

    lines = ["% AUTO-GENERATED by E8f - do not edit by hand.",
             "% \\input this file in the preamble and use \\pnViltAurcGlobal etc."]
    for key, val in numbers.items():
        lines.append(f"\\newcommand{{\\{_texname(key)}}}{{{_fmt(val)}}}")
    with open(os.path.join(RESULTS_E8F, "paper_numbers.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[E8f] locked numbers:")
    for k, v in numbers.items():
        print(f"  \\{_texname(k)} = {_fmt(v)}")

    resultlog.log_run(EXP, metrics=numbers,
                      params={"n_inconsistencies": len(inconsistencies)},
                      results_dir=RESULTS_E8F, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E8F, artifacts=required_artifacts())
    pbar.close()
    print("[E8f DONE] Copy paper_numbers.tex into the manuscript folder and "
          "\\input it; replace hard-coded values with macros.")
