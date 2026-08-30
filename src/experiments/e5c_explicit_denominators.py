"""E5c - Explicit gated-metric denominators and raw counts (CPU, cached data).

Review Critical #3: gated GDMR (0.875 at 90% coverage) is conditional on the
refused-unanswerable subset and can be misread as end-to-end recovery. E5c
computes, for each coverage target and backbone, the four quantities the
revised Section IV defines -- each with raw numerator/denominator counts and
bootstrap CIs:

  conditional guidance precision = |{R & U : match}| / |{R & U : guided}|
  end-to-end match coverage      = |{R & U : match}| / |U|
  gated retake burden            = |{R & A : guided}| / |A|
  refusal precision              = |R & U| / |R|

where R = refused, U = unanswerable, A = answerable, guided = top predicted
defect above cutoff, match = guided and the defect is in the GT set.

Needs: master.parquet, split_ids (E1), E6/E6b predictions, defect logits (E4).
"""
import json
import os

import numpy as np
import pandas as pd

from src import actionable, config, env, expstate, progress, resultlog
from src.data_assembly import QUALITY_FLAWS

EXP = "E5C"
RESULTS_E5C = os.path.join(config.RESULTS, "E5c_explicit_denominators")
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
COVERAGE_TARGETS = (0.9, 0.8, 0.7)
GATES = {"vilt": ("E6_vqaconf", "vqa_predictions.parquet"),
         "blip": ("E6b_vqaconf_blip", "vqa_predictions_blip.parquet")}


def required_artifacts():
    return [os.path.join(RESULTS_E5C, f"explicit_gated_{gate}_{bb}.json")
            for gate in GATES for bb in config.BACKBONES]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _ratio(num, den):
    return float(num / den) if den > 0 else float("nan")


def _explicit_metrics(refused, ans, match, guided):
    """All four quantities with raw counts. Inputs are boolean arrays."""
    U, A = ~ans, ans
    RU, RA = refused & U, refused & A
    out = {
        "conditional_guidance_precision": {
            "num": int((RU & match).sum()), "den": int((RU & guided).sum())},
        "end_to_end_match_coverage": {
            "num": int((RU & match).sum()), "den": int(U.sum())},
        "gated_retake_burden": {
            "num": int((RA & guided).sum()), "den": int(A.sum())},
        "refusal_precision": {
            "num": int(RU.sum()), "den": int(refused.sum())},
    }
    for v in out.values():
        v["value"] = _ratio(v["num"], v["den"])
    return out


def main():
    progress.install_error_hook("E5c explicit denominators")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E5C, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E5C, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E5C)
        return

    pbar = progress.notebook_bar(
        "E5c explicit denominators",
        total=1 + len(GATES) * len(config.BACKBONES))

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]
    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    rep_master = master.iloc[val_idx[rep_pos]]
    gt_defects = rep_master[defect_cols].values
    ans = rep_master["answerable"].values.astype(bool)
    progress.step(pbar, "cached E1 data loaded")

    rng = np.random.default_rng(config.SEED)
    all_results = {}
    for gate, (subdir, fname) in GATES.items():
        pq = os.path.join(config.RESULTS, subdir, fname)
        if not os.path.exists(pq):
            print(f"[E5c] gate '{gate}' predictions missing ({pq}); skipping")
            continue
        preds = pd.read_parquet(pq)
        vp = preds[preds["split"] == "val"].reset_index(drop=True)
        conf_cal = vp.iloc[cal_pos]["confidence"].values.astype(np.float64)
        conf_rep = vp.iloc[rep_pos]["confidence"].values.astype(np.float64)

        env.assert_no_rep_leakage("cal")
        taus = {c: float(np.quantile(conf_cal, 1.0 - c)) for c in COVERAGE_TARGETS}

        for bb in config.BACKBONES:
            out_json = os.path.join(RESULTS_E5C, f"explicit_gated_{gate}_{bb}.json")
            if os.path.exists(out_json) and not config.FORCE_RERUN:
                with open(out_json) as f:
                    all_results[f"{gate}_{bb}"] = json.load(f)
                progress.step(pbar, f"{gate}/{bb} cache reused")
                continue

            logits_path = os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy")
            assert os.path.exists(logits_path), f"Run E4 first! Missing: {logits_path}"
            probs_rep = _sigmoid(np.load(logits_path)[rep_pos])
            top = actionable.top_predicted_defect(probs_rep, DEFECT_NAMES)
            guided = np.array([t is not None for t in top])
            match = np.array([
                bool(t is not None and gt_defects[i, DEFECT_NAMES.index(t)] == 1)
                for i, t in enumerate(top)])

            per_cov = {}
            for c in COVERAGE_TARGETS:
                refused = conf_rep < taus[c]
                m = _explicit_metrics(refused, ans, match, guided)
                # bootstrap CIs over report examples for each ratio
                n = len(ans)
                boots = {k: [] for k in m}
                for _ in range(config.N_BOOT):
                    idx = rng.integers(0, n, n)
                    mb = _explicit_metrics(refused[idx], ans[idx],
                                           match[idx], guided[idx])
                    for k in m:
                        boots[k].append(mb[k]["value"])
                for k in m:
                    lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
                    m[k]["ci95"] = [float(lo), float(hi)]
                m["tau"] = taus[c]
                m["achieved_coverage"] = float(1 - refused.mean())
                per_cov[f"coverage_{c:.1f}"] = m

            result = {"gate": gate, "backbone": bb, "per_coverage": per_cov,
                      "n_report": int(len(ans)),
                      "n_unanswerable": int((~ans).sum()),
                      "n_answerable": int(ans.sum())}
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            all_results[f"{gate}_{bb}"] = result

            print(f"\n[E5c] gate={gate} bb={bb}")
            for key, m in per_cov.items():
                cgp = m["conditional_guidance_precision"]
                e2e = m["end_to_end_match_coverage"]
                print(f"  {key}: cond precision={cgp['value']:.3f} "
                      f"({cgp['num']}/{cgp['den']})  end-to-end="
                      f"{e2e['value']:.3f} ({e2e['num']}/{e2e['den']})")
            progress.step(pbar, f"{gate}/{bb} explicit metrics computed")

    resultlog.log_run(EXP, metrics={k: v["per_coverage"] for k, v in all_results.items()},
                      params={"coverage_targets": list(COVERAGE_TARGETS)},
                      results_dir=RESULTS_E5C, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E5C, artifacts=required_artifacts())
    pbar.close()
    print("[E5c DONE] Explicit denominators ready for Sec. IV-C / VI-D.")
