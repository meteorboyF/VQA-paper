"""E7 - Calibration + selective prediction (CPU). The C1 headline test:
defect-conditioned gating vs a global threshold, paired-bootstrap AURC delta."""
import json
import os

import numpy as np
import pandas as pd

from src import (calibration, config, env, expstate, progress, resultlog,
                 selective, stats as st)
from src.data_assembly import QUALITY_FLAWS

EXP = "E7"
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
N_DEFECTS = len(DEFECT_NAMES)


def required_artifacts():
    return [os.path.join(config.RESULTS_E7, f"aurc_comparison_{bb}.json")
            for bb in config.BACKBONES]


def main():
    progress.install_error_hook("E7 calibration/selective prediction")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()

    if expstate.is_done(EXP, config.RESULTS_E7, required=required_artifacts()):
        expstate.skip_banner(EXP, config.RESULTS_E7)
        return

    env.check_gpu(EXP)
    pbar = progress.notebook_bar("E7 calibration/selective prediction",
                                 total=4 + len(config.BACKBONES))
    progress.step(pbar, "Environment checked")

    vqa_preds = pd.read_parquet(os.path.join(config.RESULTS_E6, "vqa_predictions.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))

    val_mask = vqa_preds["split"] == "val"
    val_df = vqa_preds[val_mask].reset_index(drop=True)
    cal_df = val_df.iloc[cal_pos]
    rep_df = val_df.iloc[rep_pos]
    progress.dataframe_summary("vqa_predictions", vqa_preds)
    progress.step(pbar, "VQA predictions and cal/rep split loaded")

    confs_cal = cal_df["confidence"].values.astype(np.float64)
    corr_cal = (cal_df["correct"].values > 0).astype(float)
    confs_rep = rep_df["confidence"].values.astype(np.float64)
    corr_rep = (rep_df["correct"].values > 0).astype(float)

    # ── 1. Global temperature scaling (cal only) ──
    logits_cal = np.log(confs_cal.clip(1e-6, 1 - 1e-6) / (1 - confs_cal.clip(1e-6, 1 - 1e-6)))
    T = calibration.temperature_scale(logits_cal, corr_cal.astype(int), split_name="cal")
    print(f"[E7] Temperature T={T:.4f}")

    logits_rep = np.log(confs_rep.clip(1e-6, 1 - 1e-6) / (1 - confs_rep.clip(1e-6, 1 - 1e-6)))
    confs_rep_raw = confs_rep
    confs_rep_temp = calibration.apply_temperature(logits_rep, T)

    ece_raw = calibration.ece(confs_rep_raw, corr_rep)
    ece_temp = calibration.ece(confs_rep_temp, corr_rep)
    print(f"[E7] ECE: raw={ece_raw:.4f}  temp-scaled={ece_temp:.4f}  "
          f"dECE={ece_raw - ece_temp:.4f}")
    progress.step(pbar, "Global temperature scaling completed")

    # ── 2. Defect-aware calibration + AURC comparison per backbone ──
    all_results = {}
    for bb in config.BACKBONES:
        logits_path = os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy")
        if not os.path.exists(logits_path):
            print(f"[E7] Skip {bb} - no defect logits (run E4 first)")
            progress.step(pbar, f"{bb} skipped: missing E4 defect logits")
            continue

        out_path = os.path.join(config.RESULTS_E7, f"aurc_comparison_{bb}.json")
        if os.path.exists(out_path) and not config.FORCE_RERUN:
            print(f"[E7] cache hit: {out_path}")
            with open(out_path) as f:
                all_results[bb] = json.load(f)
            progress.step(pbar, f"{bb} cache reused")
            continue

        defect_logits_val = np.load(logits_path)
        defect_probs_cal = 1 / (1 + np.exp(-defect_logits_val[cal_pos]))
        defect_probs_rep = 1 / (1 + np.exp(-defect_logits_val[rep_pos]))
        defect_ids_cal = defect_probs_cal.argmax(axis=1)   # top predicted defect
        defect_ids_rep = defect_probs_rep.argmax(axis=1)

        defect_scalers = calibration.defect_aware_calibration(
            confs_cal, corr_cal, defect_ids_cal, N_DEFECTS, split_name="cal")
        confs_rep_defect = calibration.apply_defect_aware_calibration(
            confs_rep_raw, defect_ids_rep, defect_scalers)

        ece_defect = calibration.ece(confs_rep_defect, corr_rep)

        # ── 3. Risk-coverage / AURC comparison ──
        aurc_rand = 1 - float(corr_rep.mean())
        aurc_global = selective.aurc(confs_rep_raw, corr_rep)
        aurc_temp = selective.aurc(confs_rep_temp, corr_rep)
        aurc_defect = selective.aurc(confs_rep_defect, corr_rep)

        # Paired bootstrap: defect-aware vs global (THE HEADLINE TEST)
        delta, ci_lo, ci_hi, p = st.paired_bootstrap_delta(
            lambda y, s: selective.aurc(s, y),
            corr_rep,
            confs_rep_raw,       # global policy
            confs_rep_defect,    # defect-aware policy
            n_boot=config.N_BOOT,
        )

        print(f"[E7] {bb}: AURC global={aurc_global:.4f}  defect-aware={aurc_defect:.4f}  "
              f"delta={delta:.4f} [{ci_lo:.4f},{ci_hi:.4f}] p={p:.4f}")
        if ci_lo > 0:
            print("  Defect-aware is BETTER (delta>0, CI does not cross 0) - C1 claim SUPPORTED")
        else:
            print(f"  WARNING: CI crosses 0 - C1 claim may need reframing. "
                  f"CI=[{ci_lo:.4f},{ci_hi:.4f}]")

        rc_data = {}
        for label, confs_x in [
            ("random", np.random.default_rng(42).random(len(corr_rep))),
            ("global_raw", confs_rep_raw),
            ("global_temp", confs_rep_temp),
            ("defect_aware", confs_rep_defect),
        ]:
            rc_data[label] = selective.risk_coverage_for_figure(confs_x, corr_rep, label)
        rc_data["delta_p"] = float(p)

        result = {
            "backbone": bb,
            "T": T,
            "ece_raw": ece_raw, "ece_temp": ece_temp, "ece_defect": ece_defect,
            "aurc_random": aurc_rand,
            "aurc_global": aurc_global,
            "aurc_temp": aurc_temp,
            "aurc_defect": aurc_defect,
            "delta_aurc": delta, "delta_aurc_ci_lo": ci_lo,
            "delta_aurc_ci_hi": ci_hi, "delta_aurc_p": p,
            "risk_coverage": rc_data,
            "reliability": {
                "raw": calibration.ece_diagram_data(confs_rep_raw, corr_rep),
                "temp": calibration.ece_diagram_data(confs_rep_temp, corr_rep),
            },
        }
        os.makedirs(config.RESULTS_E7, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        all_results[bb] = result
        progress.step(pbar, f"{bb} AURC comparison completed")

    resultlog.log_run(EXP, metrics=all_results,
                      params={"backbones": config.BACKBONES, "T": T,
                              "N_BOOT": config.N_BOOT},
                      results_dir=config.RESULTS_E7, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E7, artifacts=required_artifacts())
    progress.step(pbar, "E7 result logged")
    pbar.close()
    print("[E7 DONE]")
