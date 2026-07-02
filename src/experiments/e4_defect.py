"""E4 - Multi-label defect diagnosis head (any GPU, minutes)."""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, heads, progress, resultlog, train_eval
from src.data_assembly import QUALITY_FLAWS

EXP = "E4"
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
N_DEFECTS = len(DEFECT_NAMES)


def required_artifacts():
    arts = []
    for bb in config.BACKBONES:
        arts.append(os.path.join(config.RESULTS_E4, f"per_defect_auroc_{bb}.json"))
        arts.append(os.path.join(config.ARTIFACTS, f"defect_{bb}.pt"))
        arts.append(os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy"))
    return arts


def main():
    progress.install_error_hook("E4 defect head")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()

    if expstate.is_done(EXP, config.RESULTS_E4, required=required_artifacts()):
        expstate.skip_banner(EXP, config.RESULTS_E4)
        return

    env.check_gpu(EXP)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pbar = progress.notebook_bar("E4 defect head", total=3 + len(config.BACKBONES))
    progress.step(pbar, f"Environment checked: device={device}")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_mask = master["split"] == "val"
    val_idx = np.where(val_mask)[0]
    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]
    progress.dataframe_summary("master", master)
    progress.step(pbar, "master and cal/rep split loaded")

    all_results = {}
    for bb in config.BACKBONES:
        emb_path = os.path.join(config.ARTIFACTS, f"emb_{bb}.npy")
        assert os.path.exists(emb_path), f"Run E2 first! Missing: {emb_path}"
        print(f"\n[E4] backbone={bb}")

        out_metrics = os.path.join(config.RESULTS_E4, f"per_defect_auroc_{bb}.json")
        out_model = os.path.join(config.ARTIFACTS, f"defect_{bb}.pt")
        out_logits = os.path.join(config.ARTIFACTS, f"defect_logits_{bb}.npy")

        if (os.path.exists(out_metrics) and os.path.exists(out_model)
                and os.path.exists(out_logits) and not config.FORCE_RERUN):
            print(f"  cache hit: {out_metrics}")
            with open(out_metrics) as f:
                all_results[bb] = json.load(f)
            progress.step(pbar, f"{bb} cache reused")
            continue

        emb = np.load(emb_path).astype(np.float32)
        Y = master[defect_cols].values.astype(np.float32)

        train_mask = (master["split"] == "train").values
        X_train = emb[train_mask]; Y_train = Y[train_mask]
        X_cal = emb[val_idx[cal_pos]]; Y_cal = Y[val_idx[cal_pos]]
        X_rep = emb[val_idx[rep_pos]]; Y_rep = Y[val_idx[rep_pos]]
        dim = X_train.shape[1]

        per_seed_logits_val = []
        per_seed_logits_rep = []
        per_seed_thresholds = []
        per_seed_metrics = []

        for seed in config.SEEDS:
            env.seed_everything(seed)
            model = heads.MLPHead(dim, N_DEFECTS, multilabel=True)
            model, _ = train_eval.train_head(model, X_train, Y_train, X_cal, Y_cal,
                                             seed=seed, device=device,
                                             loss_variant="pos_weight")
            with torch.inference_mode():
                lc = model(torch.tensor(X_cal, dtype=torch.float32).to(device)).float().cpu().numpy()
                lr = model(torch.tensor(X_rep, dtype=torch.float32).to(device)).float().cpu().numpy()

            # Per-defect thresholds on cal only
            thresholds = np.zeros(N_DEFECTS)
            for d in range(N_DEFECTS):
                thresholds[d] = train_eval.find_threshold(
                    Y_cal[:, d], lc[:, d], split_name="cal")

            metrics = train_eval.evaluate_multilabel(Y_rep, lr, thresholds, DEFECT_NAMES)
            logits_val = np.zeros((len(val_idx), N_DEFECTS), dtype=np.float32)
            logits_val[cal_pos] = lc
            logits_val[rep_pos] = lr
            per_seed_logits_val.append(logits_val)
            per_seed_logits_rep.append(lr)
            per_seed_thresholds.append(thresholds)
            per_seed_metrics.append(metrics)

        mean_logits_val = np.mean(per_seed_logits_val, axis=0)
        mean_logits_rep = np.mean(per_seed_logits_rep, axis=0)
        mean_thresholds = np.mean(per_seed_thresholds, axis=0)

        def agg(key, subkey=None):
            if subkey:
                vals = [m[key][subkey] for m in per_seed_metrics]
            else:
                vals = [m[key] for m in per_seed_metrics]
            arr = np.array([v for v in vals
                            if not (isinstance(v, float) and np.isnan(v))], float)
            return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                    "seeds": arr.tolist()}

        result = {
            "backbone": bb,
            "macro_F1": agg("macro_F1"),
            "micro_F1": agg("micro_F1"),
            "mAP": agg("mAP"),
            "per_defect_auroc": {d: agg("per_defect_auroc", d) for d in DEFECT_NAMES},
            "per_defect_auprc": {d: agg("per_defect_auprc", d) for d in DEFECT_NAMES},
            "per_defect_f1": {d: agg("per_defect_f1", d) for d in DEFECT_NAMES},
        }

        # BH-FDR placeholder across per-defect tests (updated in E8 ablations)
        from src.stats import benjamini_hochberg
        result["bh_fdr_reject"] = benjamini_hochberg([0.05] * N_DEFECTS).tolist()

        np.save(out_logits, mean_logits_val)
        np.save(out_logits.replace(".npy", "_rep.npy"), mean_logits_rep)
        np.save(out_logits.replace(".npy", "_thresholds.npy"), mean_thresholds)

        env.seed_everything(0)
        model0 = heads.MLPHead(dim, N_DEFECTS, multilabel=True)
        model0, _ = train_eval.train_head(model0, X_train, Y_train, X_cal, Y_cal,
                                          seed=0, device=device,
                                          loss_variant="pos_weight")
        torch.save(model0.state_dict(), out_model)

        os.makedirs(config.RESULTS_E4, exist_ok=True)
        with open(out_metrics, "w") as f:
            json.dump(result, f, indent=2)
        all_results[bb] = result
        print(f"  [E4] {bb}: mAP={result['mAP']['mean']:.4f}+/-{result['mAP']['std']:.4f}  "
              f"macro_F1={result['macro_F1']['mean']:.4f}")
        progress.step(pbar, f"{bb} trained/evaluated")

    resultlog.log_run(EXP, metrics=all_results,
                      params={"backbones": config.BACKBONES, "seeds": config.SEEDS},
                      results_dir=config.RESULTS_E4, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E4, artifacts=required_artifacts())
    progress.step(pbar, "E4 result logged")
    pbar.close()
    print("[E4 DONE]")
