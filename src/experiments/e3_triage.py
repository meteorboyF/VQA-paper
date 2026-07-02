"""E3 - Binary answerability triage head (any GPU, minutes)."""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, heads, progress, resultlog, train_eval

EXP = "E3"


def required_artifacts():
    arts = []
    for bb in config.BACKBONES:
        arts.append(os.path.join(config.RESULTS_E3, f"metrics_{bb}.json"))
        arts.append(os.path.join(config.ARTIFACTS, f"triage_{bb}.pt"))
        arts.append(os.path.join(config.ARTIFACTS, f"triage_logits_{bb}.npy"))
    return arts


def main():
    progress.install_error_hook("E3 triage head")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()

    if expstate.is_done(EXP, config.RESULTS_E3, required=required_artifacts()):
        expstate.skip_banner(EXP, config.RESULTS_E3)
        return

    env.check_gpu(EXP)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pbar = progress.notebook_bar("E3 triage head", total=3 + len(config.BACKBONES))
    progress.step(pbar, f"Environment checked: device={device}")

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_mask = master["split"] == "val"
    val_idx = np.where(val_mask)[0]
    progress.dataframe_summary("master", master)
    progress.step(pbar, "master and cal/rep split loaded")

    from sklearn.metrics import f1_score
    from src.stats import bootstrap_ci
    from sklearn.metrics import roc_auc_score as _roc

    all_results = {}
    for bb in config.BACKBONES:
        emb_path = os.path.join(config.ARTIFACTS, f"emb_{bb}.npy")
        assert os.path.exists(emb_path), f"Run E2 first! Missing: {emb_path}"
        print(f"\n[E3] backbone={bb}")

        out_metrics = os.path.join(config.RESULTS_E3, f"metrics_{bb}.json")
        out_model = os.path.join(config.ARTIFACTS, f"triage_{bb}.pt")
        out_logits = os.path.join(config.ARTIFACTS, f"triage_logits_{bb}.npy")

        if (os.path.exists(out_metrics) and os.path.exists(out_model)
                and os.path.exists(out_logits) and not config.FORCE_RERUN):
            print(f"  cache hit: {out_metrics}")
            with open(out_metrics) as f:
                all_results[bb] = json.load(f)
            progress.step(pbar, f"{bb} cache reused")
            continue

        emb = np.load(emb_path).astype(np.float32)
        y = master["answerable"].values

        train_mask = (master["split"] == "train").values
        X_train = emb[train_mask]; y_train = y[train_mask]
        X_cal = emb[val_idx[cal_pos]]; y_cal = y[val_idx[cal_pos]]
        X_rep = emb[val_idx[rep_pos]]; y_rep = y[val_idx[rep_pos]]
        dim = X_train.shape[1]

        # ── Baselines ──
        majority = int(y_train.mean() > 0.5)
        bl_f1 = f1_score(y_rep, np.full(len(y_rep), majority), zero_division=0)
        print(f"  Majority-class baseline: F1={bl_f1:.4f}  AUROC=0.5")

        # ── Multi-seed MLP runs ──
        def make_mlp():
            return heads.MLPHead(dim, 1)

        def thresh_fn(y_c, logits, split_name="cal"):
            return train_eval.find_threshold(y_c, logits, split_name=split_name)

        def eval_fn(y_r, logits_r, t):
            return train_eval.evaluate_binary(y_r, logits_r, t)

        print("  Running 5-seed MLP...")
        mlp_results = train_eval.run_multi_seed(
            make_mlp, X_train, y_train, X_cal, y_cal, X_rep, y_rep,
            label_names=["answerable"], threshold_fn=thresh_fn, eval_fn=eval_fn,
            seeds=config.SEEDS, device=device, loss_variant="pos_weight")

        # ── Bootstrap CI on mean-across-seed logits ──
        mean_logits = np.mean(mlp_results["_logits_rep"], axis=0)
        auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(
            lambda y_, s_: _roc(y_, s_), y_rep, mean_logits)

        np.save(out_logits, np.array(mlp_results["_logits_rep"]))

        # Save best model (retrain seed 0 for reproducibility)
        env.seed_everything(0)
        model0 = make_mlp()
        model0, _ = train_eval.train_head(model0, X_train, y_train, X_cal, y_cal,
                                          seed=0, device=device,
                                          loss_variant="pos_weight")
        torch.save(model0.state_dict(), out_model)

        result = {
            "backbone": bb,
            "AUROC": mlp_results.get("AUROC", {}),
            "AUPRC": mlp_results.get("AUPRC", {}),
            "F1": mlp_results.get("F1", {}),
            "balanced_acc": mlp_results.get("balanced_acc", {}),
            "auroc_bootstrap": {"mean": auroc_mean, "ci_lo": auroc_lo, "ci_hi": auroc_hi},
            "baseline_majority_f1": float(bl_f1),
        }
        os.makedirs(config.RESULTS_E3, exist_ok=True)
        with open(out_metrics, "w") as f:
            json.dump(result, f, indent=2)
        all_results[bb] = result

        auroc_m = result.get("AUROC", {}).get("mean", 0)
        auroc_s = result.get("AUROC", {}).get("std", 0)
        print(f"  [E3] {bb}: AUROC={auroc_m:.4f}+/-{auroc_s:.4f}  "
              f"AUPRC={result.get('AUPRC', {}).get('mean', 0):.4f}")
        progress.step(pbar, f"{bb} trained/evaluated")

    resultlog.log_run(EXP, metrics=all_results,
                      params={"backbones": config.BACKBONES, "seeds": config.SEEDS},
                      results_dir=config.RESULTS_E3, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, config.RESULTS_E3, artifacts=required_artifacts())
    progress.step(pbar, "E3 result logged")
    pbar.close()
    print("[E3 DONE]")
