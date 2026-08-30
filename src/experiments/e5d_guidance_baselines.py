"""E5d - Trivial guidance-policy baselines (CPU, cached data).

Review Critical #4: framing is annotated in 56% of images, so a policy that
always says "step back so the whole item is in frame" may reach a substantial
GDMR without learning anything. E5d scores the reference policies the revised
Section IV promises, under the identical GDMR/AIRB protocol as the learned
head (top-1 defect, same match rule, same report split):

  always_framing     - always predict framing
  most_prevalent_una - always predict the most prevalent defect among
                       UNANSWERABLE training images (question-independent prior)
  prevalence_sampled - sample a defect per image from the training prevalence
                       distribution among unanswerable images
  uniform_random     - sample uniformly among the 7 defects
  oracle_random_gt   - pick a uniformly random ground-truth defect (upper
                       bound of top-1 matching; AIRB = GT defect prevalence)
  no_guidance        - never issue guidance (GDMR 0, AIRB 0; floor reference)

All stochastic policies use seed 42 and are averaged over 100 draws.
Needs: master.parquet, split_ids (E1). No GPU, no model outputs.
"""
import json
import os

import numpy as np
import pandas as pd

from src import config, env, expstate, progress, resultlog
from src.data_assembly import QUALITY_FLAWS

EXP = "E5D"
RESULTS_E5D = os.path.join(config.RESULTS, "E5d_guidance_baselines")
DEFECT_NAMES = QUALITY_FLAWS + ["unrecognizable"]
N_DRAWS = 100


def required_artifacts():
    return [os.path.join(RESULTS_E5D, "guidance_baselines.json")]


def _score(top_idx, gt_defects, ans):
    """GDMR/AIRB for an integer top-defect array (None -> -1)."""
    top_idx = np.asarray(top_idx)
    una = ans == 0
    guided = top_idx >= 0
    hits = [int(gt_defects[i, top_idx[i]] == 1) if top_idx[i] >= 0 else 0
            for i in np.where(una)[0]]
    gdmr = float(np.mean(hits)) if hits else float("nan")
    airb = float(guided[ans == 1].mean())
    return gdmr, airb


def main():
    progress.install_error_hook("E5d guidance baselines")
    env.seed_everything()
    env.mount_drive()
    config.ensure_output_dirs()
    os.makedirs(RESULTS_E5D, exist_ok=True)

    if expstate.is_done(EXP, RESULTS_E5D, required=required_artifacts()):
        expstate.skip_banner(EXP, RESULTS_E5D)
        return

    pbar = progress.notebook_bar("E5d guidance baselines", total=3)

    master = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "master.parquet"))
    cal_pos, rep_pos = env.load_split_ids(os.path.join(config.RESULTS_E1, "split_ids.json"))
    val_idx = np.where((master["split"] == "val").values)[0]
    defect_cols = [f"q_{d}" for d in DEFECT_NAMES]

    rep = master.iloc[val_idx[rep_pos]]
    gt = rep[defect_cols].values
    ans = rep["answerable"].values.astype(int)
    n = len(rep)

    # Priors are computed on the TRAINING split only (frozen-knob rule).
    train = master[master["split"] == "train"]
    train_una = train[train["answerable"] == 0]
    prevalence_una = train_una[defect_cols].values.mean(axis=0)
    most_prev_idx = int(np.argmax(prevalence_una))
    p_sample = prevalence_una / prevalence_una.sum()
    framing_idx = DEFECT_NAMES.index("framing")
    print(f"[E5d] train-unanswerable defect prevalence: "
          + "  ".join(f"{d}={p:.3f}" for d, p in zip(DEFECT_NAMES, prevalence_una)))
    print(f"[E5d] most prevalent among unanswerable: {DEFECT_NAMES[most_prev_idx]}")
    progress.step(pbar, "priors computed on train split")

    rng = np.random.default_rng(config.SEED)
    results = {}

    # Deterministic policies.
    results["always_framing"] = dict(zip(
        ("GDMR", "AIRB"), _score(np.full(n, framing_idx), gt, ans)))
    results["most_prevalent_una"] = dict(zip(
        ("GDMR", "AIRB"), _score(np.full(n, most_prev_idx), gt, ans)))
    results["no_guidance"] = dict(zip(
        ("GDMR", "AIRB"), _score(np.full(n, -1), gt, ans)))

    # Stochastic policies, averaged over draws.
    for name, chooser in {
        "prevalence_sampled": lambda: rng.choice(len(DEFECT_NAMES), size=n, p=p_sample),
        "uniform_random": lambda: rng.integers(0, len(DEFECT_NAMES), n),
    }.items():
        gs, as_ = [], []
        for _ in range(N_DRAWS):
            g, a = _score(chooser(), gt, ans)
            gs.append(g)
            as_.append(a)
        results[name] = {"GDMR": float(np.mean(gs)), "GDMR_std": float(np.std(gs)),
                         "AIRB": float(np.mean(as_)), "AIRB_std": float(np.std(as_))}

    # Oracle: uniformly random GT defect where one exists; no guidance otherwise.
    gs, as_ = [], []
    for _ in range(N_DRAWS):
        top = np.full(n, -1)
        for i in range(n):
            present = np.where(gt[i] == 1)[0]
            if len(present):
                top[i] = rng.choice(present)
        g, a = _score(top, gt, ans)
        gs.append(g)
        as_.append(a)
    results["oracle_random_gt"] = {"GDMR": float(np.mean(gs)),
                                   "AIRB": float(np.mean(as_))}
    progress.step(pbar, "all baseline policies scored")

    out = {"policies": results,
           "train_unanswerable_prevalence": dict(zip(DEFECT_NAMES,
                                                     prevalence_una.tolist())),
           "n_report": n, "n_unanswerable": int((ans == 0).sum()),
           "n_answerable": int((ans == 1).sum())}
    with open(required_artifacts()[0], "w") as f:
        json.dump(out, f, indent=2)

    print("\n[E5d] baseline GDMR/AIRB on report split:")
    for name, m in results.items():
        print(f"  {name:>20s}: GDMR={m['GDMR']:.4f}  AIRB={m['AIRB']:.4f}")

    resultlog.log_run(EXP, metrics=results, params={"n_draws": N_DRAWS},
                      results_dir=RESULTS_E5D, repo_root=config.REPO_ROOT)
    expstate.mark_done(EXP, RESULTS_E5D, artifacts=required_artifacts())
    pbar.close()
    print("[E5d DONE] Compare the learned head's GDMR against these floors "
          "in Sec. VI-D.")
