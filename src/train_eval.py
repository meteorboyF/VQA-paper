"""
Training loops and evaluation metrics for the triage and defect heads.

Rules enforced here:
- AdamW comes from torch.optim (NOT transformers).
- Real minibatch DataLoader (never full-batch single-step training).
- pos_weight / focal variants for imbalance handling.
- All threshold selection uses ONLY the cal split (enforced via env.assert_no_rep_leakage).
- Returns per-seed metric arrays, never scalars.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix,
)

from src import env, config


# ── Loss variants ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits).detach()
        pt = torch.where(targets == 1, p, 1 - p)
        focal_w = (1 - pt) ** self.gamma
        return (focal_w * bce).mean()


def build_criterion(loss_variant: str, pos_weight=None, device="cpu"):
    if loss_variant == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_variant == "pos_weight":
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32).to(device)
        else:
            pw = None
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    elif loss_variant == "focal":
        pw = torch.tensor(pos_weight, dtype=torch.float32).to(device) if pos_weight is not None else None
        return FocalLoss(gamma=2.0, pos_weight=pw)
    else:
        raise ValueError(f"Unknown loss_variant: {loss_variant}")


# ── Compute pos_weight from training labels ───────────────────────────────────

def compute_pos_weight(labels: np.ndarray) -> np.ndarray:
    """pos_weight[i] = (#negatives / #positives) for class i."""
    pos = labels.sum(axis=0).clip(min=1)
    neg = (labels.shape[0] - labels.sum(axis=0)).clip(min=1)
    return neg / pos


# ── Single training run ───────────────────────────────────────────────────────

def train_head(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int = 42,
    lr: float = None,
    weight_decay: float = None,
    max_epochs: int = None,
    patience: int = None,
    batch_size: int = None,
    loss_variant: str = "pos_weight",
    device: str = "cuda",
):
    """
    Train a head on (X_train, y_train), early-stop on val AUROC via (X_cal, y_cal).
    Returns trained model and cal-split logits.
    """
    env.seed_everything(seed)
    lr           = lr           or config.LR
    weight_decay = weight_decay or config.WEIGHT_DECAY
    max_epochs   = max_epochs   or config.MAX_EPOCHS
    patience     = patience     or config.PATIENCE
    batch_size   = batch_size   or config.BATCH_SIZE

    y_train_np = np.array(y_train, dtype=np.float32)
    pos_w = compute_pos_weight(y_train_np.reshape(-1, 1) if y_train_np.ndim == 1
                                else y_train_np)
    criterion = build_criterion(loss_variant, pos_w, device)

    model = model.to(device)
    # Use torch.optim.AdamW (NOT transformers.AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train_np)
    ds = TensorDataset(X_t, y_t)
    nw = min(max(1, (os.cpu_count() or 2) - 1), 4)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=0,            # 0 for TensorDataset on GPU
                    pin_memory=(device != "cpu"))

    X_cal_t = torch.tensor(X_cal, dtype=torch.float32).to(device)

    best_auroc = -1.0
    best_state = None
    patience_ctr = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device != "cpu")):
                logits = model(xb)
                loss = criterion(logits.squeeze(-1) if logits.shape[-1] == 1 else logits,
                                 yb)
            loss.backward()
            optimizer.step()

        # Evaluate on cal split
        model.eval()
        with torch.inference_mode():
            cal_logits = model(X_cal_t).squeeze(-1).float().cpu().numpy()
        y_cal_np = np.array(y_cal, dtype=np.float32)
        try:
            if y_cal_np.ndim == 1:
                auroc = roc_auc_score(y_cal_np, cal_logits)
            else:
                auroc = roc_auc_score(y_cal_np, cal_logits, average="macro",
                                       multi_class="ovr")
        except ValueError:
            auroc = 0.0

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_auroc


# ── Threshold selection (cal only) ───────────────────────────────────────────

def find_threshold(y_true: np.ndarray, logits: np.ndarray,
                   split_name: str = "cal",
                   criterion: str = "f1") -> float:
    """
    Select decision threshold on the cal split. Raises if called with rep/test data.
    Returns the scalar threshold.
    """
    env.assert_no_rep_leakage(split_name)
    probs = 1 / (1 + np.exp(-logits))
    candidates = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -1.0
    for t in candidates:
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)


# ── Evaluation on the report split ───────────────────────────────────────────

def evaluate_binary(y_true: np.ndarray, logits: np.ndarray,
                    threshold: float) -> dict:
    """Binary triage metrics at a frozen threshold."""
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    auroc  = roc_auc_score(y_true, probs)
    auprc  = average_precision_score(y_true, probs)
    f1     = f1_score(y_true, preds, zero_division=0)
    prec   = precision_score(y_true, preds, zero_division=0)
    rec    = recall_score(y_true, preds, zero_division=0)
    bacc   = balanced_accuracy_score(y_true, preds)
    cm     = confusion_matrix(y_true, preds).tolist()
    return dict(AUROC=auroc, AUPRC=auprc, F1=f1,
                precision=prec, recall=rec, balanced_acc=bacc,
                confusion_matrix=cm, threshold=threshold)


def evaluate_multilabel(y_true: np.ndarray, logits: np.ndarray,
                        thresholds: np.ndarray,
                        label_names: list) -> dict:
    """
    Multi-label defect metrics.
    Returns per-defect AUROC/AUPRC + macro/micro-F1 + mAP.
    NEVER returns a single 7×7 confusion matrix — uses one-vs-rest 2×2 per defect.
    """
    probs = 1 / (1 + np.exp(-logits))
    n_labels = y_true.shape[1]

    per_defect = {}
    cms = {}
    for i, name in enumerate(label_names):
        yi, pi, ti = y_true[:, i], probs[:, i], thresholds[i]
        try:
            auroc_i = roc_auc_score(yi, pi)
        except ValueError:
            auroc_i = float("nan")
        auprc_i = average_precision_score(yi, pi)
        preds_i = (pi >= ti).astype(int)
        f1_i   = f1_score(yi, preds_i, zero_division=0)
        prec_i = precision_score(yi, preds_i, zero_division=0)
        rec_i  = recall_score(yi, preds_i, zero_division=0)
        cm_i   = confusion_matrix(yi, preds_i).tolist()   # 2×2
        per_defect[name] = dict(AUROC=auroc_i, AUPRC=auprc_i,
                                F1=f1_i, precision=prec_i, recall=rec_i)
        cms[name] = cm_i

    preds_all = (probs >= thresholds).astype(int)
    macro_f1 = f1_score(y_true, preds_all, average="macro",  zero_division=0)
    micro_f1 = f1_score(y_true, preds_all, average="micro",  zero_division=0)
    mAP      = float(np.nanmean([per_defect[n]["AUPRC"] for n in label_names]))

    return dict(
        per_defect_auroc={n: per_defect[n]["AUROC"]  for n in label_names},
        per_defect_auprc={n: per_defect[n]["AUPRC"]  for n in label_names},
        per_defect_f1   ={n: per_defect[n]["F1"]     for n in label_names},
        per_defect_prec ={n: per_defect[n]["precision"] for n in label_names},
        per_defect_rec  ={n: per_defect[n]["recall"]  for n in label_names},
        confusion_matrices_one_vs_rest=cms,   # 2×2 per defect
        macro_F1=macro_f1, micro_F1=micro_f1, mAP=mAP,
    )


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def run_multi_seed(
    make_model_fn,
    X_train, y_train,
    X_cal,   y_cal,
    X_rep,   y_rep,
    label_names,
    threshold_fn,
    eval_fn,
    seeds=None,
    device="cuda",
    **train_kwargs,
) -> dict:
    """
    Train `make_model_fn()` for each seed. Returns aggregated stats dict.
    threshold_fn(y_cal, logits_cal) → threshold(s); must pass split="cal".
    eval_fn(y_rep, logits_rep, threshold) → metrics dict.
    """
    from src.stats import multi_seed as agg_seeds
    seeds = seeds or config.SEEDS

    all_logits_cal = []
    all_logits_rep = []
    all_thresholds = []

    def run_one(seed):
        env.seed_everything(seed)
        model = make_model_fn()
        model, _ = train_head(model, X_train, y_train, X_cal, y_cal,
                               seed=seed, device=device, **train_kwargs)
        with torch.inference_mode():
            device_t = next(model.parameters()).device
            lc = model(torch.tensor(X_cal, dtype=torch.float32).to(device_t))
            lr = model(torch.tensor(X_rep, dtype=torch.float32).to(device_t))
            if isinstance(lc, tuple):
                lc, lr = lc[0], lr[0]     # joint head → take triage branch
            lc = lc.squeeze(-1).float().cpu().numpy()
            lr = lr.squeeze(-1).float().cpu().numpy()
        threshold = threshold_fn(np.array(y_cal), lc)
        metrics   = eval_fn(np.array(y_rep), lr, threshold)
        all_logits_cal.append(lc)
        all_logits_rep.append(lr)
        all_thresholds.append(threshold)
        return metrics

    results = [run_one(s) for s in seeds]

    # Aggregate across seeds
    keys = results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        if isinstance(vals[0], (int, float)):
            arr = np.array(vals, float)
            agg[k] = dict(mean=float(arr.mean()), std=float(arr.std(ddof=1)),
                          seeds=arr.tolist())
        elif isinstance(vals[0], dict):
            agg[k] = {sub: dict(
                mean=float(np.mean([v[sub] for v in vals])),
                std=float(np.std([v[sub] for v in vals], ddof=1)),
                seeds=[v[sub] for v in vals],
            ) for sub in vals[0]}
        else:
            agg[k] = vals   # non-numeric (e.g., confusion matrices)

    agg["_logits_cal"] = all_logits_cal
    agg["_logits_rep"] = all_logits_rep
    agg["_thresholds"] = all_thresholds
    return agg
