"""
Statistical helpers — shared by all experiments (§4.5.3–4.5.4).

multi_seed        — run over SEEDS=[0..4], return mean/std/ci95
bootstrap_ci      — percentile 95% CI over report samples
paired_bootstrap_delta — paired test of metric(A)-metric(B), p-value
benjamini_hochberg — BH-FDR correction for per-defect multi-test
delong_auroc      — DeLong's test for correlated AUROCs (E9 cross-check)
"""
import numpy as np
from sklearn.metrics import roc_auc_score


# ── Multi-seed aggregation ────────────────────────────────────────────────────

def multi_seed(train_fn, seeds=(0, 1, 2, 3, 4), **kw) -> dict:
    """
    Call train_fn(seed=s, **kw) for each seed in `seeds`.
    train_fn must return a dict of scalar metrics.
    Returns per-metric dict with keys: mean, std, ci95, seeds.
    """
    runs = [train_fn(seed=s, **kw) for s in seeds]
    keys = runs[0].keys()
    agg = {}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        se = v.std(ddof=1) / np.sqrt(len(v))
        agg[k] = dict(
            mean=float(v.mean()),
            std=float(v.std(ddof=1)),
            ci95=float(1.96 * se),
            seeds=v.tolist(),
        )
    return agg


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(metric_fn, *arrays, n_boot: int = 2000, seed: int = 42):
    """
    Percentile 95% CI of metric_fn(*arrays) by resampling sample indices
    with replacement (N_BOOT=2000).
    Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    n   = len(arrays[0])
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(*[a[idx] for a in arrays]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)


# ── Paired bootstrap delta ────────────────────────────────────────────────────

def paired_bootstrap_delta(
    metric_fn, y, score_a, score_b,
    n_boot: int = 2000, seed: int = 42
):
    """
    Paired test of metric(A) − metric(B) on the SAME resampled indices.
    Returns (delta, lo, hi, p_two_sided).
    The AURC delta for E7 and the ΔAUROC for E9 are the headline uses.
    """
    rng    = np.random.default_rng(seed)
    n      = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        da  = metric_fn(y[idx], score_a[idx])
        db  = metric_fn(y[idx], score_b[idx])
        deltas.append(da - db)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided p: fraction of bootstrap replicates on the "wrong" side
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(deltas.mean()), float(lo), float(hi), float(p)


# ── Benjamini–Hochberg FDR correction ────────────────────────────────────────

def benjamini_hochberg(pvals, alpha: float = 0.05) -> np.ndarray:
    """
    Return boolean reject array under BH-FDR for per-defect multiple tests.
    Applied whenever we compare per-defect metrics across backbones or conditions.
    """
    p      = np.asarray(pvals, dtype=float)
    m      = len(p)
    order  = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k      = np.where(passed)[0]
    cut    = order[:k.max() + 1] if len(k) else np.array([], int)
    out    = np.zeros(m, dtype=bool)
    out[cut] = True
    return out


# ── DeLong's test for correlated AUROCs ──────────────────────────────────────

def _delong_structural_components(y, proba):
    """Compute structural components for DeLong AUC variance estimation."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    pos = proba[y == 1]
    neg = proba[y == 0]

    # V10[i] = mean indicator (pos[i] > neg) over neg
    V10 = np.zeros(n1)
    for i, p in enumerate(pos):
        V10[i] = np.mean(neg < p) + 0.5 * np.mean(neg == p)

    # V01[j] = mean indicator over pos
    V01 = np.zeros(n0)
    for j, n_val in enumerate(neg):
        V01[j] = np.mean(pos > n_val) + 0.5 * np.mean(pos == n_val)

    return V10, V01


def delong_auroc(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray):
    """
    DeLong's test for comparing two correlated AUROCs on the same sample set.
    Returns (auc_a, auc_b, delta_auc, z_stat, p_two_sided).
    Used as the parametric cross-check for the groundability ΔAUROC in E9/RQ3a.
    """
    from scipy import stats as scipy_stats

    y = np.asarray(y, int)
    auc_a = roc_auc_score(y, score_a)
    auc_b = roc_auc_score(y, score_b)

    V10_a, V01_a = _delong_structural_components(y, score_a)
    V10_b, V01_b = _delong_structural_components(y, score_b)

    n1, n0 = len(V10_a), len(V01_a)

    S01 = (np.cov(V10_a, V10_b) / n1)   # 2×2 covariance matrix part
    S10 = (np.cov(V01_a, V01_b) / n0)

    var_a  = np.var(V10_a, ddof=1) / n1 + np.var(V01_a, ddof=1) / n0
    var_b  = np.var(V10_b, ddof=1) / n1 + np.var(V01_b, ddof=1) / n0
    cov_ab = (np.cov(V10_a, V10_b)[0, 1] / n1 +
              np.cov(V01_a, V01_b)[0, 1] / n0)

    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        var_diff = 1e-12  # numerical safety

    delta = auc_a - auc_b
    z     = delta / np.sqrt(var_diff)
    p     = float(2 * scipy_stats.norm.sf(abs(z)))
    return float(auc_a), float(auc_b), float(delta), float(z), p
