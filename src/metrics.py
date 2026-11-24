"""One harness for every run: predictive, calibration and fairness metrics.

Decision convention
-------------------
The model outputs p = P(default). An applicant is APPROVED when p < threshold.
Equal opportunity (Hardt et al., 2016) is measured on the *favourable* outcome:
among applicants who actually repaid (TARGET == 0), the approval rate should
not depend on the protected group. The gap is the max-min difference of that
rate across groups (NFR3 / ETH2: gap <= 0.05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

NFR1_AUC = 0.75
NFR2_ECE = 0.05
NFR3_GAP = 0.05
MIN_CELL = 30  # R-4: a group with fewer repayers than this is reported, not gated


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile") -> float:
    """Expected calibration error with equal-mass bins (robust at an 8% base rate)."""
    y, p = np.asarray(y), np.asarray(p)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        rows.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "obs_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


def group_rates(y, approve, groups) -> pd.DataFrame:
    """Per-group approval rate among repayers (TPR on the favourable outcome),
    with a Wilson 95% interval and the cell size."""
    d = pd.DataFrame({"y": np.asarray(y), "a": np.asarray(approve).astype(int), "g": np.asarray(groups)})
    rows = []
    for g, sub in d.groupby("g"):
        rep = sub[sub.y == 0]
        n = len(rep)
        k = int(rep.a.sum())
        rate = k / n if n else np.nan
        z = 1.96
        if n:
            den = 1 + z**2 / n
            c = (rate + z**2 / (2 * n)) / den
            h = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / den
        else:
            c = h = np.nan
        rows.append({"group": g, "n_applicants": len(sub), "n_repayers": n,
                     "n_defaults": int((sub.y == 1).sum()),
                     "approval_rate": float(sub.a.mean()),
                     "tpr_repayers": rate, "ci_low": c - h, "ci_high": c + h,
                     "default_rate": float(sub.y.mean())})
    return pd.DataFrame(rows)


def eo_gap(y, approve, groups) -> float:
    t = group_rates(y, approve, groups)
    t = t[t.n_repayers >= MIN_CELL]
    return float(t.tpr_repayers.max() - t.tpr_repayers.min())


def decision_metrics(y, approve) -> dict:
    y, a = np.asarray(y), np.asarray(approve).astype(bool)
    return {
        "approval_rate": float(a.mean()),
        "default_rate_among_approved": float(y[a].mean()) if a.any() else np.nan,
        "share_of_defaulters_declined": float((~a[y == 1]).mean()),
        "share_of_repayers_approved": float(a[y == 0].mean()),
    }


def expected_cost(y, approve, c_fn: float, c_fp: float) -> float:
    """Average cost per applicant: approving a defaulter costs c_fn, declining a repayer c_fp."""
    y, a = np.asarray(y), np.asarray(approve).astype(bool)
    return float((c_fn * ((y == 1) & a) + c_fp * ((y == 0) & ~a)).mean())


def evaluate(y, p, groups: dict[str, np.ndarray] | None = None, threshold: float = 0.5,
             approve: np.ndarray | None = None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    if approve is None:
        approve = p < threshold
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "threshold": threshold,
        **decision_metrics(y, approve),
    }
    if groups:
        for name, g in groups.items():
            out[f"eo_gap_{name}"] = eo_gap(y, approve, g)
    return out


def gate(m: dict, fairness_key: str = "eo_gap_gender") -> dict:
    checks = {
        "NFR1_auc": m["roc_auc"] >= NFR1_AUC,
        "NFR2_ece": m["ece"] <= NFR2_ECE,
        "NFR3_eo_gap": m[fairness_key] <= NFR3_GAP,
    }
    return {"passed": all(checks.values()), "checks": checks}


# ---------------------------------------------------------------------------
# Improvement 4: bootstrap intervals for every gate metric
# ---------------------------------------------------------------------------
BOOT_B = 1000
BOOT_SEED = 42


def bootstrap(y, p, approve, groups: dict[str, np.ndarray], B: int = BOOT_B, seed: int = BOOT_SEED,
              level: float = 0.95) -> dict:
    """Percentile bootstrap over applicants (resampled with replacement, B
    times) for AUC, ECE and each equal-opportunity gap. Each replicate is a
    vector of multiplicities w, so every statistic is a weighted sum and
    one replicate costs O(n): AUC from per-score class weights (ties count
    half, as in roc_auc_score), ECE with the bins of the full sample, gaps
    from weighted repayer approval rates per group (cells under MIN_CELL
    repayers on the full sample are left out, as in eo_gap)."""
    y, p, a = np.asarray(y), np.asarray(p, dtype=float), np.asarray(approve).astype(bool)
    n = len(y)
    rng = np.random.default_rng(seed)
    u, inv = np.unique(p, return_inverse=True)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    gcodes = {}
    for name, g in groups.items():
        g = np.asarray(g)
        keep = [k for k in np.unique(g) if ((g == k) & (y == 0)).sum() >= MIN_CELL]
        gcodes[name] = (np.array([keep.index(k) if k in keep else -1 for k in g]), len(keep))
    rep = y == 0
    out = {"auc": [], "ece": [], **{f"eo_gap_{k}": [] for k in groups}}
    for _ in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        pos = np.bincount(inv, w * y, len(u)); neg = np.bincount(inv, w * (1 - y), len(u))
        below = np.cumsum(neg) - neg
        out["auc"].append(float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum())))
        wb = np.bincount(bins, w, nb)
        mp = np.bincount(bins, w * p, nb); my = np.bincount(bins, w * y, nb)
        ok = wb > 0
        out["ece"].append(float((np.abs(mp[ok] - my[ok])).sum() / w.sum()))
        for name, (code, k) in gcodes.items():
            m = rep & (code >= 0)
            den = np.bincount(code[m], w[m], k); num = np.bincount(code[m], w[m] * a[m], k)
            r = num[den > 0] / den[den > 0]
            out[f"eo_gap_{name}"].append(float(r.max() - r.min()))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    point = {"auc": float(roc_auc_score(y, p)), "ece": ece(y, p),
             **{f"eo_gap_{k}": eo_gap(y, a, g) for k, g in groups.items()}}
    return {k: {"point": point[k], "low": float(np.percentile(v, lo)), "high": float(np.percentile(v, hi))}
            for k, v in out.items()} | {"B": B, "level": level}
