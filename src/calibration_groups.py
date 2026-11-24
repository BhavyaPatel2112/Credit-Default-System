"""Improvement 4b: calibration within each gender x age group, a per-group
correction, and whether the correction changes the fairness result.

The overall ECE can be small while one group's risk is read too low: in 1.2,
men aged 35 and over were predicted 7.8% against 9.3% observed. Here:
  1. per group on TEST: mean predicted PD, observed default rate (Wilson 95%
     interval), calibration-in-the-large gap, ECE, reliability table;
  2. correction: a per-group intercept shift in log-odds, fitted on VALID
     (keeps the model's ranking within every group; four parameters, so it
     cannot overfit the way per-group isotonic regression could);
  3. the deployed policy (expected profit + group offsets) refitted on the
     corrected PDs, and its gaps and profit compared with the uncorrected one.

A per-group correction uses gender and age to set the PD itself, which has
the same legal status as using them to set the cut-off; the report treats it
as a diagnostic and an option, and the deployed model keeps one calibrator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

import policy as P
from fairness import group_key
from metrics import bootstrap, ece, eo_gap, reliability_table
from profit_policy import decide, fit_policy, summarise
from train import RUNS, groups_of, log_run


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    r = k / n
    den = 1 + z**2 / n
    c = (r + z**2 / (2 * n)) / den
    h = z * np.sqrt(r * (1 - r) / n + z**2 / (4 * n**2)) / den
    return float(c - h), float(c + h)


def table(d: pd.DataFrame, p: np.ndarray) -> list[dict]:
    keys, y = group_key(d), d.TARGET.values
    rows = []
    for k in sorted(set(keys)):
        m = keys == k
        lo, hi = wilson(int(y[m].sum()), int(m.sum()))
        rows.append({"group": k, "n": int(m.sum()), "mean_pd": float(p[m].mean()), "observed": float(y[m].mean()),
                     "observed_ci": [lo, hi], "gap_points": float((y[m].mean() - p[m].mean()) * 100),
                     "outside_ci": not (lo <= p[m].mean() <= hi), "ece": ece(y[m], p[m]),
                     "reliability": reliability_table(y[m], p[m], 5).to_dict("records")})
    return rows


def fit_shift(p: np.ndarray, y: np.ndarray) -> float:
    """Log-odds shift s with mean(expit(logit(p) + s)) = mean(y)."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return float(brentq(lambda s: expit(z + s).mean() - y.mean(), -3, 3))


def shifted(p: np.ndarray, keys: np.ndarray, shifts: dict) -> np.ndarray:
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return expit(z + np.array([shifts[k] for k in keys]))


def main() -> None:
    pv = pd.read_parquet(RUNS / "pred_valid.parquet")
    pt = pd.read_parquet(RUNS / "pred_test.parquet")
    kv, kt = group_key(pv), group_key(pt)
    before = table(pt, pt.p_cal.values)
    shifts = {k: fit_shift(pv.p_cal.values[kv == k], pv.TARGET.values[kv == k]) for k in sorted(set(kv))}
    p_corr = shifted(pt.p_cal.values, kt, shifts)
    after = table(pt, p_corr)
    for b, a in zip(before, after):
        print(f"{b['group']:20s} predicted {b['mean_pd']:.3%} observed {b['observed']:.3%} -> corrected {a['mean_pd']:.3%}")

    # Does the fairness result change? Refit the deployed policy type on corrected PDs.
    econ = dict(P.ECONOMICS)
    pol0 = fit_policy(pv, econ, fair=True)
    pv_c = pv.assign(p_cal=shifted(pv.p_cal.values, kv, shifts))
    pt_c = pt.assign(p_cal=p_corr)
    pol1 = fit_policy(pv_c, econ, fair=True)
    a0, _ = decide(pol0, pt)
    a1, _ = decide(pol1, pt_c)
    s0 = summarise("deployed, one calibrator", a0, pt, econ, boot=True)
    s1 = summarise("refitted on per-group corrected PDs", a1, pt_c, econ, boot=True)
    # and without any fairness offsets: does correcting calibration alone narrow the gaps?
    plain0, _ = decide(fit_policy(pv, econ, fair=False), pt)
    plain1, _ = decide(fit_policy(pv_c, econ, fair=False), pt_c)
    y = pt.TARGET.values
    log_run("r10_group_calibration", {
        "groups_before": before, "correction": "per-group log-odds intercept shift, fitted on valid",
        "shifts_logodds": shifts, "groups_after": after,
        "overall_ece_before": ece(y, pt.p_cal.values), "overall_ece_after": ece(y, p_corr),
        "policy_before": s0, "policy_after": s1,
        "offsets_before": pol0["offsets"], "offsets_after": pol1["offsets"],
        "no_offsets_gaps": {"before": {"gender": eo_gap(y, plain0, pt.gender.values), "age": eo_gap(y, plain0, pt.age_under_35.values)},
                            "after": {"gender": eo_gap(y, plain1, pt.gender.values), "age": eo_gap(y, plain1, pt.age_under_35.values)}},
        "bootstrap_after_correction": bootstrap(y, p_corr, a1, groups_of(pt)),
    })


if __name__ == "__main__":
    main()
