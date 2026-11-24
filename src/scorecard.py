"""Improvement 3: a traditional bank scorecard as the challenger model.

Weight-of-Evidence bins (optbinning, monotonic where the data allow) feed a
logistic regression, converted to points: 600 points at odds of 50:1 of
repaying, and 20 more points double those odds (PDO 20). Trained on the same
train split, calibrated with isotonic regression on valid like the champion,
reported on test, and run through the same gate requirement checks
(src/gate.py: requirement_checks), with the same 5:1 single cut-off and the
same levelling-up group cut-offs as the champion.

Writes
  models/scorecard.pkl              the fitted scorecard + calibrator
  reports/scorecard_points.csv      the points table (published)
  runs/r9_scorecard.json            champion vs challenger
  runs/frontier_extra.csv           the challenger's points on the frontier
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fairness import apply_thresholds, fit_group_thresholds, group_key
from gate import requirement_checks
from metrics import ece, eo_gap, expected_cost
from train import C_FN, C_FP, MODELS, RUNS, SEED, THRESHOLD, load, log_run

REPORTS = RUNS.parent / "reports"
N_VARIABLES = 30   # a scorecard a credit committee can read
PDO, BASE_POINTS, BASE_ODDS = 20, 600, 50


def candidates(tr: pd.DataFrame, feats: list[str], n: int = 80) -> list[str]:
    """Shortlist by the champion's gain (one-hot levels are binned like any
    numeric column), so binning 400 columns does not dominate the run."""
    gain = pd.read_csv(RUNS / "feature_importance_gain.csv", index_col=0).iloc[:, 0]
    top = [f for f in gain.index if f in feats][:n]
    return top


def fit(tr, feats):
    X, y = tr[feats], tr.TARGET.values
    bp = BinningProcess(feats, selection_criteria={"iv": {"min": 0.02, "top": N_VARIABLES}},
                        binning_fit_params={f: {"monotonic_trend": "auto_asc_desc"} for f in feats},
                        n_jobs=8)
    sc = Scorecard(binning_process=bp, estimator=LogisticRegression(C=1.0, max_iter=2000),
                   scaling_method="pdo_odds",
                   scaling_method_params={"pdo": PDO, "odds": BASE_ODDS, "scorecard_points": BASE_POINTS},
                   intercept_based=False, reverse_scorecard=False)
    sc.fit(X, y)
    return sc


def main() -> None:
    parts, feats = load()
    tr, va, te = parts["train"], parts["valid"], parts["test"]
    cand = candidates(tr, feats)
    sc = fit(tr, cand)
    chosen = [v for v in cand if v in sc.binning_process_.get_support(names=True)]
    print(f"scorecard: {len(chosen)} variables from {len(cand)} candidates")

    raw_va = sc.predict_proba(va[cand])[:, 1]
    raw_te = sc.predict_proba(te[cand])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, va.TARGET.values)
    p_va, p_te = iso.predict(raw_va), iso.predict(raw_te)
    pickle.dump({"scorecard": sc, "calibrator": iso, "candidates": cand}, open(MODELS / "scorecard.pkl", "wb"))

    table = sc.table(style="detailed")
    table = table[["Variable", "Bin", "Count", "Event rate", "WoE", "IV", "Points"]]
    table.to_csv(REPORTS / "scorecard_points.csv", index=False)
    score_te = sc.score(te[cand])

    yv, yt = va.TARGET.values, te.TARGET.values
    kv, kt = group_key(va), group_key(te)
    th = fit_group_thresholds(p_va, yv, kv, 0.03, grid=np.linspace(THRESHOLD, 0.6, 900))
    a_single = p_te < THRESHOLD
    a_group = apply_thresholds(p_te, kt, th) if th else a_single

    champ = pd.read_parquet(RUNS / "pred_test.parquet")
    assert (champ.SK_ID_CURR.values == te.SK_ID_CURR.values).all()

    def block(p, a):
        return {"roc_auc": float(roc_auc_score(yt, p)), "pr_auc": float(average_precision_score(yt, p)),
                "brier": float(brier_score_loss(yt, p)), "ece": ece(yt, p), "approval_rate": float(a.mean()),
                "default_rate_among_approved": float(yt[a].mean()), "expected_cost_5to1": expected_cost(yt, a, C_FN, C_FP),
                "eo_gap_gender": eo_gap(yt, a, te.gender.values), "eo_gap_age": eo_gap(yt, a, te.age_under_35.values),
                "ece_by_group": {k: ece(yt[kt == k], p[kt == k]) for k in sorted(set(kt))}}

    champ_va = pd.read_parquet(RUNS / "pred_valid.parquet")
    assert (champ_va.SK_ID_CURR.values == va.SK_ID_CURR.values).all()
    ch_th = fit_group_thresholds(champ_va.p_cal.values, yv, kv, 0.03, grid=np.linspace(THRESHOLD, 0.6, 900))
    ch_group = apply_thresholds(champ.p_cal.values, kt, ch_th)
    gate_rows = requirement_checks(yt, p_te, a_group, te.gender.values, te.age_under_35.values, kt)
    gate_view = [{"check": n, "value": v, "pass": ok} for n, v, ok in gate_rows]

    log_run("r9_scorecard", {
        "method": f"WoE binning (optbinning) + logistic regression, {len(chosen)} variables, "
                  f"{BASE_POINTS} points at {BASE_ODDS}:1 odds of repaying, PDO {PDO}",
        "variables": chosen,
        "score_range_test": {"p1": float(np.percentile(score_te, 1)), "median": float(np.median(score_te)),
                             "p99": float(np.percentile(score_te, 99))},
        "test": block(p_te, a_group) | {"group_cutoffs": th},
        "test_single_cutoff": block(p_te, a_single),
        "champion_xgboost": block(champ.p_cal.values, ch_group),
        "gate": gate_view, "gate_passed": all(ok is not False for _, _, ok in gate_rows),
        "auc_gain_of_champion": float(roc_auc_score(yt, champ.p_cal.values) - roc_auc_score(yt, p_te)),
    })

    rows = []
    for name, a, eps in [("Scorecard challenger, single cut-off", a_single, None),
                         ("Scorecard challenger, group cut-offs", a_group, 0.03)]:
        r = {"method": name, "eps": eps, "thresholds": json.dumps(th if eps else {"all": THRESHOLD}),
             "test_gap_gender": eo_gap(yt, a, te.gender.values), "test_gap_age": eo_gap(yt, a, te.age_under_35.values),
             "test_gap_intersection": eo_gap(yt, a, kt), "test_cost": expected_cost(yt, a, C_FN, C_FP),
             "test_auc": float(roc_auc_score(yt, p_te)), "test_approval": float(a.mean()),
             "uses_protected_at_decision": eps is not None}
        r["test_gap_max"] = max(r["test_gap_gender"], r["test_gap_age"])
        rows.append(r)
    path = RUNS / "frontier_extra.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.method.str.startswith("Scorecard")]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
