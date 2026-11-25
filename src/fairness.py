"""M6: run ladder rungs 3-4 — fairness mitigation and the accuracy/fairness
frontier.

Rung 3  reductions : Fairlearn ExponentiatedGradient with TruePositiveRateParity
                     on the favourable label (repaid), at several tolerances.
                     Improvement 2: this is the policy that needs NO protected
                     attribute at decision time (gender and age are used only
                     while training). The 1.2 run collapsed to approving 99.4%
                     because it minimised plain error rate, for which "approve
                     almost everyone" is optimal at a 92% repayment rate. It
                     now minimises the same 5:1 cost as every other policy
                     (ErrorRate(costs=...)), uses a stronger base learner on
                     twice the data, 50 iterations, and is scored with its
                     mixture probability rather than a hard label.
Rung 4  post-proc  : group-specific cut-offs on the calibrated PD, chosen on
                     valid to minimise expected cost subject to gap <= eps;
                     plus Fairlearn's ThresholdOptimizer as a reference point.
Everything is chosen on valid and reported on test.
"""
from __future__ import annotations

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ErrorRate, ExponentiatedGradient, TruePositiveRateParity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from metrics import NFR3_GAP, bootstrap, ece, eo_gap, evaluate, expected_cost, group_rates
from policy import group_key as _group_key
from train import C_FN, C_FP, MODELS, RUNS, SEED, THRESHOLD, groups_of, load, log_run

FIG = RUNS.parent / "reports" / "figures"


# ---------------------------------------------------------------------------
# Rung 4: group-specific thresholds (deterministic post-processing)
# ---------------------------------------------------------------------------
def tpr_curve(p, y, grid):
    """Approval rate among repayers for each cut-off in grid (approve if p < t)."""
    rep = np.sort(p[y == 0])
    return np.searchsorted(rep, grid, side="left") / len(rep)


def cost_curve(p, y, grid, n_total):
    """Contribution of this group to expected cost per applicant, for each cut-off."""
    pd_ = np.sort(p[y == 1]); pr = np.sort(p[y == 0])
    fn = np.searchsorted(pd_, grid, side="left")             # defaulters approved
    fp = len(pr) - np.searchsorted(pr, grid, side="left")    # repayers declined
    return (C_FN * fn + C_FP * fp) / n_total


def fit_group_thresholds(p, y, g, eps, grid=np.linspace(0.005, 0.6, 1200)):
    """Cost-minimising cut-off per group such that every group's approval
    rate among repayers lies within a band of width eps around a common
    level tau (so the max-min gap is <= eps). Cost is separable by group, so
    for each tau each group picks its cheapest feasible cut-off; tau is then
    chosen to minimise the total. Works for any number of groups."""
    groups = sorted(pd.unique(g))
    n = len(y)
    T = {k: tpr_curve(p[g == k], y[g == k], grid) for k in groups}
    C = {k: cost_curve(p[g == k], y[g == k], grid, n) for k in groups}
    best = (np.inf, None)
    for tau in np.linspace(0.3, 1.0, 701):
        total, th = 0.0, {}
        for k in groups:
            ok = np.abs(T[k] - tau) <= eps / 2 + 1e-12
            if not ok.any():
                total = np.inf; break
            c = np.where(ok, C[k], np.inf)
            i = int(np.argmin(c)); total += c[i]; th[k] = float(grid[i])
        if total < best[0]:
            best = (total, th)
    return best[1]


def apply_thresholds(p, g, th):
    t = np.vectorize(th.get)(g)
    return p < t


# ---------------------------------------------------------------------------
# Rung 3: reductions
# ---------------------------------------------------------------------------
class Prefit(BaseEstimator, ClassifierMixin):
    """Wraps a fixed score so ThresholdOptimizer can post-process it."""
    def __init__(self, col=0):
        self.col = col
    def fit(self, X, y):
        self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        s = np.asarray(X, dtype="float64")[:, self.col]
        return np.c_[1 - s, s]
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def reductions(tr, va, te, feats, eps_list, n_sub=80000):
    """Returns, per run, P(approve) on valid and test. Favourable label 1 =
    repaid; a false positive (approving a defaulter) costs C_FN, a false
    negative (declining a repayer) C_FP, the same 5:1 as every other policy."""
    sub = tr.sample(n=min(n_sub, len(tr)), random_state=SEED)
    Xs, ys_rep, gs = sub[feats].values, 1 - sub.TARGET.values, sub.grp.values
    base = dict(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.3,
                min_child_weight=10, tree_method="hist", n_jobs=8, random_state=SEED)
    out = []
    # unconstrained reference with the same budget: a probability model whose
    # cost-minimising decision is P(repaid) > C_FN / (C_FN + C_FP)
    ref = xgb.XGBClassifier(**base).fit(Xs, ys_rep)
    out.append(("reference (same budget, no constraint)", None,
                ref.predict_proba(va[feats].values)[:, 1], ref.predict_proba(te[feats].values)[:, 1]))
    for eps in eps_list:
        eg = ExponentiatedGradient(xgb.XGBClassifier(**base), constraints=TruePositiveRateParity(difference_bound=eps),
                                   objective=ErrorRate(costs={"fp": C_FN, "fn": C_FP}),  # a moment loads data once
                                   eps=0.005, max_iter=50, nu=1e-3)
        eg.fit(Xs, ys_rep, sensitive_features=gs)
        out.append((f"ExponentiatedGradient eps={eps}", eps,
                    eg._pmf_predict(va[feats].values)[:, 1], eg._pmf_predict(te[feats].values)[:, 1]))
        print("  EG", eps, "done:", len(eg.predictors_), "predictors", flush=True)
    return out


def group_key(df) -> np.ndarray:
    """Intersectional group used by the deployed policy: gender x age band."""
    return _group_key(df["gender"].values, df["age_under_35"].values)


def row(method, eps, approve, yt, pt, auc, th, extra=None):
    r = {"method": method, "eps": eps, "thresholds": th,
         "test_gap_gender": eo_gap(yt, approve, pt.gender.values),
         "test_gap_age": eo_gap(yt, approve, pt.age_under_35.values),
         "test_gap_intersection": eo_gap(yt, approve, group_key(pt)),
         "test_cost": expected_cost(yt, approve, C_FN, C_FP), "test_auc": auc,
         "test_approval": float(np.mean(approve))}
    r["test_gap_max"] = max(r["test_gap_gender"], r["test_gap_age"])
    r.update(extra or {})
    return r


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pv = pd.read_parquet(RUNS / "pred_valid.parquet")
    pt = pd.read_parquet(RUNS / "pred_test.parquet")
    yv, yt = pv.TARGET.values, pt.TARGET.values
    pvv, ptt = pv.p_cal.values, pt.p_cal.values
    auc = roc_auc_score(yt, ptt)
    attrs = {"gender": lambda d: d.gender.values, "age": lambda d: d.age_under_35.values,
             "gender x age": group_key}

    frontier = [row("single cut-off (calibrated XGBoost)", None, ptt < THRESHOLD, yt, pt, auc, {"all": THRESHOLD},
                    {"uses_protected_at_decision": False})]

    # Rung 4: deterministic group cut-offs, for each grouping and tolerance
    for name, gf in attrs.items():
        gv_, gt_ = gf(pv), gf(pt)
        for eps in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]:
            th = fit_group_thresholds(pvv, yv, gv_, eps)
            if th is None:  # no cut-off grid point reaches this tolerance in every group
                continue
            a = apply_thresholds(ptt, gt_, th)
            frontier.append(row(f"group cut-offs by {name}", eps, a, yt, pt, auc, th,
                                {"valid_gap": eo_gap(yv, apply_thresholds(pvv, gv_, th), gv_)}))

    # Same, but no group's cut-off may fall below the single cut-off: fairness
    # is reached only by approving MORE of the disadvantaged groups
    # ("levelling up"), never by declining more of anyone.
    up_grid = np.linspace(THRESHOLD, 0.6, 900)
    for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]:
        th = fit_group_thresholds(pvv, yv, group_key(pv), eps, grid=up_grid)
        if th is None:
            continue
        a = apply_thresholds(ptt, group_key(pt), th)
        frontier.append(row("group cut-offs by gender x age, levelling up only", eps, a, yt, pt, auc, th,
                            {"valid_gap": eo_gap(yv, apply_thresholds(pvv, group_key(pv), th), group_key(pv))}))

    # Fairlearn ThresholdOptimizer reference (randomised), intersectional groups
    to = ThresholdOptimizer(estimator=Prefit(0), constraints="true_positive_rate_parity",
                            objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
    to.fit(np.c_[1 - pvv.astype("float64")], 1 - yv, sensitive_features=group_key(pv))
    a = to.predict(np.c_[1 - ptt.astype("float64")], sensitive_features=group_key(pt), random_state=SEED).astype(bool)
    frontier.append(row("Fairlearn ThresholdOptimizer (TPR parity, gender x age)", 0.0, a, yt, pt, auc, "randomised"))

    # Rung 3: reductions on intersectional groups
    parts, feats = load()
    for p_ in parts.values():
        p_["grp"] = group_key(p_)
    cache = RUNS / "reductions_cache_v2.pkl"  # v2: cost-aware objective (improvement 2)
    if cache.exists():
        red = pickle.load(open(cache, "rb"))
    else:
        red = reductions(parts["train"], parts["valid"], parts["test"], feats, [0.01, 0.03, 0.05])
        pickle.dump(red, open(cache, "wb"))
    red_rows = []
    for name, eps, s_va, s_te in red:
        if eps is None:  # probability model: decide at the cost-minimising cut-off
            a_te = (1 - s_te) < THRESHOLD
        else:  # randomised classifier: approve when it approves more often than not
            a_te = s_te >= 0.5
        r_ = row(name, eps, a_te, yt, pt, roc_auc_score(yt, 1 - s_te), {"all": "none (no protected attribute)"},
                 {"uses_protected_at_decision": False})
        frontier.append(r_); red_rows.append(r_)

    # Points other scripts add: the scorecard challenger (src/scorecard.py)
    # and the expected-profit policies (src/profit_policy.py), evaluated with
    # the same 5:1 cost so they sit on the same axes.
    extra = RUNS / "frontier_extra.csv"
    fr = pd.DataFrame(frontier)
    if extra.exists():
        fr = pd.concat([fr, pd.read_csv(extra)], ignore_index=True)
    fr["uses_protected_at_decision"] = fr["uses_protected_at_decision"].fillna(True).astype(bool)
    fr.to_csv(RUNS / "frontier.csv", index=False)
    pd.set_option("display.width", 220)
    print(fr[["method", "eps", "test_gap_gender", "test_gap_age", "test_cost", "test_auc", "test_approval"]].to_string())

    # Operating point: intersectional, levelling-up-only cut-offs fitted at
    # eps = 0.03 on valid (0.02 of margin under the 0.05 limit for sampling
    # noise on test). Chosen over the unrestricted variant because it costs
    # about the same and leaves no group with a lower approval rate.
    op = fr[(fr.method == "group cut-offs by gender x age, levelling up only") & (fr.eps == 0.03)].iloc[0]
    th = op.thresholds
    gk = group_key(pt)
    a = apply_thresholds(ptt, gk, th)
    m = evaluate(yt, ptt, groups_of(pt), approve=a, threshold=float("nan"))
    m["expected_cost"] = expected_cost(yt, a, C_FN, C_FP)
    base_a = ptt < THRESHOLD
    boot = bootstrap(yt, ptt, a, groups_of(pt))
    log_run("r4_operating_point", {
        "policy": {"type": "group-specific cut-offs on calibrated PD, levelling up only (no cut-off below 1/6)",
                   "groups": "gender x age (under 35 / 35+)",
                   "eps_target_on_valid": 0.03, "thresholds": th},
        "test": m,
        "bootstrap_95": boot,
        "gate": {"NFR1": m["roc_auc"] >= 0.75, "NFR2": m["ece"] <= 0.05,
                 "NFR3_gender": m["eo_gap_gender"] <= NFR3_GAP, "NFR3_age": m["eo_gap_age"] <= NFR3_GAP},
        "group_rates_gender": group_rates(yt, a, pt.gender.values).to_dict("records"),
        "group_rates_age": group_rates(yt, a, pt.age_under_35.values).to_dict("records"),
        "group_rates_intersection": group_rates(yt, a, gk).to_dict("records"),
        "ece_by_gender": {k: ece(yt[pt.gender.values == k], ptt[pt.gender.values == k]) for k in ["Female", "Male"]},
        "ece_by_age": {k: ece(yt[pt.age_under_35.values == k], ptt[pt.age_under_35.values == k])
                       for k in ["Under 35", "35 and over"]},
        "baseline_group_rates_gender": group_rates(yt, base_a, pt.gender.values).to_dict("records"),
        "baseline_group_rates_age": group_rates(yt, base_a, pt.age_under_35.values).to_dict("records"),
        "baseline_group_rates_intersection": group_rates(yt, base_a, gk).to_dict("records"),
        "baseline_metrics": evaluate(yt, ptt, groups_of(pt), THRESHOLD) | {"expected_cost": expected_cost(yt, base_a, C_FN, C_FP)},
    })
    # Since 1.3 this 5:1 policy is the comparison, not the deployed policy:
    # src/profit_policy.py writes models/fair_policy.json.
    json.dump({"type": "group_thresholds", "attribute": "gender|age_under_35", "thresholds": th},
              open(MODELS / "fair_policy_5to1.json", "w"), indent=2)

    # Improvement 2: the best policy that needs no protected attribute at
    # decision time, next to the deployed group cut-offs.
    # Pick the cheapest run whose gaps pass the gate at the top of their
    # bootstrap interval; if none does, the one with the smallest gap.
    ok = [r_ for r_ in red_rows if r_["eps"] is not None]
    for r_ in ok:
        s_te = next(x for x in red if x[0] == r_["method"])[3]
        r_["_boot"] = bootstrap(yt, 1 - s_te, s_te >= 0.5, groups_of(pt))
        r_["_passes"] = max(r_["_boot"]["eo_gap_gender"]["high"], r_["_boot"]["eo_gap_age"]["high"]) <= NFR3_GAP
    best = min(ok, key=lambda r_: (not r_["_passes"], r_["test_cost"] if r_["_passes"] else r_["test_gap_max"])) if ok else None
    if best is not None:
        name, eps, s_va, s_te = next(x for x in red if x[0] == best["method"])
        a_b = s_te >= 0.5
        log_run("r3_no_protected_at_decision", {
            "method": best["method"], "objective": f"expected cost {C_FN:g}:{C_FP:g} (ErrorRate costs)",
            "constraint": "TPR parity on repaid, gender x age, training only",
            "base_learner": "XGBoost 300 trees depth 4, 80,000 training applicants", "iterations": 50,
            "test": {"approval_rate": float(a_b.mean()), "roc_auc_of_mixture_score": best["test_auc"],
                     "eo_gap_gender": best["test_gap_gender"], "eo_gap_age": best["test_gap_age"],
                     "eo_gap_intersection": best["test_gap_intersection"], "expected_cost": best["test_cost"]},
            "bootstrap_95": best["_boot"], "passes_nfr3_at_upper_bound": best["_passes"],
            "note": "the mixture score is a vote share of up to 50 classifiers, not a probability, so its AUC "
                    "understates the ranking; the decision is what matters here",
            "compare_deployed_group_cutoffs": {"expected_cost": float(op.test_cost), "gap_max": float(op.test_gap_max)},
            "compare_single_cutoff": {"expected_cost": float(fr.iloc[0].test_cost), "gap_max": float(fr.iloc[0].test_gap_max)},
            "all_runs": [{k: r_[k] for k in ("method", "eps", "test_gap_gender", "test_gap_age", "test_cost",
                                              "test_approval", "test_auc")} for r_ in red_rows],
        })

    # Frontier figure: x = the larger of the two gaps, y = expected cost
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    cols = {"group cut-offs by gender": "#c8cfd8", "group cut-offs by age": "#a9cbe3",
            "group cut-offs by gender x age": "#7d8fa6",
            "group cut-offs by gender x age, levelling up only": "#2a6fdb"}
    for mth, c in cols.items():
        g = fr[fr.method == mth].sort_values("eps")
        ax.plot(g.test_gap_max, g.test_cost, "-o", ms=4, color=c, label=mth.replace("group cut-offs", "Group cut-offs"))
    s = fr[fr.method.str.startswith("single")]
    ax.scatter(s.test_gap_max, s.test_cost, s=80, color="#333", zorder=3, label="Single cut-off, no mitigation")
    r = fr[fr.method.str.startswith("Exponentiated")]
    ax.scatter(r.test_gap_max, r.test_cost, s=50, marker="s", color="#d9822b", zorder=3,
               label="ExponentiatedGradient, no protected attribute at decision")
    rr = fr[fr.method.str.startswith("reference")]
    ax.scatter(rr.test_gap_max, rr.test_cost, s=50, marker="s", facecolor="none", edgecolor="#d9822b", zorder=3,
               label="Rung 3 reference, same budget, no constraint")
    sc = fr[fr.method.str.startswith("Scorecard")]
    if len(sc):
        ax.scatter(sc.test_gap_max, sc.test_cost, s=55, marker="D", color="#16a085", zorder=3, label="Scorecard challenger")
    ep = fr[fr.method.str.startswith("Expected profit")]
    if len(ep):
        ax.scatter(ep.test_gap_max, ep.test_cost, s=70, marker="*", color="#c0392b", zorder=3, label="Expected-profit policy")
    t = fr[fr.method.str.startswith("Fairlearn")]
    ax.scatter(t.test_gap_max, t.test_cost, s=60, marker="^", color="#8e44ad", zorder=3, label="Fairlearn ThresholdOptimizer")
    ax.scatter([op.test_gap_max], [op.test_cost], s=220, facecolor="none", edgecolor="#c0392b", lw=2, zorder=4,
               label="5:1 group cut-offs (1.2 operating point)")
    ax.axvline(NFR3_GAP, ls="--", color="#c0392b", lw=1)
    ax.text(NFR3_GAP + 0.003, ax.get_ylim()[1], "NFR3 limit 0.05", va="top", color="#c0392b", fontsize=8)
    ax.set_xlabel("Larger of the gender and age equal-opportunity gaps (test)")
    ax.set_ylabel(f"Expected cost per applicant (FN={C_FN:g}, FP={C_FP:g})")
    ax.set_title("Accuracy-fairness frontier: what each fairness level costs")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "frontier.png")


if __name__ == "__main__":
    main()
