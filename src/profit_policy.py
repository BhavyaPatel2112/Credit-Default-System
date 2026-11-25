"""Improvement 1: decide on each loan's expected profit instead of a fixed
5:1 cost ratio, keep the fairness constraint on top, and compare with the
5:1 policies. Writes the deployed policy, models/fair_policy.json.

Approve loan i when its expected profit (src/policy.py)
    (1 - PD_i) * m_i * EAD_i - PD_i * LGD * EAD_i - c
is at least h_g, the offset of the applicant's gender x age group. h_g = 0
is the plain profit rule. Offsets are chosen on VALID, as the 5:1 group
cut-offs were: maximise the realised profit of the approved loans subject
to every group's approval rate among repayers lying within a band of width
eps = 0.03, and h_g <= 0 only (levelling up: no group is approved less often
than under the plain profit rule). Everything is reported on TEST.

Sensitivity: LGD 25% to 75% x margin kept 15% to 40%, each refitted on valid.

Run after train.py; before fairness.py (which plots the points written to
runs/frontier_extra.csv).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import policy as P
from fairness import apply_thresholds, fit_group_thresholds, group_key
from metrics import bootstrap, ece, eo_gap, expected_cost
from train import C_FN, C_FP, MODELS, RUNS, THRESHOLD, groups_of, log_run

EPS = 0.03


def fit_offsets(prof_hat: np.ndarray, real: np.ndarray, y: np.ndarray, g: np.ndarray, eps: float = EPS,
                n_grid: int = 600) -> dict | None:
    """Offset h_g <= 0 per group (approve when prof_hat > h_g) maximising the
    total realised profit, with every group's repayer approval rate within
    eps of a common level. Same search as fairness.fit_group_thresholds,
    with profit in place of cost."""
    keys = sorted(pd.unique(g))
    H, T, V = {}, {}, {}
    for k in keys:
        m = g == k
        ph, rp, yk = prof_hat[m], real[m], y[m]
        # candidate offsets: 0 and quantiles of the group's negative expected profits
        neg = ph[ph < 0]
        grid = np.unique(np.r_[0.0, np.quantile(neg, np.linspace(0, 1, n_grid)) - 1e-6 if len(neg) else []])
        grid = grid[grid <= 0]
        order = np.argsort(-ph)
        ph_s, rp_s, rep_s = ph[order], rp[order], (yk[order] == 0)
        # number approved for each h = count of prof_hat > h (sorted descending)
        n_ok = np.searchsorted(-ph_s, -grid, side="left")
        cum_rp = np.r_[0.0, np.cumsum(rp_s)]
        cum_rep = np.r_[0, np.cumsum(rep_s)]
        H[k], V[k], T[k] = grid, cum_rp[n_ok], cum_rep[n_ok] / max(rep_s.sum(), 1)
    best = (-np.inf, None)
    for tau in np.linspace(0.5, 1.0, 501):
        total, off = 0.0, {}
        for k in keys:
            ok = np.abs(T[k] - tau) <= eps / 2 + 1e-12
            if not ok.any():
                total = -np.inf; break
            v = np.where(ok, V[k], -np.inf)
            i = int(np.argmax(v)); total += v[i]; off[k] = float(H[k][i])
        if total > best[0]:
            best = (total, off)
    return best[1]


def summarise(name: str, approve: np.ndarray, d: pd.DataFrame, econ: dict, cut: np.ndarray | None = None,
              boot: bool = False) -> dict:
    y, p = d.TARGET.values, d.p_cal.values
    ead = d.AMT_CREDIT.values.astype(float)
    real = P.realised_profit(y, d.AMT_CREDIT.values, d.AMT_ANNUITY.values, econ)
    exp_ = P.expected_profit(p, d.AMT_CREDIT.values, d.AMT_ANNUITY.values, econ)
    a = approve
    out = {
        "policy": name, "approval_rate": float(a.mean()),
        "default_rate_among_approved": float(y[a].mean()),
        "realised_profit_total": float(real[a].sum()),
        "realised_profit_per_applicant": float(real[a].sum() / len(y)),
        "expected_profit_total": float(exp_[a].sum()),
        "expected_loss_total": float((p * econ["lgd"] * ead)[a].sum()),
        "exposure_approved": float(ead[a].sum()),
        "expected_cost_5to1": expected_cost(y, a, C_FN, C_FP),
        "eo_gap_gender": eo_gap(y, a, d.gender.values), "eo_gap_age": eo_gap(y, a, d.age_under_35.values),
        "eo_gap_intersection": eo_gap(y, a, group_key(d)),
    }
    if cut is not None:
        out["cutoff_quantiles"] = {q: float(np.quantile(cut, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    if boot:
        out["bootstrap_95"] = bootstrap(y, p, a, groups_of(d))
    return out


def frontier_row(method: str, eps, a: np.ndarray, d: pd.DataFrame, auc: float, th) -> dict:
    y = d.TARGET.values
    r = {"method": method, "eps": eps, "thresholds": json.dumps(th),
         "test_gap_gender": eo_gap(y, a, d.gender.values), "test_gap_age": eo_gap(y, a, d.age_under_35.values),
         "test_gap_intersection": eo_gap(y, a, group_key(d)), "test_cost": expected_cost(y, a, C_FN, C_FP),
         "test_auc": auc, "test_approval": float(a.mean()),
         "uses_protected_at_decision": any(v != 0 for v in th.get("offsets", {}).values())}
    r["test_gap_max"] = max(r["test_gap_gender"], r["test_gap_age"])
    return r


def fit_policy(pv: pd.DataFrame, econ: dict, fair: bool) -> dict:
    keys = group_key(pv)
    if not fair:
        return {"type": "expected_profit", "economics": econ, "offsets": {k: 0.0 for k in sorted(set(keys))}}
    ph = P.expected_profit(pv.p_cal.values, pv.AMT_CREDIT.values, pv.AMT_ANNUITY.values, econ)
    rp = P.realised_profit(pv.TARGET.values, pv.AMT_CREDIT.values, pv.AMT_ANNUITY.values, econ)
    off = fit_offsets(ph, rp, pv.TARGET.values, keys)
    return {"type": "expected_profit", "economics": econ, "offsets": off}


def decide(pol: dict, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cut = P.cutoffs(pol, group_key(d), d.AMT_CREDIT.values, d.AMT_ANNUITY.values)
    return d.p_cal.values < cut, cut


def main() -> None:
    pv = pd.read_parquet(RUNS / "pred_valid.parquet")
    pt = pd.read_parquet(RUNS / "pred_test.parquet")
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(pt.TARGET.values, pt.p_cal.values)
    econ = dict(P.ECONOMICS)

    # The two 5:1 policies of release 1.2, refitted on this model
    th = fit_group_thresholds(pv.p_cal.values, pv.TARGET.values, group_key(pv), EPS,
                              grid=np.linspace(THRESHOLD, 0.6, 900))
    a_single = pt.p_cal.values < THRESHOLD
    a_5to1 = apply_thresholds(pt.p_cal.values, group_key(pt), th)

    plain = fit_policy(pv, econ, fair=False)
    fair = fit_policy(pv, econ, fair=True)
    a_plain, c_plain = decide(plain, pt)
    a_fair, c_fair = decide(fair, pt)

    rows = [summarise("single cut-off 1/6 (5:1)", a_single, pt, econ, boot=True),
            summarise("group cut-offs, 5:1, levelling up (1.2 policy)", a_5to1, pt, econ, boot=True),
            summarise("expected profit, no fairness constraint", a_plain, pt, econ, c_plain, boot=True),
            summarise("expected profit + group offsets, levelling up (deployed)", a_fair, pt, econ, c_fair, boot=True)]
    for r in rows:
        print(f"{r['policy'][:58]:58s} approve {r['approval_rate']:.3f}  profit/app {r['realised_profit_per_applicant']:9.0f}"
              f"  gaps {r['eo_gap_gender']:.3f} {r['eo_gap_age']:.3f}")

    # How the cut-off moves with the loan's size and term (deployed policy)
    t = pt.assign(cut=c_fair, size=pd.qcut(pt.AMT_CREDIT, 5, labels=["smallest 20%", "2", "3", "4", "largest 20%"]),
                  term=pd.qcut(P.gross_interest(pt.AMT_CREDIT, pt.AMT_ANNUITY), 3, labels=["short", "medium", "long"]))
    by_size = t.groupby("size", observed=True).cut.median().to_dict()
    by_term = t.groupby("term", observed=True).cut.median().to_dict()

    # Sensitivity: refit on valid for every LGD x margin assumption
    sens = []
    for lgd in P.LGD_RANGE:
        for mk in P.MARGIN_KEPT_RANGE:
            e = {**econ, "lgd": lgd, "margin_kept": mk}
            for is_fair in (False, True):
                pol = fit_policy(pv, e, is_fair)
                if pol["offsets"] is None:
                    continue
                a, _ = decide(pol, pt)
                s = summarise("fair" if is_fair else "plain", a, pt, e)
                sens.append({"lgd": lgd, "margin_kept": mk, "fairness": is_fair,
                             **{k: s[k] for k in ("approval_rate", "default_rate_among_approved",
                                                  "realised_profit_per_applicant", "eo_gap_gender", "eo_gap_age")},
                             # what the 5:1 group policy would earn under the same economics
                             "profit_per_applicant_5to1_group_policy":
                                 summarise("", a_5to1, pt, e)["realised_profit_per_applicant"]})
    pd.DataFrame(sens).to_csv(RUNS / "expected_profit_sensitivity.csv", index=False)

    log_run("r8_expected_profit", {
        "rule": "approve when (1-PD)*m*EAD - PD*LGD*EAD - c >= h_group",
        "economics": econ,
        "sources": {"lgd": "Basel framework CRE32.6: foundation-IRB supervisory LGD 45% for senior unsecured claims",
                    "monthly_rate": "median implied rate of 312,536 approved previous cash loans (previous_application)",
                    "margin_kept": "stated assumption; sensitivity 15% to 40%",
                    "fixed_cost": "stated assumption, about 1% of the median loan"},
        "eps_target_on_valid": EPS, "offsets": fair["offsets"],
        "thresholds_5to1_refitted": th,
        "test": rows[-1], "comparison": rows,
        "median_cutoff_by_loan_size": by_size, "median_cutoff_by_term": by_term,
        "sensitivity_rows": len(sens),
    })
    json.dump(fair, open(MODELS / "fair_policy.json", "w"), indent=2)

    extra = [frontier_row("Expected profit, no fairness constraint", None, a_plain, pt, auc, {"offsets": plain["offsets"]}),
             frontier_row("Expected profit + group offsets (deployed)", EPS, a_fair, pt, auc, {"offsets": fair["offsets"]})]
    path = RUNS / "frontier_extra.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.method.str.startswith("Expected profit")]
    pd.concat([old, pd.DataFrame(extra)], ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    main()
