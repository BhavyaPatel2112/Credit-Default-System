"""Improvement 6: stress test the portfolio and calculate the regulatory
capital each policy ties up.

Capital: Basel IRB risk-weight function for other retail exposures
(src/policy.py: irb_retail_capital), K per unit of exposure at the model's
PD, LGD 45%, EAD = loan amount. Minimum capital = K * EAD (8% of RWA).

Stress: a recession-style shock to every applicant's PD, as a log-odds shift
that keeps the model's ranking and scales the portfolio's mean PD by 1.5
(moderate) or 2.0 (severe), with a downturn LGD of 55% in the severe case
(Basel CRE36: downturn LGD must exceed the long-run average). Two views:
  book     the loans each policy approved in normal times, re-measured
           under stress: expected loss and capital rise;
  lending  the same decision rule applied to stressed PDs: who would still
           be approved, and what happens to the fairness gaps. Outcomes are
           the observed ones (no stressed outcomes exist), so the gaps show
           how the rule treats today's repayers under a stressed model.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

import policy as P
from fairness import apply_thresholds, fit_group_thresholds, group_key
from metrics import NFR3_GAP, eo_gap
from profit_policy import fit_policy
from train import MODELS, RUNS, THRESHOLD, log_run

FIG = RUNS.parent / "reports" / "figures"
SCENARIOS = [("baseline", 1.0, P.LGD), ("moderate recession", 1.5, P.LGD), ("severe recession", 2.0, 0.55)]


def shift_for(p: np.ndarray, factor: float) -> float:
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return 0.0 if factor == 1 else float(brentq(lambda s: expit(z + s).mean() - factor * p.mean(), 0, 5))


def book(a: np.ndarray, pd_: np.ndarray, ead: np.ndarray, lgd: float) -> dict:
    k = P.irb_retail_capital(pd_[a], lgd)
    cap = float((k * ead[a]).sum())
    return {"approved": int(a.sum()), "exposure": float(ead[a].sum()),
            "expected_loss": float((pd_[a] * lgd * ead[a]).sum()),
            "capital": cap, "rwa": cap * 12.5, "capital_per_exposure": cap / float(ead[a].sum()),
            "mean_pd_approved": float(pd_[a].mean())}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pv = pd.read_parquet(RUNS / "pred_valid.parquet")
    pt = pd.read_parquet(RUNS / "pred_test.parquet")
    y, p, ead, ann = pt.TARGET.values, pt.p_cal.values, pt.AMT_CREDIT.values.astype(float), pt.AMT_ANNUITY.values
    keys = group_key(pt)
    econ = dict(P.ECONOMICS)
    th = fit_group_thresholds(pv.p_cal.values, pv.TARGET.values, group_key(pv), 0.03,
                              grid=np.linspace(THRESHOLD, 0.6, 900))
    deployed = json.load(open(MODELS / "fair_policy.json"))
    rules = {
        "single cut-off 1/6": lambda q: q < THRESHOLD,
        "5:1 group cut-offs (1.2)": lambda q: apply_thresholds(q, keys, th),
        "expected profit, no fairness": lambda q: P.approve(fit_policy(pv, econ, False), q, keys, ead, ann),
        "expected profit + offsets (deployed)": lambda q: P.approve(deployed, q, keys, ead, ann),
    }
    base_a = {n: f(p) for n, f in rules.items()}

    out = {"scenarios": [], "lgd": P.LGD}
    for name, factor, lgd in SCENARIOS:
        s = shift_for(p, factor)
        q = expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) + s)
        sc = {"scenario": name, "pd_factor": factor, "logodds_shift": s, "lgd": lgd, "mean_pd": float(q.mean()),
              "policies": {}}
        for n, f in rules.items():
            a_new = f(q)
            sc["policies"][n] = {
                "book": book(base_a[n], q, ead, lgd),
                "lending": {"approval_rate": float(a_new.mean()),
                            "eo_gap_gender": eo_gap(y, a_new, pt.gender.values),
                            "eo_gap_age": eo_gap(y, a_new, pt.age_under_35.values),
                            "eo_gap_intersection": eo_gap(y, a_new, keys),
                            "within_limit": max(eo_gap(y, a_new, pt.gender.values),
                                                eo_gap(y, a_new, pt.age_under_35.values)) <= NFR3_GAP,
                            **{f"capital_{k}": v for k, v in book(a_new, q, ead, lgd).items()
                               if k in ("capital", "capital_per_exposure", "expected_loss")}},
            }
        out["scenarios"].append(sc)
        print(name, {n: (round(v["lending"]["approval_rate"], 3), round(v["book"]["capital"] / 1e6))
                     for n, v in sc["policies"].items()})

    b = out["scenarios"][0]["policies"]
    fair, plain = b["expected profit + offsets (deployed)"]["book"], b["expected profit, no fairness"]["book"]
    out["fairness_capital_effect"] = {
        "extra_capital": fair["capital"] - plain["capital"],
        "extra_capital_pct": (fair["capital"] / plain["capital"] - 1) * 100,
        "extra_exposure": fair["exposure"] - plain["exposure"],
        "capital_per_exposure_plain": plain["capital_per_exposure"],
        "capital_per_exposure_fair": fair["capital_per_exposure"],
    }
    # capital per group under the deployed policy
    a = base_a["expected profit + offsets (deployed)"]
    kcap = np.zeros(len(p)); kcap[a] = P.irb_retail_capital(p[a]) * ead[a]
    out["deployed_capital_by_group"] = {k: {"capital": float(kcap[keys == k].sum()),
                                            "capital_per_exposure": float(kcap[(keys == k) & a].sum() / ead[(keys == k) & a].sum())}
                                        for k in sorted(set(keys))}
    log_run("r11_stress_capital", out)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160)
    names = list(rules)
    x = np.arange(len(SCENARIOS))
    w = 0.2
    for i, n in enumerate(names):
        axs[0].bar(x + (i - 1.5) * w, [s["policies"][n]["book"]["capital"] / 1e9 for s in out["scenarios"]], w, label=n)
        axs[1].bar(x + (i - 1.5) * w, [max(s["policies"][n]["lending"]["eo_gap_gender"],
                                           s["policies"][n]["lending"]["eo_gap_age"]) * 100 for s in out["scenarios"]], w)
    for ax in axs:
        ax.set_xticks(x, [s[0] for s in SCENARIOS], fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("IRB capital for the approved book (billions)")
    axs[1].set_ylabel("Larger equal-opportunity gap (points)"); axs[1].axhline(NFR3_GAP * 100, ls="--", color="#c0392b", lw=1)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Stress test: capital tied up by each policy, and fairness of the same rule on stressed PDs", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "stress_capital.png")


if __name__ == "__main__":
    main()
