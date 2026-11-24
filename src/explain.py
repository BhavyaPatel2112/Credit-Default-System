"""FR2 + FR3: SHAP explanations and DiCE counterfactual recourse.

Recourse only varies what an applicant can actually change at the counter:
the loan amount (AMT_CREDIT) and the instalment (AMT_ANNUITY, i.e. the term).
Gender, age, family status and every bureau/history feature are immutable.
Derived ratios are recomputed from the changed amounts, so a counterfactual
is scored exactly as the deployed policy would score it.
"""
from __future__ import annotations

import json
import pickle
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

import policy as P
from train import MODELS, RUNS, SEED, load

warnings.filterwarnings("ignore")
FIG = RUNS.parent / "reports" / "figures"
ACTIONABLE = ["AMT_CREDIT", "AMT_ANNUITY"]


class Scorer:
    """Booster + isotonic calibrator + the deployed policy (src/policy.py)."""
    def __init__(self):
        self.booster = xgb.Booster(); self.booster.load_model(MODELS / "xgb.json")
        self.iso = pickle.load(open(MODELS / "calibrator.pkl", "rb"))
        self.features = json.load(open(MODELS / "features.json"))
        self.policy = json.load(open(MODELS / "fair_policy.json"))

    @staticmethod
    def recompute(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]
        X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]
        X["CREDIT_TERM"] = X["AMT_ANNUITY"] / X["AMT_CREDIT"]
        X["GOODS_CREDIT_RATIO"] = X["AMT_GOODS_PRICE"] / X["AMT_CREDIT"]
        return X

    def pd_(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(xgb.DMatrix(X[self.features].astype("float32")))
        return self.iso.predict(raw)

    @staticmethod
    def key(df: pd.DataFrame) -> np.ndarray:
        return P.group_key(df["gender"].values, df["age_under_35"].values)

    def approve(self, X: pd.DataFrame, keys) -> np.ndarray:
        # under the expected-profit policy the cut-off moves with the loan,
        # so a counterfactual with a smaller loan is judged on its own cut-off
        return P.approve(self.policy, self.pd_(X), keys, X["AMT_CREDIT"].values, X["AMT_ANNUITY"].values)


def shap_global(sc: Scorer, te: pd.DataFrame, n=4000):
    s = te.sample(n=min(n, len(te)), random_state=SEED)
    expl = shap.TreeExplainer(sc.booster)
    sv = expl.shap_values(s[sc.features])
    imp = pd.Series(np.abs(sv).mean(0), index=sc.features).sort_values(ascending=False)
    imp.head(30).to_csv(RUNS / "shap_global.csv", header=["mean_abs_shap"])
    fig = plt.figure()
    shap.summary_plot(sv, s[sc.features], max_display=15, show=False, plot_size=(8, 6))
    plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=160); plt.close("all")
    # per-decision top-5 coverage check (FR2) on the whole sample
    top5 = np.argsort(-np.abs(sv), axis=1)[:, :5]
    coverage = float((np.abs(np.take_along_axis(sv, top5, 1)) > 0).all(1).mean())
    # local example
    i = int(np.argmax(sc.pd_(s)))
    ex = pd.DataFrame({"feature": np.array(sc.features)[top5[i]], "value": s[sc.features].iloc[i].values[top5[i]],
                       "shap_logodds": sv[i][top5[i]]})
    t0 = time.perf_counter()
    for _ in range(200):
        expl.shap_values(s[sc.features].iloc[[0]])
    per_call_ms = (time.perf_counter() - t0) / 200 * 1000
    return {"top_global": imp.head(15).to_dict(), "top5_coverage": coverage,
            "example_highest_risk": ex.to_dict("records"), "treeshap_ms_per_applicant": per_call_ms}


def recourse(sc: Scorer, tr: pd.DataFrame, te: pd.DataFrame, n=100):
    import dice_ml
    approve = sc.approve(te, Scorer.key(te))
    if not (~approve).any():
        return {"declines_sampled": 0}
    declined = te[~approve].sample(n=min(n, int((~approve).sum())), random_state=SEED)
    ref = tr[ACTIONABLE].sample(20000, replace=len(tr) < 20000, random_state=SEED).reset_index(drop=True)
    ref["outcome"] = np.r_[np.zeros(10000), np.ones(10000)].astype(int)  # placeholder labels: DiCE only needs ranges
    data = dice_ml.Data(dataframe=ref,
                        continuous_features=ACTIONABLE, outcome_name="outcome")
    rows, examples = [], []
    for _, r in declined.iterrows():
        base = r.to_frame().T.infer_objects()

        class One:
            def predict_proba(self_inner, Xa):
                X = pd.concat([base] * len(Xa), ignore_index=True)
                for c in ACTIONABLE:
                    X[c] = Xa[c].values.astype(float)
                X = Scorer.recompute(X)
                a = sc.approve(X, [f"{r.gender}|{r.age_under_35}"] * len(X)).astype(float)
                return np.c_[1 - a, a]
            def predict(self_inner, Xa):
                return self_inner.predict_proba(Xa)[:, 1].round().astype(int)

        m = dice_ml.Model(model=One(), backend="sklearn")
        exp = dice_ml.Dice(data, m, method="random")
        q = base[ACTIONABLE].astype(float).reset_index(drop=True)
        cr, ca = float(r.AMT_CREDIT), float(r.AMT_ANNUITY)
        ranges = {"AMT_CREDIT": [max(cr * 0.2, 45000.0), cr], "AMT_ANNUITY": [max(ca * 0.2, 2000.0), ca * 1.2]}
        ok = False; best = None; source = None
        # 1) DiCE (random method), hard time limit so an applicant with no
        #    counterfactual in range cannot stall the run.
        import signal
        def _timeout(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(8)
        try:
            res = exp.generate_counterfactuals(q, total_CFs=3, desired_class=1, permitted_range=ranges,
                                               features_to_vary=ACTIONABLE, random_seed=SEED, verbose=False)
            cf = res.cf_examples_list[0].final_cfs_df
            if cf is not None and len(cf):
                cf = cf[ACTIONABLE].astype(float)
                cf = cf[One().predict_proba(cf)[:, 1] == 1]
                if len(cf):
                    ok, source = True, "DiCE"
                    cf = cf.assign(credit_cut=1 - cf.AMT_CREDIT / cr)
                    best = cf.sort_values("credit_cut").iloc[0]
        except Exception:
            pass
        finally:
            signal.alarm(0)
        # 2) Exhaustive grid over the same actionable range: smallest loan
        #    cut (and a longer term) that the policy would approve.
        if not ok:
            cuts = np.linspace(0, 0.8, 17)
            terms = [1.0, 0.85, 0.7, 0.55]
            G = pd.DataFrame([{"AMT_CREDIT": cr * (1 - c), "AMT_ANNUITY": ca * (1 - c) * t}
                              for c in cuts for t in terms])
            G = G[(G.AMT_CREDIT >= ranges["AMT_CREDIT"][0]) & (G.AMT_ANNUITY >= ranges["AMT_ANNUITY"][0])]
            if len(G):
                hit = G[One().predict_proba(G)[:, 1] == 1]
                if len(hit):
                    ok, source = True, "grid"
                    best = hit.assign(credit_cut=1 - hit.AMT_CREDIT / cr).sort_values("credit_cut").iloc[0]
        rows.append({"SK_ID_CURR": int(r.SK_ID_CURR), "gender": r.gender, "age_group": r.age_under_35, "pd": float(sc.pd_(base)[0]),
                     "has_recourse": ok, "source": source,
                     "loan_cut_pct": float(best.credit_cut) if ok else np.nan})
        if ok and len(examples) < 5:
            examples.append({"SK_ID_CURR": int(r.SK_ID_CURR), "pd_now": float(sc.pd_(base)[0]),
                             "credit_now": cr, "annuity_now": ca,
                             "credit_cf": float(best.AMT_CREDIT), "annuity_cf": float(best.AMT_ANNUITY)})
    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "recourse.csv", index=False)
    return {"declines_sampled": len(df), "share_with_recourse": float(df.has_recourse.mean()),
            "median_loan_cut_pct": float(df.loan_cut_pct.median()),
            "found_by": df.source.value_counts().to_dict(),
            "share_with_recourse_by_gender": df.groupby("gender").has_recourse.mean().to_dict(),
            "share_with_recourse_by_age": df.groupby("age_group").has_recourse.mean().to_dict(),
            "pd_with_recourse_median": float(df[df.has_recourse].pd.median()) if df.has_recourse.any() else None,
            "pd_without_recourse_median": float(df[~df.has_recourse].pd.median()) if (~df.has_recourse).any() else None,
            "examples": examples}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    parts, _ = load()
    sc = Scorer()
    out = {"shap": shap_global(sc, parts["test"])}
    print(json.dumps(out["shap"], indent=1, default=float)[:1500])
    out["recourse"] = recourse(sc, parts["train"], parts["test"])
    print(json.dumps(out["recourse"], indent=1, default=float))
    (RUNS / "r5_explainability.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
