"""Improvement 7: selection bias and reject inference.

application_train has outcomes only for loans that were granted, so the
model has never seen how refused applicants would have behaved. The only
refused applications in the data are in previous_application (290,678
refusals). This script

A. measures how different refused applications are from approved ones on
   the fields both have (amounts, term, contract type, price band), and how
   applicants with a past refusal repaid the loan they later got;
B. builds a reject sample: for each TRAIN-split applicant, their latest
   refused previous application, carried onto their own application-level
   features (loan amount, instalment and goods price from the refused
   application; employment and registration dates moved back to the refusal
   date; ratios recomputed). Approximation, stated in the report: personal
   characteristics are assumed unchanged between the two applications, and
   these people were later accepted, so they are not "never accepted";
C. tries two standard reject-inference methods on the application-level
   model (the credit-history tables would carry information from after the
   refusal, so they are left out of this experiment):
     parcelling    rejects get a default label drawn at the accepts' default
                   rate of their score band times an uplift (2x), then join
                   the training set;
     augmentation  accepts are reweighted by 1 / P(accepted | x) from an
                   accept-vs-reject classifier, so applicants like the
                   rejects count more;
   and reports, on the test split's accepts (the only outcomes there are),
   whether either changes the model, plus how each model scores the test
   split's own rejects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data import RAW, apply_cutoff
from metrics import ece
from train import BASE, SEED, THRESHOLD, load, log_run

HIST = ("BUR_", "PREV_", "POS_", "INS_", "CC_")
UPLIFT = 2.0
SHARED = ["AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_PAYMENT"]


def previous() -> pd.DataFrame:
    p = pd.read_csv(RAW / "previous_application.csv",
                    usecols=["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "DAYS_DECISION",
                             "NAME_YIELD_GROUP", "CODE_REJECT_REASON", *SHARED])
    p = apply_cutoff(p, "previous_application")
    return p[p.NAME_CONTRACT_STATUS.isin(["Approved", "Refused"])]


def smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


def compare(p: pd.DataFrame, tr: pd.DataFrame) -> dict:
    ref, app = p[p.NAME_CONTRACT_STATUS == "Refused"], p[p.NAME_CONTRACT_STATUS == "Approved"]
    out = {"n_refused": int(len(ref)), "n_approved": int(len(app)),
           "standardised_mean_difference": {c: smd(np.log1p(ref[c]), np.log1p(app[c])) for c in SHARED},
           "median_refused": {c: float(ref[c].median()) for c in SHARED},
           "median_approved": {c: float(app[c].median()) for c in SHARED},
           "contract_type_share": {s: d.NAME_CONTRACT_TYPE.value_counts(normalize=True).round(4).to_dict()
                                   for s, d in (("refused", ref), ("approved", app))},
           "price_band_share": {s: d.NAME_YIELD_GROUP.value_counts(normalize=True).round(4).to_dict()
                                for s, d in (("refused", ref), ("approved", app))},
           "reject_reason_share": ref.CODE_REJECT_REASON.value_counts(normalize=True).round(4).to_dict()}
    had = tr.SK_ID_CURR.isin(ref.SK_ID_CURR)
    out["default_rate_later_loan"] = {"with_a_past_refusal": float(tr.TARGET[had].mean()),
                                      "without": float(tr.TARGET[~had].mean()), "share_with_refusal": float(had.mean())}
    return out


def reject_rows(p: pd.DataFrame, base: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Latest refused cash/consumer application of each applicant in base,
    carried onto that applicant's application-level row."""
    ref = p[(p.NAME_CONTRACT_STATUS == "Refused") & p.SK_ID_CURR.isin(base.SK_ID_CURR)
            & p.NAME_CONTRACT_TYPE.isin(["Cash loans", "Consumer loans", "Revolving loans"])
            & (p.AMT_CREDIT > 0) & p.AMT_ANNUITY.notna()]
    ref = ref.sort_values("DAYS_DECISION").groupby("SK_ID_CURR").tail(1).set_index("SK_ID_CURR")
    r = base.set_index("SK_ID_CURR").loc[ref.index, feats].copy()
    r["AMT_CREDIT"], r["AMT_ANNUITY"] = ref.AMT_CREDIT.values, ref.AMT_ANNUITY.values
    r["AMT_GOODS_PRICE"] = ref.AMT_GOODS_PRICE.values
    revolving = (ref.NAME_CONTRACT_TYPE == "Revolving loans").values.astype("float32")
    if "NAME_CONTRACT_TYPE_Cash_loans" in r:
        r["NAME_CONTRACT_TYPE_Cash_loans"], r["NAME_CONTRACT_TYPE_Revolving_loans"] = 1 - revolving, revolving
    for c in ("DAYS_EMPLOYED", "DAYS_REGISTRATION"):
        if c in r:
            v = r[c].values - ref.DAYS_DECISION.values   # DAYS_DECISION < 0: move the date back
            r[c] = np.where(v >= 0, np.nan, v)
    r["CREDIT_INCOME_RATIO"] = r.AMT_CREDIT / r.AMT_INCOME_TOTAL
    r["ANNUITY_INCOME_RATIO"] = r.AMT_ANNUITY / r.AMT_INCOME_TOTAL
    r["CREDIT_TERM"] = r.AMT_ANNUITY / r.AMT_CREDIT
    r["GOODS_CREDIT_RATIO"] = r.AMT_GOODS_PRICE / r.AMT_CREDIT
    return r.reset_index()


PARAMS = {**BASE, "max_depth": 4, "min_child_weight": 5, "colsample_bytree": 0.3, "eta": 0.05}


def fit(X, y, w=None, rounds=500):
    return xgb.train(PARAMS, xgb.DMatrix(X, label=y, weight=w), num_boost_round=rounds)


def assess(name, b, va, te, rej_te, feats) -> dict:
    s_va, s_te = b.predict(xgb.DMatrix(va[feats])), b.predict(xgb.DMatrix(te[feats]))
    iso = IsotonicRegression(out_of_bounds="clip").fit(s_va, va.TARGET.values)
    p_te = iso.predict(s_te)
    p_rej = iso.predict(b.predict(xgb.DMatrix(rej_te[feats])))
    return {"model": name, "test_auc": float(roc_auc_score(te.TARGET.values, p_te)), "test_ece": ece(te.TARGET.values, p_te),
            "test_approval_at_1_6": float((p_te < THRESHOLD).mean()),
            "rejects_mean_pd": float(p_rej.mean()), "rejects_approval_at_1_6": float((p_rej < THRESHOLD).mean())}


def main() -> None:
    parts, all_feats = load()
    tr, va, te = parts["train"], parts["valid"], parts["test"]
    feats = [f for f in all_feats if not f.startswith(HIST)]
    p = previous()
    part_a = compare(p, tr)
    rej_tr, rej_te = reject_rows(p, tr, feats), reject_rows(p, te, feats)
    print(f"rejects: {len(rej_tr):,} train, {len(rej_te):,} test; application-level features: {len(feats)}")

    # How different are they? Accept-vs-reject classifier on application features
    X_ar = pd.concat([tr[feats], rej_tr[feats]], ignore_index=True)
    y_ar = np.r_[np.ones(len(tr)), np.zeros(len(rej_tr))]
    i_fit, i_ev = train_test_split(np.arange(len(y_ar)), test_size=0.25, stratify=y_ar, random_state=SEED)
    ar = xgb.train({**PARAMS, "eval_metric": "auc"}, xgb.DMatrix(X_ar.iloc[i_fit], label=y_ar[i_fit]), 300)
    ar_auc = float(roc_auc_score(y_ar[i_ev], ar.predict(xgb.DMatrix(X_ar.iloc[i_ev]))))

    ytr = tr.TARGET.values
    accepts = fit(tr[feats], ytr)
    res = [assess("accepts only (no reject inference)", accepts, va, te, rej_te, feats)]

    # Parcelling: score bands from the accepts-only model
    s_acc = accepts.predict(xgb.DMatrix(tr[feats]))
    s_rej = accepts.predict(xgb.DMatrix(rej_tr[feats]))
    edges = np.quantile(s_acc, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
    band_acc, band_rej = np.digitize(s_acc, edges[1:-1]), np.digitize(s_rej, edges[1:-1])
    bad = pd.Series(ytr).groupby(band_acc).mean()
    rate = np.clip(bad.reindex(band_rej).values * UPLIFT, 0, 1)
    y_rej = (np.random.default_rng(SEED).random(len(rate)) < rate).astype(int)
    parcel = fit(pd.concat([tr[feats], rej_tr[feats]], ignore_index=True), np.r_[ytr, y_rej])
    res.append(assess(f"parcelling (reject bad rate = {UPLIFT:g}x accepts' band rate)", parcel, va, te, rej_te, feats))

    # Augmentation: reweight accepts by 1 / P(accept | x)
    p_acc = np.clip(ar.predict(xgb.DMatrix(tr[feats])), 0.05, 1.0)
    w = 1 / p_acc; w = w / w.mean()
    augm = fit(tr[feats], ytr, w)
    res.append(assess("augmentation (accepts weighted by 1 / P(accept))", augm, va, te, rej_te, feats))
    for r in res:
        print(r)

    base = res[0]
    log_run("r12_reject_inference", {
        "part_a_how_different": part_a,
        "reject_sample": {"train": int(len(rej_tr)), "test": int(len(rej_te)),
                          "construction": "latest refused previous application per applicant, on that applicant's "
                                          "application-level features; amounts from the refused application; "
                                          "employment and registration dates moved back to the refusal date"},
        "accept_vs_reject_auc": ar_auc, "parcelling_uplift": UPLIFT,
        "parcelling_rejects_assigned_default_rate": float(y_rej.mean()),
        "augmentation_weight_p99": float(np.percentile(w, 99)),
        "models": res,
        "change_vs_accepts_only": [{"model": r["model"], "auc_change": r["test_auc"] - base["test_auc"],
                                    "ece_change": r["test_ece"] - base["test_ece"],
                                    "rejects_mean_pd_change": r["rejects_mean_pd"] - base["rejects_mean_pd"]} for r in res[1:]],
        "features": "application-level only (%d); history tables excluded from this experiment" % len(feats),
    })


if __name__ == "__main__":
    main()
