"""Builds data/processed/demo_applicants.parquet: 400 real TEST-split
applicants for the console's applicant picker, so a loan officer (or an
examiner) can open a real file instead of typing 30 fields.

Stratified on policy group x saved decision x actual outcome, so the set
contains approvals and referrals, repayers and defaulters, in every group.
Only test-split applicants are used: the model never trained on them.

Needs data/processed/features.parquet and protected.parquet
(python src/data.py && python src/prepare.py). The file is derived from the
Kaggle data, so like data/raw it must not be published.
"""
import json

import numpy as np
import pandas as pd

import policy as P
from data import PROC, ROOT
from train import SEED


def main(n: int = 400) -> None:
    feats = json.load(open(ROOT / "models" / "features.json"))
    pol = json.load(open(ROOT / "models" / "fair_policy.json"))
    pt = pd.read_parquet(ROOT / "runs" / "pred_test.parquet")
    prot = pd.read_parquet(PROC / "protected.parquet")[["SK_ID_CURR", "age_years"]]
    pt = pt.merge(prot, on="SK_ID_CURR")
    key = pd.Series(P.group_key(pt.gender.values, pt.age_under_35.values), index=pt.index)
    pt["approved"] = P.approve(pol, pt.p_cal.values, key.values, pt.AMT_CREDIT.values, pt.AMT_ANNUITY.values)
    strata = key + "|" + pt.approved.astype(str) + "|" + pt.TARGET.astype(str)
    # equal share per stratum (capped by its size), so rare cases are represented
    per = int(np.ceil(n / strata.nunique()))
    pick = pt.groupby(strata, group_keys=False).apply(
        lambda g: g.sample(n=min(per, len(g)), random_state=SEED))
    pick = pick.sample(frac=1, random_state=SEED).head(n)
    F = pd.read_parquet(PROC / "features.parquet", columns=["SK_ID_CURR"] + feats)
    out = pick[["SK_ID_CURR", "gender", "age_years", "TARGET", "p_cal"]].merge(F, on="SK_ID_CURR")
    assert {"AMT_CREDIT", "AMT_ANNUITY"} <= set(out.columns)
    out.to_parquet(PROC / "demo_applicants.parquet", index=False)
    print(f"demo applicants: {len(out)} -> {PROC / 'demo_applicants.parquet'}")
    print(out.groupby(["gender"]).size().to_dict(), "defaulted:", int(out.TARGET.sum()))


if __name__ == "__main__":
    main()
