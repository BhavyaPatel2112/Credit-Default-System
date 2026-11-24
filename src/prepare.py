"""M2-M4: EDA tables, the ETH4 proxy audit, the feature matrix, and frozen,
checksummed splits.

Protected attributes (gender, age, family status) are kept *beside* the
feature matrix for auditing and are never model inputs.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import PROC, ROOT

REPORTS = ROOT / "reports"
SEED = 42

PROTECTED_RAW = ["CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS"]
# Direct derivations of a protected attribute are excluded with it.
DERIVED_FROM_PROTECTED = ["EMPLOYED_BIRTH_RATIO"]
# Improvement 5: features a lender could not defend as a reason on a referral
# notice, because the applicant cannot act on them or they say nothing about
# ability to repay (when the form was filled in, which contact details were
# given, which documents were ticked, how the applicant's social contacts
# repaid, opaque seller codes). api/features.py keeps the same list, checked
# in tests/test_api.py.
NOT_DEFENSIBLE = ["HOUR_APPR_PROCESS_START", "WEEKDAY_APPR_PROCESS_START", "DAYS_LAST_PHONE_CHANGE",
                  "DAYS_ID_PUBLISH", "FLAG_MOBIL", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
                  "FLAG_WORK_PHONE"]
NOT_DEFENSIBLE_PREFIXES = ("FLAG_DOCUMENT_", "OBS_", "DEF_", "PREV_HOUR_APPR_PROCESS_START",
                           "PREV_NFLAG_LAST_APPL_IN_DAY", "PREV_SELLERPLACE_AREA")


def not_defensible(columns) -> list[str]:
    return [c for c in columns if c in NOT_DEFENSIBLE or c.startswith(NOT_DEFENSIBLE_PREFIXES)]


def protected_frame(df: pd.DataFrame) -> pd.DataFrame:
    age = (-df["DAYS_BIRTH"] / 365.25)
    return pd.DataFrame({
        "SK_ID_CURR": df["SK_ID_CURR"].values,
        "gender": df["CODE_GENDER"].map({"F": "Female", "M": "Male"}).values,
        "age_years": age.round(1).values,
        "age_band": pd.cut(age, [0, 25, 35, 45, 55, 120], right=False,
                           labels=["<25", "25-34", "35-44", "45-54", "55+"]).astype(str).values,
        "age_under_35": np.where(age < 35, "Under 35", "35 and over"),
        "family_status": df["NAME_FAMILY_STATUS"].values,
    })


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["SK_ID_CURR", "TARGET"] + PROTECTED_RAW + DERIVED_FROM_PROTECTED)
    X = X.drop(columns=not_defensible(X.columns))
    cats = X.select_dtypes(exclude="number").columns.tolist()
    X = pd.get_dummies(X, columns=cats, dummy_na=True, dtype="float32")
    X.columns = [c.replace(" ", "_").replace(",", "").replace("[", "(").replace("]", ")").replace("<", "lt")
                 for c in X.columns]
    # drop constant columns
    nun = X.nunique(dropna=False)
    X = X.loc[:, nun > 1]
    return X.astype("float32")


def eda(df: pd.DataFrame, prot: pd.DataFrame) -> dict:
    y = df["TARGET"].values
    out = {"n_applicants": int(len(df)), "n_defaults": int(y.sum()), "base_rate": float(y.mean())}
    rates = {}
    for col in ["gender", "age_band", "family_status"]:
        t = pd.DataFrame({"g": prot[col].values, "y": y}).groupby("g")["y"].agg(["size", "mean", "sum"])
        rates[col] = {k: {"n": int(v["size"]), "defaults": int(v["sum"]), "default_rate": float(v["mean"])}
                      for k, v in t.iterrows()}
    out["default_rate_by_group"] = rates
    miss = df.isna().mean().sort_values(ascending=False)
    out["columns_over_50pct_missing"] = int((miss > 0.5).sum())
    out["top_missing"] = {k: float(v) for k, v in miss.head(10).items()}
    out["ext_source_missing"] = {c: float(df[c].isna().mean()) for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]}
    out["days_employed_sentinel_share"] = float(df["DAYS_EMPLOYED_ANOM"].mean())
    out["days_employed_sentinel_default_rate"] = float(df.loc[df.DAYS_EMPLOYED_ANOM == 1, "TARGET"].mean())
    out["days_employed_normal_default_rate"] = float(df.loc[df.DAYS_EMPLOYED_ANOM == 0, "TARGET"].mean())
    out["income_p99"] = float(df["AMT_INCOME_TOTAL"].quantile(0.99))
    out["income_max"] = float(df["AMT_INCOME_TOTAL"].max())
    out["has_bureau_history"] = float(df["BUR_COUNT"].notna().mean())
    out["has_previous_application"] = float(df["PREV_COUNT"].notna().mean())
    return out


def proxy_audit(X: pd.DataFrame, prot: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    targets = {
        "gender_male": (prot["gender"] == "Male").astype(float).values,
        "age_years": prot["age_years"].astype(float).values,
        "married": (prot["family_status"] == "Married").astype(float).values,
    }
    Z = X.astype("float64")
    Z = Z.fillna(Z.median())
    Zc = (Z - Z.mean()) / Z.std(ddof=0).replace(0, np.nan)
    rows = []
    for name, t in targets.items():
        tc = (t - t.mean()) / t.std()
        r = (Zc.mul(tc, axis=0)).mean()
        for feat, val in r.items():
            if abs(val) > 0.2:
                rows.append({"feature": feat, "protected": name, "r": float(val), "flagged": abs(val) > threshold})
    out = pd.DataFrame(rows, columns=["feature", "protected", "r", "flagged"])
    return out.sort_values("r", key=abs, ascending=False)


def checksum(ids: np.ndarray) -> str:
    return hashlib.sha256(np.sort(ids).astype("int64").tobytes()).hexdigest()


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROC / "applicants.parquet")
    prot = protected_frame(df)
    X = feature_matrix(df)
    print("feature matrix", X.shape)

    e = eda(df, prot)
    (REPORTS / "eda.json").write_text(json.dumps(e, indent=2))

    pa = proxy_audit(X, prot)
    pa.to_csv(REPORTS / "proxy_audit.csv", index=False)
    print(pa.to_string())

    strat = df["TARGET"].astype(str) + "_" + prot["gender"]
    ids = df["SK_ID_CURR"].values
    tr, tmp = train_test_split(ids, test_size=0.30, stratify=strat, random_state=SEED)
    strat_tmp = strat.set_axis(ids).loc[tmp]
    va, te = train_test_split(tmp, test_size=0.50, stratify=strat_tmp, random_state=SEED)
    splits = {"train": tr, "valid": va, "test": te}
    manifest = {}
    for k, v in splits.items():
        pd.Series(np.sort(v), name="SK_ID_CURR").to_csv(PROC / f"split_{k}.csv", index=False)
        manifest[k] = {"n": int(len(v)), "sha256": checksum(v)}
    manifest["seed"] = SEED
    manifest["stratified_on"] = "TARGET x gender"
    (PROC / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))

    X.insert(0, "SK_ID_CURR", ids)
    X["TARGET"] = df["TARGET"].values
    X.to_parquet(PROC / "features.parquet", index=False)
    prot.to_parquet(PROC / "protected.parquet", index=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
