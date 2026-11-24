"""M2 + M4: assemble the one-row-per-applicant modelling table from the seven
Home Credit tables, with a hard temporal cut-off at the application date.

Every DAYS_* / MONTHS_BALANCE column in Home Credit is relative to the
application date (0 = the day of application, negative = before). Anything
dated at or after 0 describes what happened *after* the decision and must
never reach the model (risk R-2). `apply_cutoff` enforces that, and
tests/test_leakage.py checks it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SENTINEL = 365243  # Home Credit's "not applicable" code for day counts

TRANSFORM_LOG: list[dict] = []


def log(step: str, detail: str, rows_affected: int | None = None) -> None:
    TRANSFORM_LOG.append({"step": step, "detail": detail, "rows_affected": rows_affected})
    print(f"[{step}] {detail}" + (f" ({rows_affected:,} rows)" if rows_affected is not None else ""))


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        if c.startswith("SK_ID"):
            df[c] = df[c].astype("int32")
        else:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def read(name: str, **kw) -> pd.DataFrame:
    return downcast(pd.read_csv(RAW / f"{name}.csv", **kw))


# --------------------------------------------------------------------------
# Leakage control (R-2)
# --------------------------------------------------------------------------
# Columns that date an event relative to the application. A record is kept
# only if its *defining* date column is strictly before the application.
CUTOFF_RULES = {
    "bureau": {"filter": ["DAYS_CREDIT"]},
    "bureau_balance": {"filter": ["MONTHS_BALANCE"]},
    "previous_application": {"filter": ["DAYS_DECISION"]},
    "POS_CASH_balance": {"filter": ["MONTHS_BALANCE"]},
    "credit_card_balance": {"filter": ["MONTHS_BALANCE"]},
    "installments_payments": {"filter": ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]},
}
# Date columns that may legitimately hold a *future* value in a pre-application
# record (e.g. a scheduled end date). A scheduled date is known at application
# time; an *observed* outcome date is not, so observed ones are nulled.
OBSERVED_DATE_COLS = {
    "bureau": ["DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"],
    "previous_application": ["DAYS_LAST_DUE", "DAYS_TERMINATION"],
}


def apply_cutoff(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Drop records dated at/after application; null observed outcome dates
    that fall after it. Returns a new frame."""
    n0 = len(df)
    mask = pd.Series(True, index=df.index)
    for col in CUTOFF_RULES[table]["filter"]:
        if col in df:
            # NaN dates (e.g. installment not yet paid) are unknown -> keep row
            # but those cannot be after-application facts by construction.
            mask &= ~(df[col] >= 0)
    out = df.loc[mask].copy()
    log("cutoff", f"{table}: dropped records dated on/after application", n0 - len(out))
    for col in OBSERVED_DATE_COLS.get(table, []):
        if col in out:
            fut = out[col] >= 0
            fut &= out[col] != SENTINEL
            n = int(fut.sum())
            out.loc[fut, col] = np.nan
            log("cutoff", f"{table}.{col}: observed date after application -> NaN", n)
    return out


# --------------------------------------------------------------------------
# Application table cleaning
# --------------------------------------------------------------------------
def clean_application(app: pd.DataFrame) -> pd.DataFrame:
    app = app.copy()
    n_xna = int((app["CODE_GENDER"] == "XNA").sum())
    app = app[app["CODE_GENDER"] != "XNA"]
    log("clean", "CODE_GENDER == 'XNA' rows removed (gender unknown, cannot be audited)", n_xna)

    sent = app["DAYS_EMPLOYED"] == SENTINEL
    app["DAYS_EMPLOYED_ANOM"] = sent.astype("int8")
    app.loc[sent, "DAYS_EMPLOYED"] = np.nan
    log("clean", "DAYS_EMPLOYED == 365243 decoded to NaN + flag DAYS_EMPLOYED_ANOM", int(sent.sum()))

    for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        app[f"{c}_MISSING"] = app[c].isna().astype("int8")
        log("clean", f"{c} missingness flag added", int(app[c].isna().sum()))

    # Domain ratios (all computable at application time)
    app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
    app["ANNUITY_INCOME_RATIO"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]
    app["CREDIT_TERM"] = app["AMT_ANNUITY"] / app["AMT_CREDIT"]
    app["GOODS_CREDIT_RATIO"] = app["AMT_GOODS_PRICE"] / app["AMT_CREDIT"]
    app["EMPLOYED_BIRTH_RATIO"] = app["DAYS_EMPLOYED"] / app["DAYS_BIRTH"]
    app["EXT_SOURCE_MEAN"] = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    app["EXT_SOURCE_PROD"] = app["EXT_SOURCE_1"] * app["EXT_SOURCE_2"] * app["EXT_SOURCE_3"]
    log("feature", "7 application-level ratio features added")
    return app


def agg_numeric(df: pd.DataFrame, by: str, prefix: str, aggs=("mean", "max", "min", "sum")) -> pd.DataFrame:
    num = df.drop(columns=[c for c in df.columns if c.startswith("SK_ID") and c != by])
    num = num.select_dtypes("number")
    g = num.groupby(by).agg(list(aggs))
    g.columns = [f"{prefix}_{a}_{b}".upper() for a, b in g.columns]
    g[f"{prefix}_COUNT".upper()] = df.groupby(by).size()
    return g.astype("float32")


def agg_categorical(df: pd.DataFrame, by: str, prefix: str, cols: list[str]) -> pd.DataFrame:
    d = pd.get_dummies(df[[by] + cols], columns=cols, dummy_na=False, dtype="float32")
    g = d.groupby(by).mean()
    g.columns = [f"{prefix}_{c}_SHARE".upper() for c in g.columns]
    return g


def build_bureau() -> pd.DataFrame:
    bb = read("bureau_balance", dtype={"STATUS": "category"})
    bb = apply_cutoff(bb, "bureau_balance")
    bb["STATUS_LATE"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("float32")
    bb["STATUS_SEVERE"] = bb["STATUS"].isin(["3", "4", "5"]).astype("float32")
    bbg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "size"),
        BB_LATE_SHARE=("STATUS_LATE", "mean"),
        BB_SEVERE_SHARE=("STATUS_SEVERE", "mean"),
    )
    del bb
    b = read("bureau")
    b = apply_cutoff(b, "bureau")
    b = b.join(bbg, on="SK_ID_BUREAU")
    b["DEBT_CREDIT_RATIO"] = b["AMT_CREDIT_SUM_DEBT"] / b["AMT_CREDIT_SUM"].replace(0, np.nan)
    num = agg_numeric(b.drop(columns=["SK_ID_BUREAU"]), "SK_ID_CURR", "BUR", ("mean", "max", "sum"))
    cat = agg_categorical(b, "SK_ID_CURR", "BUR", ["CREDIT_ACTIVE", "CREDIT_TYPE"])
    active = b[b["CREDIT_ACTIVE"] == "Active"]
    act = agg_numeric(active[["SK_ID_CURR", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT"]],
                      "SK_ID_CURR", "BUR_ACTIVE", ("mean", "sum"))
    out = num.join(cat).join(act)
    log("aggregate", f"bureau + bureau_balance -> {out.shape[1]} features")
    return out


def build_previous() -> pd.DataFrame:
    p = read("previous_application")
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        n = int((p[c] == SENTINEL).sum())
        p.loc[p[c] == SENTINEL, c] = np.nan
        log("clean", f"previous_application.{c} sentinel 365243 -> NaN", n)
    p = apply_cutoff(p, "previous_application")
    p["APP_CREDIT_RATIO"] = p["AMT_APPLICATION"] / p["AMT_CREDIT"].replace(0, np.nan)
    num = agg_numeric(p.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "PREV", ("mean", "max", "min"))
    cat = agg_categorical(p, "SK_ID_CURR", "PREV", ["NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_YIELD_GROUP"])
    out = num.join(cat)
    log("aggregate", f"previous_application -> {out.shape[1]} features")
    return out


def build_pos() -> pd.DataFrame:
    pos = read("POS_CASH_balance")
    pos = apply_cutoff(pos, "POS_CASH_balance")
    pos["DPD_POS"] = (pos["SK_DPD"] > 0).astype("float32")
    out = agg_numeric(pos[["SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
                           "SK_DPD", "SK_DPD_DEF", "DPD_POS"]], "SK_ID_CURR", "POS", ("mean", "max"))
    log("aggregate", f"POS_CASH_balance -> {out.shape[1]} features")
    return out


def build_installments() -> pd.DataFrame:
    ins = read("installments_payments")
    ins = apply_cutoff(ins, "installments_payments")
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"].replace(0, np.nan)
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    ins["LATE"] = (ins["DPD"] > 0).astype("float32")
    out = agg_numeric(ins[["SK_ID_CURR", "PAYMENT_PERC", "PAYMENT_DIFF", "DPD", "DBD", "LATE",
                           "AMT_INSTALMENT", "AMT_PAYMENT"]], "SK_ID_CURR", "INS", ("mean", "max", "sum"))
    log("aggregate", f"installments_payments -> {out.shape[1]} features")
    return out


def build_credit_card() -> pd.DataFrame:
    cc = read("credit_card_balance")
    cc = apply_cutoff(cc, "credit_card_balance")
    cc["UTIL"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    out = agg_numeric(cc[["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "UTIL",
                          "AMT_DRAWINGS_CURRENT", "SK_DPD", "SK_DPD_DEF", "CNT_DRAWINGS_CURRENT"]],
                      "SK_ID_CURR", "CC", ("mean", "max"))
    log("aggregate", f"credit_card_balance -> {out.shape[1]} features")
    return out


def build() -> pd.DataFrame:
    PROC.mkdir(parents=True, exist_ok=True)
    app = read("application_train")
    log("load", f"application_train: {app.shape[0]:,} rows x {app.shape[1]} cols, "
                f"{int(app.TARGET.sum()):,} defaults ({app.TARGET.mean():.4%})")
    app = clean_application(app)
    df = app.set_index("SK_ID_CURR")
    for builder in (build_bureau, build_previous, build_pos, build_installments, build_credit_card):
        df = df.join(builder())
    df = df.replace([np.inf, -np.inf], np.nan)
    assert df.index.is_unique, "join produced duplicate applicants"
    log("assemble", f"modelling table: {df.shape[0]:,} applicants x {df.shape[1]} columns")
    df.reset_index().to_parquet(PROC / "applicants.parquet", index=False)
    (PROC / "transformation_log.json").write_text(json.dumps(TRANSFORM_LOG, indent=2))
    return df


if __name__ == "__main__":
    build()
