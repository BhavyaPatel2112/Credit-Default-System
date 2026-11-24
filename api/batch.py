"""Improvement 8: batch scoring. A CSV of applications in, every row
validated and scored by the same code as POST /score, out a summary, the
rejected rows with their problems in plain words, approval rates by group,
and a results CSV.

  POST /api/batch          {"csv": "<file text>", "filename": "..."}
  GET  /api/batch/sample   a ready-made CSV: real test applicants plus a few
                           deliberately broken rows, to see the validation

CSV format: one row per application, a header row of field names exactly as
in GET /schema (application fields, credit-history features), plus
`gender` (Female/Male) and `age_years`, which only select the policy group.
Optional: `applicant_id` (echoed back) and `TARGET` (a known outcome, 1 =
defaulted; never scored, only used to show repayer approval by group).
An empty cell means "not given"; for DAYS_EMPLOYED it means "no current
employment". Every scored row is written to the decision log with
source = "batch", exactly like a single decision.
"""
from __future__ import annotations

import ast
import csv
import io
import math
import re
from collections import Counter

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from api import app as core
from api.features import CATEGORICAL, NUMERIC_NULLABLE_REQUIRED, not_used

router = APIRouter(prefix="/api/batch")
MAX_ROWS = 5000
MAX_BYTES = 8_000_000
META = {"gender", "age_years", "applicant_id", "TARGET"}

LABELS = {
    "AMT_INCOME_TOTAL": "Annual income", "AMT_CREDIT": "Loan amount", "AMT_ANNUITY": "Instalment",
    "AMT_GOODS_PRICE": "Price of goods", "NAME_CONTRACT_TYPE": "Loan type", "NAME_INCOME_TYPE": "Income type",
    "DAYS_EMPLOYED": "Days in current job", "ORGANIZATION_TYPE": "Employer type", "OCCUPATION_TYPE": "Occupation",
    "NAME_EDUCATION_TYPE": "Education", "CNT_CHILDREN": "Children", "CNT_FAM_MEMBERS": "Family members",
    "NAME_HOUSING_TYPE": "Housing", "FLAG_OWN_REALTY": "Owns property", "FLAG_OWN_CAR": "Owns a car",
    "EXT_SOURCE_1": "Agency 1 score", "EXT_SOURCE_2": "Agency 2 score", "EXT_SOURCE_3": "Agency 3 score",
    "DAYS_REGISTRATION": "Days at registered address", "REGION_RATING_CLIENT": "Region rating",
    "REGION_RATING_CLIENT_W_CITY": "Region rating with city", "REGION_POPULATION_RELATIVE": "Region population",
    "FLAG_EMP_PHONE": "Employer phone given", "gender": "Gender", "age_years": "Age",
}


def _readable_choices(msg: str) -> str:
    """"one of ['Cash_loans', 'Revolving_loans']" -> "allowed: Cash loans, Revolving loans"."""
    m = re.search(r"one of (\[.*\])", msg)
    if not m:
        return msg
    try:
        values = [str(v).replace("_", " ") for v in ast.literal_eval(m.group(1))]
    except (ValueError, SyntaxError):
        return msg
    return msg[:m.start()] + "allowed: " + ", ".join(values)


def plain(problem: str) -> str:
    """'AMT_CREDIT=-5: must be positive' -> 'Loan amount (AMT_CREDIT) is -5: must be positive'."""
    problem = _readable_choices(problem)
    head, _, msg = problem.partition(": ")
    field, _, value = head.partition("=")
    field = field.removeprefix("history.")
    label = LABELS.get(field)
    if not label:
        return problem
    name = f"{label} ({field})"
    if msg.startswith("required"):
        return f"{name} is missing" + (msg[len("required"):] if "one of" in msg else "")
    if value:
        return f"{name} is {value}: {msg}"
    return f"{name}: {msg}"


class BatchIn(BaseModel):
    csv: str = Field(..., description="The CSV file's text, header row first")
    filename: str | None = None


def _number(field: str, text: str, errors: list[str]):
    try:
        v = float(text.replace(",", "")) if any(ch.isdigit() for ch in text) else float(text)
    except ValueError:
        errors.append(f"{field}: {text!r} is not a number")
        return None
    if not math.isfinite(v):
        errors.append(f"{field}: must be a finite number")
        return None
    return v


def parse_row(row: dict[str, str]) -> tuple[dict | None, list[str], dict]:
    """CSV row (all text) -> a /score request, or the problems with it."""
    errors: list[str] = []
    meta = {"applicant_id": (row.get("applicant_id") or "").strip() or None}
    application, history = {}, {}
    for k, raw in row.items():
        if k is None:  # more cells than header columns
            errors.append(f"row has {len(raw)} more value(s) than the header has columns")
            continue
        k, text = k.strip(), (raw or "").strip()
        if k in META or not k:
            continue
        if not_used(k):
            continue  # ignored, like POST /score
        target = history if k in core.CONTRACT.history else application
        if text == "":
            if k in NUMERIC_NULLABLE_REQUIRED:
                application[k] = None
            continue
        if k in CATEGORICAL:
            application[k] = text
        else:
            v = _number(k, text, errors)
            if v is not None:
                target[k] = v
    gender = (row.get("gender") or "").strip()
    if gender not in ("Female", "Male"):
        errors.append(f"gender: must be Female or Male, got {gender!r}" if gender else "gender: required")
    age = None
    age_text = (row.get("age_years") or "").strip()
    if not age_text:
        errors.append("age_years: required")
    else:
        age = _number("age_years", age_text, errors)
        if age is not None and not 18 <= age <= 100:
            errors.append(f"age_years={age_text}: must be between 18 and 100")
            age = None
    t = (row.get("TARGET") or "").strip()
    meta["target"] = int(float(t)) if t in ("0", "1", "0.0", "1.0") else None
    if errors:
        return None, errors, meta
    return {"application": application, "history": history, "gender": gender, "age_years": age}, [], meta


def score_csv(text: str) -> dict:
    if len(text.encode()) > MAX_BYTES:
        raise HTTPException(413, [f"file is larger than {MAX_BYTES // 1_000_000} MB"])
    text = text.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(422, ["the file is empty or has no header row"])
    header = [h.strip() for h in reader.fieldnames]
    dupes = [h for h, n in Counter(header).items() if n > 1]
    if dupes:
        raise HTTPException(422, [f"column {d!r} appears more than once in the header" for d in dupes])
    known = core.CONTRACT.application_keys | set(core.CONTRACT.history) | META
    unknown = [h for h in header if h and h not in known and not not_used(h)]
    ignored = [h for h in header if not_used(h)]
    missing_meta = [m for m in ("gender", "age_years") if m not in header]
    rows = list(reader)
    if not rows:
        raise HTTPException(422, ["the file has a header but no applications"])
    if len(rows) > MAX_ROWS:
        raise HTTPException(413, [f"{len(rows):,} rows; the limit is {MAX_ROWS:,} per file"])

    results, rejected = [], []
    for i, row in enumerate(rows, start=1):
        line = i + 1  # the header is line 1
        req, errs, meta = parse_row(row)
        if req is not None:
            a = core.Applicant(**req)
            errs = core.problems(a)
        if errs:
            rejected.append({"row": i, "line": line, "applicant_id": meta["applicant_id"],
                             "problems": [plain(e) for e in errs]})
            results.append({"row": i, "applicant_id": meta["applicant_id"], "status": "rejected",
                            "problems": "; ".join(plain(e) for e in errs)})
            continue
        d = core.decide(a, source="batch")
        results.append({"row": i, "applicant_id": meta["applicant_id"], "status": "scored",
                        "group": core.P.key_of(a.gender, a.age_years),
                        "probability_of_default": round(d.probability_of_default, 6), "band": d.band,
                        "decision": d.decision, "cutoff": round(d.cutoff_applied, 6),
                        "expected_profit": None if d.expected_profit is None else round(d.expected_profit, 2),
                        "reasons": "; ".join(g["reason"] for g in d.adverse_action_reasons),
                        "decision_id": d.decision_id, "target": meta["target"]})

    scored = [r for r in results if r["status"] == "scored"]
    groups = []
    for k in sorted({r["group"] for r in scored}):
        g = [r for r in scored if r["group"] == k]
        rep = [r for r in g if r["target"] == 0]
        groups.append({"group": k, "scored": len(g), "approved": sum(r["decision"] == "approve" for r in g),
                       "approval_rate": float(np.mean([r["decision"] == "approve" for r in g])),
                       "repayers_known": len(rep),
                       "approval_among_repayers": float(np.mean([r["decision"] == "approve" for r in rep])) if rep else None,
                       "mean_pd": float(np.mean([r["probability_of_default"] for r in g]))})
    rates = [g["approval_rate"] for g in groups]
    rep_rates = [g["approval_among_repayers"] for g in groups if g["approval_among_repayers"] is not None
                 and g["repayers_known"] >= 30]

    out = io.StringIO()
    cols = ["row", "applicant_id", "status", "group", "probability_of_default", "band", "decision", "cutoff",
            "expected_profit", "reasons", "problems", "decision_id"]
    w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in results:
        w.writerow(r)

    return {
        "rows": len(rows), "scored": len(scored),
        "approved": sum(r["decision"] == "approve" for r in scored),
        "referred": sum(r["decision"] == "refer" for r in scored),
        "rejected": len(rejected),
        "file_notes": {"unknown_columns": unknown, "ignored_columns": ignored, "missing_columns": missing_meta},
        "rejected_rows": rejected[:500], "rejected_rows_truncated": len(rejected) > 500,
        "groups": groups,
        "approval_rate_gap": (max(rates) - min(rates)) if len(rates) > 1 else None,
        "repayer_approval_gap": (max(rep_rates) - min(rep_rates)) if len(rep_rates) > 1 else None,
        "results": results[:200],
        "results_csv": out.getvalue(),
        "versions": core.VERSIONS,
    }


@router.post("")
def batch(body: BatchIn):
    return score_csv(body.csv)


@router.get("/sample", response_class=PlainTextResponse)
def sample(n: int = 200):
    """Real held-out test applicants as a batch file, plus four broken rows
    (a missing income, a text value in a number column, an unknown loan
    type, an impossible age), so the validation can be seen at scale."""
    from api.console import _demo
    from api.features import request_from_encoded
    d = _demo()
    rows = []
    if d is not None:
        for _, r in d.head(n).iterrows():
            app_, hist = request_from_encoded(core.CONTRACT, {f: r[f] for f in core.FEATURES})
            rows.append({"applicant_id": str(int(r.SK_ID_CURR)), "gender": r.gender,
                         "age_years": round(float(r.age_years), 1), "TARGET": int(r.TARGET), **app_, **hist})
    else:
        from api.console import EXAMPLE_APPLICATION
        rows.append({"applicant_id": "example-1", "gender": "Female", "age_years": 41, **EXAMPLE_APPLICATION})
    good = rows[0]
    broken = [
        {**{k: v for k, v in good.items() if k != "AMT_INCOME_TOTAL"}, "applicant_id": "broken-no-income"},
        {**good, "applicant_id": "broken-text-in-number", "AMT_CREDIT": "four hundred thousand"},
        {**good, "applicant_id": "broken-loan-type", "NAME_CONTRACT_TYPE": "Mortgage"},
        {**good, "applicant_id": "broken-age", "age_years": 12, "EXT_SOURCE_2": 3.5},
    ]
    rows += broken
    cols = ["applicant_id", "gender", "age_years", "TARGET"]
    for r in rows:
        cols += [k for k in r if k not in cols]
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return PlainTextResponse(out.getvalue(), media_type="text/csv",
                             headers={"content-disposition": "attachment; filename=batch_sample.csv"})
