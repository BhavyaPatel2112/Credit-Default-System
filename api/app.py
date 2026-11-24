"""M7: scoring service. FR1 (calibrated PD + band), FR2 (reasons), ETH3
(no automatic decline), EU AI Act Art. 12 (every decision logged).

The request is the application form in raw form, not the 449 encoded model
columns. api/features.py validates it and builds the encoded row with the
same formulas as training, so a missing or malformed field is rejected with
a list of every problem instead of being silently scored.

Run:  uvicorn api.app:app --port 8000      (docs at /docs)
Decision log: logs/decisions.jsonl, or the path in CAP501_DECISION_LOG.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.features import (CATEGORICAL_OPTIONAL, CATEGORICAL_REQUIRED, NOT_USED, NOT_USED_PREFIXES,
                          NUMERIC_NULLABLE_REQUIRED, NUMERIC_REQUIRED, Contract, reason_group)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))
import policy as P  # noqa: E402  (one decision rule for training, gate and service)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


booster = xgb.Booster(); booster.load_model(MODELS / "xgb.json")
iso = pickle.load(open(MODELS / "calibrator.pkl", "rb"))
FEATURES: list[str] = json.load(open(MODELS / "features.json"))
POLICY = json.load(open(MODELS / "fair_policy.json"))
CONTRACT = Contract(FEATURES)
EXPLAINER = shap.TreeExplainer(booster)
BANDS = [(0.05, "A - low risk"), (0.10, "B"), (0.20, "C"), (0.35, "D"), (1.01, "E - high risk")]
MAX_ADVERSE_REASONS = 4  # US Regulation B: up to four principal reasons on an adverse action notice
VERSIONS = {"model_sha256": sha256(MODELS / "xgb.json")[:16],
            "calibrator_sha256": sha256(MODELS / "calibrator.pkl")[:16],
            "policy_sha256": sha256(MODELS / "fair_policy.json")[:16]}


def log_path() -> Path:
    return Path(os.environ.get("CAP501_DECISION_LOG", ROOT / "logs" / "decisions.jsonl"))


app = FastAPI(title="CAP 501 credit scoring", version="1.3")


class Applicant(BaseModel):
    application: dict[str, float | int | str | None] = Field(
        ..., description="The application form: amounts, days, flags, and categories as text "
                         "(GET /schema lists every field and allowed value)")
    history: dict[str, float | int | None] = Field(
        default_factory=dict, description="Optional credit-history aggregates (BUR_, PREV_, POS_, INS_, CC_). "
                                          "Leave empty for a thin-file applicant")
    gender: str = Field(..., description="Female/Male. Used ONLY to select the policy group's offset (ETH2), never as a model input")
    age_years: float = Field(..., ge=18, le=100, description="Used ONLY to select the policy group's offset, never as a model input")


class Decision(BaseModel):
    decision_id: str
    probability_of_default: float
    band: str
    decision: str
    cutoff_applied: float
    expected_profit: float | None = None
    needs_human_review: bool
    adverse_action_reasons: list[dict]
    top_factors: list[dict]
    data_quality: dict
    versions: dict
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES), "policy": POLICY, "versions": VERSIONS}


def policy_groups() -> list[str]:
    return sorted(P.groups(POLICY))


@app.get("/schema")
def schema():
    """Everything a client (or a UI) needs to build a valid request."""
    cat = lambda gs: {g: [c for c in CONTRACT.categories[g] if c != "nan"] for g in gs}  # noqa: E731
    return {"required_numbers": NUMERIC_REQUIRED, "required_but_nullable": NUMERIC_NULLABLE_REQUIRED,
            "required_categories": cat(CATEGORICAL_REQUIRED), "optional_categories": cat(CATEGORICAL_OPTIONAL),
            "optional_numbers": sorted(CONTRACT.numeric_optional),
            "not_used_ignored_if_sent": sorted(NOT_USED) + [p + "*" for p in NOT_USED_PREFIXES],
            "history_features": CONTRACT.history, "policy_groups": policy_groups(), "policy_type": POLICY.get("type")}


def explain(row: np.ndarray, approve: bool) -> tuple[list[dict], list[dict]]:
    """Sum SHAP values within each reason group (they are additive), then
    rank. top_factors is for the loan officer (both directions);
    adverse_action_reasons is what a referred applicant is told."""
    sv = EXPLAINER.shap_values(row.reshape(1, -1).astype("float32"))[0]
    groups: dict[str, dict] = {}
    for i in np.flatnonzero(sv):
        lab = reason_group(FEATURES[i])
        g = groups.setdefault(lab, {"reason": lab, "shap_logodds": 0.0, "features": []})
        g["shap_logodds"] += float(sv[i])
        g["features"].append((FEATURES[i], float(sv[i])))
    for g in groups.values():
        # A history group whose columns are all empty means "no record on
        # file" (a thin-file applicant); say so rather than imply a bad record.
        if all(np.isnan(row[CONTRACT.index[f]]) for f, _ in g["features"]) and \
                any(f.startswith(("BUR_", "PREV_", "POS_", "INS_", "CC_")) for f, _ in g["features"]):
            g["reason"] += ": none on file"
        g["features"] = [f for f, _ in sorted(g["features"], key=lambda t: -abs(t[1]))[:3]]
        g["effect"] = "raises risk" if g["shap_logodds"] > 0 else "lowers risk"
    top = sorted(groups.values(), key=lambda g: -abs(g["shap_logodds"]))[:5]
    adverse = [] if approve else [g for g in sorted(groups.values(), key=lambda g: -g["shap_logodds"])
                                  if g["shap_logodds"] > 0][:MAX_ADVERSE_REASONS]
    return adverse, top


def log_decision(rec: dict) -> None:
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")


def problems(a: Applicant) -> list[str]:
    key = P.key_of(a.gender, a.age_years)
    errors = [] if key in P.groups(POLICY) or "all" in P.groups(POLICY) else \
        [f"gender: unknown policy group {key!r}, one of {policy_groups()}"]
    return errors + CONTRACT.validate(a.application, a.history)


def cutoff_for(key: str, application: dict) -> float:
    return P.cutoff_one(POLICY, key, float(application["AMT_CREDIT"]), float(application["AMT_ANNUITY"]))


def decide(a: Applicant, source: str = "score") -> Decision:
    """Score one VALID application (call problems() first), explain it and
    log it. Shared by POST /score and the batch endpoint, so a batch row is
    decided and recorded exactly like a single decision."""
    t0 = time.perf_counter()
    key = P.key_of(a.gender, a.age_years)
    row, notes = CONTRACT.encode(a.application, a.history)
    raw = booster.predict(xgb.DMatrix(row.reshape(1, -1).astype("float32"), feature_names=FEATURES))
    p = float(iso.predict(raw)[0])
    cut = cutoff_for(key, a.application)
    approve = p < cut
    profit = None
    if POLICY.get("type") == "expected_profit":
        profit = float(P.expected_profit(p, a.application["AMT_CREDIT"], a.application["AMT_ANNUITY"],
                                         POLICY["economics"]))
    adverse, top = explain(row, approve)
    band = next(b for lim, b in BANDS if p < lim)
    did = uuid.uuid4().hex
    out = Decision(decision_id=did, probability_of_default=p, band=band,
                   decision="approve" if approve else "refer", cutoff_applied=cut, expected_profit=profit,
                   # ETH3: no automatic decline - every non-approval goes to a loan officer
                   needs_human_review=not approve, adverse_action_reasons=adverse, top_factors=top,
                   data_quality=notes, versions=VERSIONS, latency_ms=(time.perf_counter() - t0) * 1000)
    # Art. 12 record: what was decided, by which model and policy, and why.
    # Only a hash of the application is stored, so the log holds no personal
    # data beyond the policy group.
    payload = json.dumps({"application": a.application, "history": a.history}, sort_keys=True, default=str)
    log_decision({"decision_id": did, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "source": source,
                  "input_sha256": hashlib.sha256(payload.encode()).hexdigest()[:16], "policy_group": key,
                  "probability_of_default": round(p, 6), "cutoff": round(cut, 6), "decision": out.decision,
                  "adverse_action_reasons": [g["reason"] for g in adverse], **VERSIONS})
    return out


@app.post("/score", response_model=Decision)
def score(a: Applicant):
    errors = problems(a)
    if errors:
        raise HTTPException(422, errors)
    return decide(a)


# ----------------------------------------------------------------------
# Web console: the /api endpoints behind it, and the built UI itself at "/".
# Registered last, so /health, /schema, /score and /docs keep priority.
# ----------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.batch import router as batch_router  # noqa: E402
from api.console import router as console_router  # noqa: E402

app.include_router(console_router)
app.include_router(batch_router)
UI_DIST = ROOT / "web" / "dist"
if UI_DIST.exists():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="console")
