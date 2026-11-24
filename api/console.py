"""Endpoints behind the web console (web/). Everything here reads committed
artefacts (models/, runs/, reports/) or the optional demo applicant file;
nothing here changes the model or the policy.

  GET  /api/applicants            real test applicants for the picker
  GET  /api/applicants/{id}       one applicant as a ready-to-score request
  POST /api/recourse              smallest change to loan amount / term that gets approved
  GET  /api/fairness?cost_ratio=  single cut-off vs deployed policy, by group
  GET  /api/fairness/curve        the same summary across cost ratios 1 to 15
  GET  /api/frontier              the accuracy-fairness frontier from runs/frontier.csv
  GET  /api/model-card            metrics, run ladder, drivers, data facts, limitations
  GET  /api/gate                  the CI gate's checks, run live
  GET  /api/decisions             the most recent entries of the decision log
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, HTTPException, Query

from api import app as core
from api.features import reason_group

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEMO = ROOT / "data" / "processed" / "demo_applicants.parquet"
sys.path.insert(0, str(ROOT / "src"))
import policy as P  # noqa: E402
from metrics import NFR3_GAP, eo_gap  # noqa: E402

router = APIRouter(prefix="/api")
# The example application of demo/example_request.json, used by the batch
# sample when no demo applicants are built.
EXAMPLE_APPLICATION = json.loads((ROOT / "demo" / "example_request.json").read_text())["application"]


def _clean(o):
    """JSON-safe: NaN and infinities become null, numpy scalars become Python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _key(gender: str, age: float) -> str:
    return P.key_of(gender, age)


def _predict(rows: np.ndarray) -> np.ndarray:
    raw = core.booster.predict(xgb.DMatrix(rows.astype("float32"), feature_names=core.FEATURES))
    return core.iso.predict(raw)


def _band(p: float) -> str:
    return next(b for lim, b in core.BANDS if p < lim)


# ---------------------------------------------------------------- applicants
@lru_cache(maxsize=1)
def _demo() -> pd.DataFrame | None:
    if not DEMO.exists():
        return None
    d = pd.read_parquet(DEMO)
    d["key"] = [_key(g, a) for g, a in zip(d.gender, d.age_years)]
    d["cutoff"] = P.cutoffs(core.POLICY, d.key.values, d.AMT_CREDIT.values, d.AMT_ANNUITY.values)
    d["pd"] = _predict(d[core.FEATURES].values)
    return d


def _category(row: pd.Series, field: str) -> str | None:
    for lv in core.CONTRACT.categories[field]:
        if row.get(f"{field}_{lv}") == 1 and lv != "nan":
            return lv.replace("_", " ")
    return None


@router.get("/applicants")
def applicants():
    d = _demo()
    if d is None:
        return {"available": False, "applicants": [],
                "note": "No demo applicants yet. Run python src/make_demo_applicants.py after building the features."}
    out = []
    for _, r in d.iterrows():
        out.append({
            "id": int(r.SK_ID_CURR), "gender": r.gender, "age": int(r.age_years), "group": r.key,
            "loan": r.AMT_CREDIT, "annuity": r.AMT_ANNUITY, "income": r.AMT_INCOME_TOTAL,
            "contract": _category(r, "NAME_CONTRACT_TYPE"), "income_type": _category(r, "NAME_INCOME_TYPE"),
            "education": _category(r, "NAME_EDUCATION_TYPE"), "thin_file": bool(np.isnan(r.get("BUR_COUNT", np.nan))),
            "pd": r.pd, "band": _band(r.pd), "decision": "approve" if r.pd < r.cutoff else "refer",
            "actual": "defaulted" if r.TARGET == 1 else "repaid",
        })
    return _clean({"available": True, "applicants": out,
                   "note": "400 real applicants from the held-out test split, balanced for demonstration "
                           "(half defaulted; the real default rate is 8.1%)."})


@router.get("/applicants/{sid}")
def applicant(sid: int):
    d = _demo()
    if d is None:
        raise HTTPException(404, "demo applicants not built")
    m = d[d.SK_ID_CURR == sid]
    if m.empty:
        raise HTTPException(404, f"applicant {sid} is not in the demo set")
    r = m.iloc[0]
    from api.features import request_from_encoded
    app_, hist = request_from_encoded(core.CONTRACT, {f: r[f] for f in core.FEATURES})
    return _clean({"id": sid, "actual": "defaulted" if r.TARGET == 1 else "repaid",
                   "request": {"application": app_, "history": hist, "gender": r.gender,
                               "age_years": round(float(r.age_years), 1)}})


# ---------------------------------------------------------------- recourse
LOAN_CUTS = np.round(np.arange(0, 0.81, 0.05), 2)
TERMS = [1.0, 0.85, 0.7]  # instalment multiplier at the same loan: 1/0.7 = up to 43% longer term


@router.post("/recourse")
def recourse(a: core.Applicant):
    """FR3: what the applicant could change at the counter (loan amount and
    term) to be approved, everything else held fixed. Same search as
    src/explain.py's grid fallback, with the term capped at 43% longer."""
    key = _key(a.gender, a.age_years)
    errors = core.problems(a)
    if errors:
        raise HTTPException(422, errors)
    cut = core.cutoff_for(key, a.application)
    row, _ = core.CONTRACT.encode(a.application, a.history)
    p0 = float(_predict(row.reshape(1, -1))[0])
    cr, an = float(a.application["AMT_CREDIT"]), float(a.application["AMT_ANNUITY"])
    current = {"loan": cr, "annuity": an, "pd": p0, "cutoff": cut, "approved": p0 < cut}
    if p0 < cut:
        return _clean({"needed": False, "current": current, "options": []})
    grid, rows = [], []
    for t in TERMS:
        for c in LOAN_CUTS:
            new = {**a.application, "AMT_CREDIT": cr * (1 - c), "AMT_ANNUITY": an * (1 - c) * t}
            if not core.CONTRACT.validate(new, a.history):
                rows.append(core.CONTRACT.encode(new, a.history)[0]); grid.append((t, float(c), new))
    ps = _predict(np.vstack(rows)) if rows else np.array([])
    # Under the expected-profit policy the cut-off moves with the loan, so
    # each candidate is judged against its own cut-off.
    cuts = [core.cutoff_for(key, new) for _, _, new in grid]
    options = []
    for t in TERMS:  # cheapest loan cut for each term choice
        hits = [(c, new, p, ct) for (tt, c, new), p, ct in zip(grid, ps, cuts) if tt == t and p < ct]
        if hits:
            c, new, p, ct = min(hits, key=lambda h: h[0])
            options.append({"loan": new["AMT_CREDIT"], "annuity": new["AMT_ANNUITY"], "loan_cut_pct": c * 100,
                            "term_change_pct": (1 / t - 1) * 100, "pd": float(p), "cutoff": float(ct)})
    options.sort(key=lambda o: (o["loan_cut_pct"], o["term_change_pct"]))
    return _clean({"needed": True, "found": bool(options), "current": current, "options": options,
                   "note": None if options else "No smaller or longer loan within range gets below the cut-off; "
                                                "this application is for the loan officer to judge."})


# ---------------------------------------------------------------- fairness
@lru_cache(maxsize=1)
def _test():
    d = pd.read_parquet(RUNS / "pred_test.parquet")
    keys = P.group_key(d.gender.values, d.age_under_35.values)
    pol = P.cutoffs(core.POLICY, keys, d.get("AMT_CREDIT"), d.get("AMT_ANNUITY"))
    return d.TARGET.values, d.p_cal.values, d.gender.values, d.age_under_35.values, keys, pol


def _median_cutoffs() -> dict[str, float]:
    """One number per group for display: under the expected-profit policy
    every loan has its own cut-off, so the console shows the group median."""
    *_, keys, pol = _test()
    return {k: float(np.median(pol[keys == k])) for k in sorted(set(keys))}


def _summary(y, approve, gender, age, keys, c_fn: float) -> dict:
    groups = []
    for k in sorted(set(keys)):
        m = keys == k
        rep = m & (y == 0)
        groups.append({"group": k, "applicants": int(m.sum()), "repayers": int(rep.sum()),
                       "approval_among_repayers": float(approve[rep].mean()),
                       "approval_rate": float(approve[m].mean()),
                       "default_among_approved": float(y[m & approve].mean()) if (m & approve).any() else None,
                       "default_rate": float(y[m].mean())})
    return {"approval_rate": float(approve.mean()),
            "default_among_approved": float(y[approve].mean()),
            "defaulters_declined": float((~approve[y == 1]).mean()),
            "expected_cost": float((c_fn * ((y == 1) & approve) + ((y == 0) & ~approve)).mean()),
            "gap_gender": eo_gap(y, approve, gender), "gap_age": eo_gap(y, approve, age),
            "gap_intersection": eo_gap(y, approve, keys), "groups": groups}


@router.get("/fairness")
def fairness(cost_ratio: float = Query(5.0, ge=1.0, le=20.0)):
    """Approving a defaulter costs cost_ratio times declining a repayer, so the
    cost-minimising single cut-off is 1 / (1 + cost_ratio)."""
    y, p, g, a, k, pol = _test()
    cut = 1.0 / (1.0 + cost_ratio)
    single, policy = p < cut, p < pol
    added, removed = policy & ~single, single & ~policy
    return _clean({"cost_ratio": cost_ratio, "cutoff": cut, "limit": NFR3_GAP, "n": int(len(y)),
                   "single": _summary(y, single, g, a, k, cost_ratio),
                   "policy": {**_summary(y, policy, g, a, k, cost_ratio), "thresholds": _median_cutoffs(),
                              "type": core.POLICY.get("type")},
                   # who the deployed policy approves that the single cut-off would not, and how they fare
                   "added_by_policy": {"count": int(added.sum()),
                                       "default_rate": float(y[added].mean()) if added.any() else None,
                                       "mean_pd": float(p[added].mean()) if added.any() else None},
                   "removed_by_policy": int(removed.sum())})


@router.get("/fairness/curve")
@lru_cache(maxsize=1)
def fairness_curve():
    y, p, g, a, k, pol = _test()
    pts = []
    for r in np.round(np.arange(1.0, 15.01, 0.5), 2):
        ap = p < 1 / (1 + r)
        pts.append({"cost_ratio": float(r), "cutoff": 1 / (1 + r), "approval_rate": float(ap.mean()),
                    "gap_gender": eo_gap(y, ap, g), "gap_age": eo_gap(y, ap, a),
                    "default_among_approved": float(y[ap].mean())})
    return _clean({"limit": NFR3_GAP, "points": pts})


@router.get("/frontier")
def frontier():
    path = RUNS / "frontier.csv"
    if not path.exists():
        return {"points": []}
    f = pd.read_csv(path)
    pts = [{"method": r.method, "eps": r.eps, "gap": r.test_gap_max, "cost": r.test_cost,
            "approval": r.test_approval, "auc": r.test_auc,
            "uses_protected_at_decision": bool(getattr(r, "uses_protected_at_decision", True))} for r in f.itertuples()]
    deployed = "Expected profit + group offsets (deployed)"
    chosen = {"method": deployed, "eps": 0.03} if (f.method == deployed).any() else \
        {"method": "group cut-offs by gender x age, levelling up only", "eps": 0.03}
    return _clean({"points": pts, "chosen": chosen, "limit": NFR3_GAP})


# ---------------------------------------------------------------- model card
def _read(name):
    p = RUNS / name
    return json.loads(p.read_text().replace("NaN", "null")) if p.exists() else None


MODEL_METRICS = ("roc_auc", "pr_auc", "brier", "ece")


def _with_model_metrics(t: dict) -> dict:
    """A policy run reports decisions; the model's own metrics (AUC, ECE, ...)
    do not depend on the policy, so they come from the calibrated model run."""
    r2b = (_read("r2b_calibrated.json") or {}).get("test", {})
    return {**{k: r2b.get(k) for k in MODEL_METRICS}, **t}


@router.get("/model-card")
@lru_cache(maxsize=1)
def model_card():
    ladder = []
    for name, label in [("r0_application_only.json", "Application table only"),
                        ("r1_baseline_unweighted.json", "All seven tables"),
                        ("r2_weighted_tuned_uncalibrated.json", "Class weighting and tuning"),
                        ("r2b_calibrated.json", "Isotonic calibration"),
                        ("r4_operating_point.json", "Fairness cut-offs at 5:1 (1.2 policy)"),
                        ("r8_expected_profit.json", "Expected profit with fairness offsets (deployed)")]:
        r = _read(name)
        if r:
            t = _with_model_metrics(r["test"]) if name.startswith("r8") else r["test"]
            ladder.append({"run": name.split("_")[0], "label": label, "auc": t.get("roc_auc"),
                           "pr_auc": t.get("pr_auc"), "ece": t.get("ece"),
                           "gap_gender": t.get("eo_gap_gender"), "gap_age": t.get("eo_gap_age")})
    drivers = []
    sg = RUNS / "shap_global.csv"
    if sg.exists():
        s = pd.read_csv(sg, index_col=0).iloc[:, 0]
        grp = s.groupby([reason_group(f) for f in s.index]).sum().sort_values(ascending=False)
        drivers = [{"reason": k, "importance": v} for k, v in grp.items()]
    eda = json.loads((ROOT / "reports" / "eda.json").read_text()) if (ROOT / "reports" / "eda.json").exists() else {}
    r4 = _read("r4_operating_point.json") or {}
    r8 = _read("r8_expected_profit.json") or {}
    lat = _read("r7_latency.json") or {}
    r9, r10, r11, r12, r3 = (_read(n) or {} for n in ("r9_scorecard.json", "r10_group_calibration.json",
                                                       "r11_stress_capital.json", "r12_reject_inference.json",
                                                       "r3_no_protected_at_decision.json"))
    return _clean({
        "versions": core.VERSIONS, "n_features": len(core.FEATURES), "policy": _median_cutoffs(),
        "policy_type": core.POLICY.get("type"), "economics": core.POLICY.get("economics"),
        "offsets": core.POLICY.get("offsets"),
        "test": _with_model_metrics(r8["test"]) if r8 else r4.get("test", {}), "baseline": r4.get("baseline_metrics", {}), "ladder": ladder,
        "challenger": {k: (r9.get(k) or {}) for k in ("test", "champion_xgboost")} | {"gate_passed": r9.get("gate_passed"),
                                                                                      "method": r9.get("method")},
        "no_protected_at_decision": r3.get("test"),
        "group_calibration": [{k: g[k] for k in ("group", "mean_pd", "observed", "observed_ci", "outside_ci")}
                              for g in r10.get("groups_before", [])],
        "stress": [{"scenario": s_["scenario"], "mean_pd": s_["mean_pd"],
                    **{n: {"capital": v["book"]["capital"], "approval_rate": v["lending"]["approval_rate"],
                           "gap": max(v["lending"]["eo_gap_gender"], v["lending"]["eo_gap_age"])}
                       for n, v in s_["policies"].items()}} for s_ in r11.get("scenarios", [])],
        "reject_inference": {"accept_vs_reject_auc": r12.get("accept_vs_reject_auc"),
                             "models": r12.get("models")},
        "drivers": drivers, "latency": {k: lat.get(k) for k in ("p50_ms", "p95_ms", "p99_ms", "n")},
        "data": {"applicants": eda.get("n_applicants"), "defaults": eda.get("n_defaults"),
                 "base_rate": eda.get("base_rate"), "by_gender": eda.get("default_rate_by_group", {}).get("gender"),
                 "by_age": eda.get("default_rate_by_group", {}).get("age_band"),
                 "has_bureau_history": eda.get("has_bureau_history")},
        "limitations": [
            "Trained on one consumer lender's data with no disclosed time frame; the method transfers, the numbers do not.",
            "Cut-offs come from each loan's expected profit, which rests on a 45% loss given default (Basel supervisory "
            "value), a 25% share of interest kept and a fixed cost per loan; the report shows 25-75% and 15-40% ranges.",
            "Only gender and two age groups can be audited; caste, religion, region and disability are not in the data.",
            "The deployed policy uses gender and age to pick a group offset, which some jurisdictions restrict as disparate "
            "treatment; an alternative that needs neither at decision time is on the frontier.",
            "The external scores (EXT_SOURCE) are opaque third-party inputs and dominate the model.",
            "Outcomes exist only for granted loans; reject inference on 64,075 refused applications gave no reliable gain, "
            "so risk estimates for applicants unlike past borrowers remain extrapolations.",
            "The fairness offsets are fixed in currency; in a moderate recession the age gap would exceed 5 points "
            "until the policy is refitted.",
        ],
    })


_GATE_CACHE: dict = {}


@router.get("/gate")
def gate_status():
    """The gate takes a few seconds (1,000 bootstrap resamples, booster
    re-score), so the result is cached until a gated file changes."""
    from gate import collect
    from manifest import BOUND
    files = BOUND + ["models/fair_policy.json", "data/processed/gate_sample.parquet"]
    stamp = tuple((ROOT / f).stat().st_mtime if (ROOT / f).exists() else None for f in files)
    if _GATE_CACHE.get("stamp") == stamp:
        return _GATE_CACHE["result"]
    policy, th, checks = collect(ROOT)
    rows = [{"id": n.split()[0], "name": " ".join(n.split()[1:]),
             "value": v["point"] if isinstance(v, dict) else v if not isinstance(v, float) else round(v, 4),
             "interval": [v["low"], v["high"]] if isinstance(v, dict) else None,
             "status": ("info" if n.startswith("INFO") else "skip") if ok is None else ("pass" if ok else "fail")}
            for n, v, ok in checks]
    result = _clean({"passed": all(r["status"] != "fail" for r in rows), "checks": rows})
    _GATE_CACHE.update(stamp=stamp, result=result)
    return result


@router.get("/decisions")
def decisions(limit: int = Query(25, ge=1, le=200)):
    p = core.log_path()
    if not p.exists():
        return {"decisions": []}
    lines = p.read_text().strip().splitlines()[-limit:]
    return {"decisions": [json.loads(x) for x in reversed(lines) if x.strip()]}
