"""M8: the CI gate. Exits non-zero if the committed model + policy fail any
check, so .github/workflows/gate.yml blocks the merge.

Integrity checks (is the gate grading the model that is actually committed?)
  G1  models/manifest.json: the model, calibrator, feature list and saved
      predictions are the exact files that were produced together.
  G2  the committed calibrator, applied to the saved raw scores, reproduces
      the saved calibrated PDs.
  G3  the committed booster, re-run on a frozen sample of test applicants
      (data/processed/gate_sample.parquet), reproduces the saved raw scores.
      Needs the sample, which src/make_gate_sample.py builds from the raw
      data; until it is committed this check is reported as NOT RUN.
Requirement checks (on the frozen test split)
  NFR1 AUC, NFR2 ECE, NFR3 gender and age gaps, R-4 cell sizes, and every
  policy group must have a cut-off (no silent fallback).
  Since 1.3 each requirement is checked on the unfavourable end of its
  bootstrap 95% interval (1,000 resamples of the test split, fixed seed):
  the lower end for AUC, the upper end for ECE and the gaps. A point value
  that passes only by luck of the sample now fails. The point value is
  still shown.

Usage: python src/gate.py [--policy models/fair_policy.json] [--root .]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import policy as P  # noqa: E402
from manifest import verify_manifest  # noqa: E402
from metrics import MIN_CELL, NFR1_AUC, NFR2_ECE, NFR3_GAP, bootstrap, eo_gap, group_rates  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-6


def integrity(root: Path, d: pd.DataFrame) -> list[tuple[str, object, bool | None]]:
    out = []
    bad = verify_manifest(root)
    out.append(("G1    model files match the saved predictions (manifest)",
                "all match" if not bad else "; ".join(bad), not bad))
    iso = pickle.load(open(root / "models" / "calibrator.pkl", "rb"))
    diff = float(np.max(np.abs(iso.predict(d.p_raw.values) - d.p_cal.values)))
    out.append(("G2    calibrator reproduces the saved PDs", f"max difference {diff:.1e}", diff <= TOL))
    sample = root / "data" / "processed" / "gate_sample.parquet"
    if sample.exists():
        import xgboost as xgb
        feats = json.load(open(root / "models" / "features.json"))
        s = pd.read_parquet(sample)
        b = xgb.Booster(); b.load_model(root / "models" / "xgb.json")
        diff = float(np.max(np.abs(b.predict(xgb.DMatrix(s[feats])) - s.p_raw_expected.values)))
        out.append((f"G3    booster re-scores {len(s):,} frozen applicants", f"max difference {diff:.1e}", diff <= 1e-5))
    else:
        out.append(("G3    booster re-scores frozen applicants",
                    "NOT RUN: gate_sample.parquet not committed (run src/make_gate_sample.py)", None))
    return out


def collect(root: Path = ROOT, policy: Path | None = None) -> tuple[Path, dict, list]:
    """Run every check. Returns (policy path, thresholds, [(name, value, ok)]),
    where ok is True, False, or None for SKIP / INFO. Used by main() and by
    the console's /api/gate endpoint."""
    root = Path(root)
    policy = Path(policy) if policy else root / "models" / "fair_policy.json"
    pol = json.load(open(policy))
    d = pd.read_parquet(root / "runs" / "pred_test.parquet")

    checks = integrity(root, d)

    keys = P.group_key(d.gender.values, d.age_under_35.values)
    th = pol.get("thresholds", pol.get("offsets"))
    missing = P.covers(pol, keys)
    checks.append(("POL   every policy group has a cut-off", "ok" if not missing else f"no cut-off for {missing}",
                   not missing))
    if missing:  # a policy that does not cover everyone cannot be evaluated
        return policy, th, checks
    if pol.get("type") == "expected_profit" and "AMT_CREDIT" not in d:
        checks.append(("POL   loan amounts available for the expected-profit policy",
                       "runs/pred_test.parquet has no AMT_CREDIT", False))
        return policy, th, checks
    approve = P.approve(pol, d.p_cal.values, keys, d.get("AMT_CREDIT"), d.get("AMT_ANNUITY"))
    checks += requirement_checks(d.TARGET.values, d.p_cal.values, approve, d.gender.values,
                                 d.age_under_35.values, keys)
    return policy, th, checks


def requirement_checks(y, p, approve, gender, age, keys) -> list[tuple[str, object, bool | None]]:
    """NFR1-NFR3 on the unfavourable end of the bootstrap 95% interval, R-4
    and the four-group gap. Used by collect() and by src/scorecard.py, so a
    challenger model is held to exactly the same gate."""
    b = bootstrap(y, p, approve, {"gender": gender, "age": age})
    cells = group_rates(y, approve, keys)
    fmt = lambda k: {"point": b[k]["point"], "low": b[k]["low"], "high": b[k]["high"]}  # noqa: E731
    return [
        ("NFR1  ROC-AUC >= %.2f, lower 95%% bound" % NFR1_AUC, fmt("auc"), b["auc"]["low"] >= NFR1_AUC),
        ("NFR2  ECE <= %.2f, upper 95%% bound" % NFR2_ECE, fmt("ece"), b["ece"]["high"] <= NFR2_ECE),
        ("NFR3  equal-opportunity gap (gender) <= %.2f, upper 95%% bound" % NFR3_GAP, fmt("eo_gap_gender"),
         b["eo_gap_gender"]["high"] <= NFR3_GAP),
        ("NFR3  equal-opportunity gap (age <35 vs 35+) <= %.2f, upper 95%% bound" % NFR3_GAP, fmt("eo_gap_age"),
         b["eo_gap_age"]["high"] <= NFR3_GAP),
        ("R-4   every gated group has >= %d repayers" % MIN_CELL, int(cells.n_repayers.min()),
         bool((cells.n_repayers >= MIN_CELL).all())),
        ("INFO  gap across the four gender x age groups (reported, not gated)", eo_gap(y, approve, keys), None),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--policy", default=None)
    args = ap.parse_args(argv)
    policy, th, checks = collect(Path(args.root), args.policy)
    return report(policy, th, checks)


def report(policy, th, checks) -> int:
    print(f"Policy: {policy}  {th}")
    for name, val, ok in checks:
        tag = ("INFO" if name.startswith("INFO") else "SKIP") if ok is None else "PASS" if ok else "FAIL"
        v = f"{val:.4f}" if isinstance(val, float) else \
            f"{val['point']:.4f} (95% CI {val['low']:.4f} to {val['high']:.4f})" if isinstance(val, dict) else str(val)
        print(f"  [{tag}] {name}: {v}")
    passed = all(ok is not False for _, _, ok in checks)
    print("GATE PASSED" if passed else "GATE FAILED - this commit cannot be merged")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
