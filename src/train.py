"""M5: run ladder rungs 1-2 — baseline XGBoost, class weighting, tuning,
isotonic calibration. Every run is written to runs/<name>.json.

Split roles
  train : fit the booster (number of trees chosen by early stopping on a 15% inner hold-out of train)
  valid : fit the isotonic calibrator, the fairness post-processor and the
          decision threshold — never the booster
  test  : touched once per rung, for reporting
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from data import PROC, ROOT
from manifest import write_manifest
from metrics import evaluate, gate, reliability_table

RUNS = ROOT / "runs"
MODELS = ROOT / "models"
SEED = 42
# Decision policy (planning assumption, stated in the report): approving a
# defaulter costs 5x the margin lost by declining a repayer, so the
# cost-minimising cut-off on a calibrated PD is 1 / (1 + 5).
C_FN, C_FP = 5.0, 1.0
THRESHOLD = C_FP / (C_FP + C_FN)


def load():
    F = pd.read_parquet(PROC / "features.parquet")
    P = pd.read_parquet(PROC / "protected.parquet")
    F = F.merge(P, on="SK_ID_CURR")
    parts = {}
    for k in ("train", "valid", "test"):
        ids = pd.read_csv(PROC / f"split_{k}.csv")["SK_ID_CURR"]
        parts[k] = F[F.SK_ID_CURR.isin(ids)].reset_index(drop=True)
    prot_cols = list(P.columns.drop("SK_ID_CURR"))
    feats = [c for c in F.columns if c not in ["SK_ID_CURR", "TARGET"] + prot_cols]
    return parts, feats


def groups_of(d: pd.DataFrame) -> dict:
    return {"gender": d["gender"].values, "age": d["age_under_35"].values}


def log_run(name: str, payload: dict) -> None:
    RUNS.mkdir(exist_ok=True)
    payload = {"run": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    (RUNS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"== {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in payload.get("test", {}).items()
                                       if isinstance(v, float)))


def fit_es(params: dict, X: pd.DataFrame, y: np.ndarray, rounds: int = 3000):
    """Early stopping on a stratified 15% inner hold-out of TRAIN (valid and
    test are never touched). Returns the booster trimmed to its best
    iteration, the number of rounds and the inner hold-out AUC."""
    from sklearn.model_selection import train_test_split
    i_fit, i_es = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=SEED)
    dfit = xgb.DMatrix(X.iloc[i_fit], label=y[i_fit])
    des = xgb.DMatrix(X.iloc[i_es], label=y[i_es])
    b = xgb.train(params, dfit, num_boost_round=rounds, evals=[(des, "es")],
                  early_stopping_rounds=60, verbose_eval=False)
    best = b.best_iteration + 1
    auc = float(b.best_score)
    del dfit, des
    return b[:best], best, auc


BASE = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
        "nthread": 8, "seed": SEED, "eta": 0.08, "max_depth": 6, "min_child_weight": 1,
        "subsample": 0.8, "colsample_bytree": 0.5}


def fit(params, X, y, rounds):
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=rounds)


def main(skip_r0: bool = False) -> None:
    MODELS.mkdir(exist_ok=True)
    parts, feats = load()
    tr, va, te = parts["train"], parts["valid"], parts["test"]
    ytr, yva, yte = tr.TARGET.values, va.TARGET.values, te.TARGET.values
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    print(f"{len(feats)} features; scale_pos_weight={spw:.2f}")

    # ---- Rung 0: application table only (leakage reference point) --------
    # (already logged by the first run; set skip_r0=False to recompute)
    app_feats = [c for c in feats if not c.startswith(("BUR_", "PREV_", "POS_", "INS_", "CC_"))]
    if not skip_r0:
      b, n, es_auc = fit_es(BASE, tr[app_feats], ytr)
      p = b.predict(xgb.DMatrix(te[app_feats]))
      log_run("r0_application_only", {"features": len(app_feats), "rounds": n, "inner_holdout_auc": es_auc,
                                    "test": evaluate(yte, p, groups_of(te), THRESHOLD)})

    # ---- Rung 1: all seven tables, unweighted ----------------------------
    b1, n, es_auc = fit_es(BASE, tr[feats], ytr)
    p1 = b1.predict(xgb.DMatrix(te[feats]))
    del b1
    log_run("r1_baseline_unweighted", {"features": len(feats), "rounds": n, "inner_holdout_auc": es_auc,
                                       "test": evaluate(yte, p1, groups_of(te), THRESHOLD)})

    # ---- Rung 2: class weighting + small tuning search -------------------
    grid = [
        {"max_depth": 4, "min_child_weight": 5, "colsample_bytree": 0.3},
        {"max_depth": 5, "min_child_weight": 10, "colsample_bytree": 0.4},
        {"max_depth": 6, "min_child_weight": 20, "colsample_bytree": 0.3},
    ]
    best = None
    search = []
    for g in grid:
        params = {**BASE, **g, "scale_pos_weight": spw}
        bb, n, a = fit_es(params, tr[feats], ytr)
        search.append({**g, "rounds": n, "inner_holdout_auc": a})
        print("  search", g, n, round(a, 4), flush=True)
        if best is None or a > best[2]:
            best = (params, n, a, bb)
        else:
            del bb
    params, n, es_auc, b2 = best
    b2.save_model(MODELS / "xgb.json")
    p2_va = b2.predict(xgb.DMatrix(va[feats]))
    p2_te = b2.predict(xgb.DMatrix(te[feats]))
    log_run("r2_weighted_tuned_uncalibrated", {"params": params, "rounds": n, "inner_holdout_auc": es_auc,
                                               "search": search,
                                               "test": evaluate(yte, p2_te, groups_of(te), THRESHOLD)})

    # ---- Rung 2b: isotonic calibration on valid --------------------------
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p2_va, yva)
    pickle.dump(iso, open(MODELS / "calibrator.pkl", "wb"))
    pc_va, pc_te = iso.predict(p2_va), iso.predict(p2_te)
    m = evaluate(yte, pc_te, groups_of(te), THRESHOLD)
    log_run("r2b_calibrated", {"calibration": "isotonic, fitted on valid", "test": m,
                               "gate": gate(m), "reliability_before": reliability_table(yte, p2_te).to_dict("records"),
                               "reliability_after": reliability_table(yte, pc_te).to_dict("records")})

    # Leakage review artefacts
    imp = pd.Series(b2.get_score(importance_type="gain")).sort_values(ascending=False)
    imp.head(40).to_csv(RUNS / "feature_importance_gain.csv", header=["gain"])

    # AMT_CREDIT / AMT_ANNUITY travel with the predictions: the expected-profit
    # policy, the stress test and the capital calculation need the exposure.
    keep = ["SK_ID_CURR", "TARGET", "gender", "age_under_35", "age_band", "family_status",
            "AMT_CREDIT", "AMT_ANNUITY"]
    va[keep].assign(p_raw=p2_va, p_cal=pc_va).to_parquet(RUNS / "pred_valid.parquet", index=False)
    te[keep].assign(p_raw=p2_te, p_cal=pc_te).to_parquet(RUNS / "pred_test.parquet", index=False)
    json.dump(feats, open(MODELS / "features.json", "w"))
    json.dump({"threshold": THRESHOLD, "c_fn": C_FN, "c_fp": C_FP}, open(MODELS / "policy.json", "w"))
    # Bind the predictions to the model that made them (gate check G1).
    write_manifest(ROOT)


if __name__ == "__main__":
    main()
