"""ETH4 ruling evidence: refit the chosen configuration without the features
the proxy audit flagged (|r| > 0.4 with a protected attribute) and compare
AUC and the age gap. The ruling in the report rests on this comparison."""
import json

import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from data import ROOT
from metrics import evaluate
from train import RUNS, THRESHOLD, groups_of, load, log_run


def main():
    parts, feats = load()
    tr, va, te = parts["train"], parts["valid"], parts["test"]
    flagged = pd.read_csv(ROOT / "reports" / "proxy_audit.csv")
    flagged = sorted(set(flagged[flagged.flagged].feature))
    best = json.load(open(RUNS / "r2_weighted_tuned_uncalibrated.json"))
    params, rounds = best["params"], best["rounds"]
    kept = [f for f in feats if f not in flagged]
    b = xgb.train(params, xgb.DMatrix(tr[kept], label=tr.TARGET), num_boost_round=rounds)
    iso = IsotonicRegression(out_of_bounds="clip").fit(b.predict(xgb.DMatrix(va[kept])), va.TARGET)
    p = iso.predict(b.predict(xgb.DMatrix(te[kept])))
    log_run("r2c_without_flagged_proxies", {"removed": flagged, "features": len(kept),
                                            "test": evaluate(te.TARGET.values, p, groups_of(te), THRESHOLD)})


if __name__ == "__main__":
    main()
