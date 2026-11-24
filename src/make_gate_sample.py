"""Builds data/processed/gate_sample.parquet for gate check G3: 2,000 test
applicants (stratified on default x gender x age) with their 449 features and
the raw score the committed booster gives them. Commit the file; from then on
the gate re-runs the booster on every pull request instead of trusting the
saved predictions.

Needs data/processed/features.parquet (python src/data.py && python src/prepare.py).
Re-run it whenever train.py writes a new model.
"""
import json

import pandas as pd
import xgboost as xgb

from data import PROC, ROOT
from train import SEED, load


def main(n: int = 2000) -> None:
    parts, _ = load()
    te = parts["test"]
    feats = json.load(open(ROOT / "models" / "features.json"))
    strata = te.TARGET.astype(str) + te.gender + te.age_under_35
    s = te.groupby(strata, group_keys=False).apply(
        lambda g: g.sample(n=max(1, round(n * len(g) / len(te))), random_state=SEED))
    b = xgb.Booster(); b.load_model(ROOT / "models" / "xgb.json")
    out = s[["SK_ID_CURR"] + feats].reset_index(drop=True)
    out["p_raw_expected"] = b.predict(xgb.DMatrix(out[feats]))
    out.to_parquet(PROC / "gate_sample.parquet", index=False)
    print(f"gate sample: {len(out):,} applicants -> {PROC / 'gate_sample.parquet'}")


if __name__ == "__main__":
    main()
