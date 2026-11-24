"""M8: drift and fairness monitoring. Compares a scoring batch with the
training snapshot on four signals and raises alerts:
  input drift       PSI per feature (alert > 0.25, watch > 0.10)
  prediction drift  PSI of the PD distribution + approval-rate change
  calibration decay realised default rate vs mean PD per band
  group fairness    equal-opportunity gap recomputed on the batch
A simulated drifted batch shows each alarm firing.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explain import Scorer
from metrics import NFR3_GAP, ece, eo_gap
from train import RUNS, SEED, load

FIG = RUNS.parent / "reports" / "figures"
WATCH = ["EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
         "DAYS_EMPLOYED", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "INS_DPD_MEAN"]


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


def report(sc: Scorer, ref: pd.DataFrame, batch: pd.DataFrame, name: str) -> dict:
    feats = [f for f in WATCH if f in batch]
    drift = {f: psi(ref[f].values.astype(float), batch[f].values.astype(float)) for f in feats}
    p_ref, p_cur = sc.pd_(ref), sc.pd_(batch)
    a_ref = sc.approve(ref, Scorer.key(ref)); a_cur = sc.approve(batch, Scorer.key(batch))
    y = batch.TARGET.values
    bands = pd.cut(p_cur, [0, .05, .1, .2, .35, 1.01])
    cal = pd.DataFrame({"band": bands.astype(str), "pd": p_cur, "y": y}).groupby("band").agg(
        n=("y", "size"), mean_pd=("pd", "mean"), realised=("y", "mean")).reset_index()
    gap = max(eo_gap(y, a_cur, batch.gender.values), eo_gap(y, a_cur, batch.age_under_35.values))
    out = {
        "batch": name, "n": int(len(batch)),
        "input_psi": drift,
        "input_alerts": [f for f, v in drift.items() if v > 0.25],
        "prediction_psi": psi(p_ref, p_cur),
        "approval_rate_ref": float(a_ref.mean()), "approval_rate_batch": float(a_cur.mean()),
        "ece_batch": ece(y, p_cur),
        "calibration_by_band": cal.to_dict("records"),
        "eo_gap_gender_batch": gap,
    }
    out["alerts"] = {
        "input_drift": bool(out["input_alerts"]),
        "prediction_drift": out["prediction_psi"] > 0.25 or abs(out["approval_rate_batch"] - out["approval_rate_ref"]) > 0.05,
        "calibration_decay": out["ece_batch"] > 0.05,
        "fairness_gap": gap > NFR3_GAP,
    }
    out["action"] = ("retrain (7 -> 3)" if out["alerts"]["input_drift"] or out["alerts"]["calibration_decay"] else "none") + \
                    ("; fairness re-audit (5 -> 3)" if out["alerts"]["fairness_gap"] else "")
    return out


def simulate_drift(df: pd.DataFrame, rng) -> pd.DataFrame:
    """A plausible shock: a bureau-score vendor change compresses EXT_SOURCE_3,
    income inflation lifts incomes 25%, and repayment stress raises defaults
    among thin-file applicants."""
    d = df.copy()
    d["EXT_SOURCE_3"] = (d["EXT_SOURCE_3"] * 0.7 + 0.25).clip(0, 1)
    d["AMT_INCOME_TOTAL"] = d["AMT_INCOME_TOTAL"] * 1.25
    d = Scorer.recompute(d)
    thin = d["BUR_COUNT"].isna().values if "BUR_COUNT" in d else np.zeros(len(d), bool)
    flip = (d.TARGET.values == 0) & thin & (rng.random(len(d)) < 0.06)
    d.loc[flip, "TARGET"] = 1
    return d


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    parts, _ = load()
    sc = Scorer()
    rng = np.random.default_rng(SEED)
    ref = parts["train"].sample(min(30000, len(parts["train"])), random_state=SEED)
    stable = report(sc, ref, parts["test"], "test split (no drift)")
    drifted = report(sc, ref, simulate_drift(parts["test"], rng), "simulated drift")
    res = {"stable": stable, "drifted": drifted}
    (RUNS / "r6_monitoring.json").write_text(json.dumps(res, indent=2, default=float))
    for r in (stable, drifted):
        print(r["batch"], r["alerts"], r["action"], round(r["prediction_psi"], 3), r["input_alerts"])

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=160)
    f = list(stable["input_psi"])
    x = np.arange(len(f))
    ax.bar(x - 0.2, [stable["input_psi"][k] for k in f], 0.4, label="Test split", color="#8aa9d6")
    ax.bar(x + 0.2, [drifted["input_psi"][k] for k in f], 0.4, label="Simulated drift", color="#c0392b")
    ax.axhline(0.25, ls="--", color="#c0392b", lw=1); ax.axhline(0.10, ls=":", color="#999", lw=1)
    ax.set_xticks(x, f, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("PSI vs training snapshot"); ax.set_title("Input drift monitor (alert at PSI 0.25)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "drift_psi.png")


if __name__ == "__main__":
    main()
