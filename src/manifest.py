"""Binds the saved predictions to the exact model files that produced them.

The CI gate evaluates runs/pred_test.parquet. Without this binding, a commit
could replace models/xgb.json and the gate would still pass, because it
would be grading the OLD model's predictions. train.py writes this manifest
whenever it writes the model and the predictions together; gate.py refuses
to run if any hash no longer matches.

Usage: python src/manifest.py            (re-record the current files)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOUND = ["models/xgb.json", "models/calibrator.pkl", "models/features.json",
         "runs/pred_valid.parquet", "runs/pred_test.parquet"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root: Path = ROOT) -> dict:
    m = {"bound_files": {f: sha256(root / f) for f in BOUND},
         "note": "Written by train.py together with the model and predictions. Do not edit by hand."}
    (root / "models" / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def verify_manifest(root: Path = ROOT) -> list[str]:
    """Names of bound files whose hash differs from the manifest (empty = all match)."""
    path = root / "models" / "manifest.json"
    if not path.exists():
        return ["models/manifest.json is missing"]
    m = json.loads(path.read_text())["bound_files"]
    bad = []
    for f in BOUND:
        p = root / f
        if not p.exists():
            bad.append(f"{f} is missing")
        elif m.get(f) != sha256(p):
            bad.append(f"{f} changed since the predictions were made")
    return bad


if __name__ == "__main__":
    print(json.dumps(write_manifest(), indent=2))
    sys.exit(0)
