"""M7 / NFR4: p95 latency of the scoring service, explanation included,
measured in-process over 1,000 real test-split applicants. Each applicant's
encoded row is turned back into the raw form the service now accepts, so the
measurement includes validation and encoding."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402

from api.app import CONTRACT, FEATURES, app  # noqa: E402
from api.features import request_from_encoded  # noqa: E402
from train import load, RUNS  # noqa: E402


def main(n=1000):
    parts, _ = load()
    te = parts["test"].sample(n=min(n, len(parts["test"])), random_state=0)
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    server_ms, wall_ms, decisions, sample, rejected = [], [], [], None, []
    for _, r in te.iterrows():
        application, history = request_from_encoded(CONTRACT, {f: r[f] for f in FEATURES})
        body = {"application": application, "history": history, "gender": r.gender, "age_years": float(r.age_years)}
        t0 = time.perf_counter()
        resp = c.post("/score", json=body)
        wall_ms.append((time.perf_counter() - t0) * 1000)
        j = resp.json()
        if resp.status_code != 200:  # e.g. the 12 training rows with no instalment amount
            rejected.append({"SK_ID_CURR": int(r.SK_ID_CURR), "errors": j["detail"]})
            continue
        server_ms.append(j["latency_ms"]); decisions.append(j["decision"])
        if sample is None and j["decision"] == "refer":
            sample = j
    out = {"n": len(te), "p50_ms": float(np.percentile(server_ms, 50)), "p95_ms": float(np.percentile(server_ms, 95)),
           "p99_ms": float(np.percentile(server_ms, 99)), "p95_ms_incl_http": float(np.percentile(wall_ms, 95)),
           "NFR4_pass": float(np.percentile(server_ms, 95)) <= 50, "refer_rate": decisions.count("refer") / len(decisions),
           "rejected_by_validation": len(rejected), "rejections": rejected[:10],
           "example_refer_response": sample}
    (RUNS / "r7_latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "example_refer_response"}, indent=1))


if __name__ == "__main__":
    main()
