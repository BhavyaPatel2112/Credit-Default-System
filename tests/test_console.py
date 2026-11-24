"""The console endpoints: they must agree with the committed results and
with the scoring service itself."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from test_api import RISKY, VALID  # noqa: E402

DEMO = ROOT / "data" / "processed" / "demo_applicants.parquet"


@pytest.fixture(scope="module")
def c(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("CAP501_DECISION_LOG", str(tmp_path_factory.mktemp("log") / "d.jsonl"))
    yield TestClient(app)
    mp.undo()


def body(application, gender="Female", age=50):
    return {"application": application, "history": {}, "gender": gender, "age_years": age}


def test_fairness_matches_the_committed_results(c):
    """The console recomputes from the saved predictions; it must agree with
    what the pipeline wrote: r2b (single cut-off at 1/6) and r8 (deployed)."""
    import json
    r2b = json.loads((ROOT / "runs" / "r2b_calibrated.json").read_text().replace("NaN", "null"))["test"]
    r8 = json.loads((ROOT / "runs" / "r8_expected_profit.json").read_text())["test"]
    f = c.get("/api/fairness", params={"cost_ratio": 5}).json()
    assert f["cutoff"] == pytest.approx(1 / 6)
    assert f["single"]["approval_rate"] == pytest.approx(r2b["approval_rate"], abs=1e-9)
    assert f["single"]["gap_age"] == pytest.approx(r2b["eo_gap_age"], abs=1e-9)
    assert f["single"]["gap_gender"] == pytest.approx(r2b["eo_gap_gender"], abs=1e-9)
    assert f["policy"]["approval_rate"] == pytest.approx(r8["approval_rate"], abs=1e-9)
    assert f["policy"]["gap_age"] == pytest.approx(r8["eo_gap_age"], abs=1e-9)
    assert f["policy"]["gap_gender"] == pytest.approx(r8["eo_gap_gender"], abs=1e-9)
    assert f["policy"]["gap_age"] <= f["limit"] and f["policy"]["gap_gender"] <= f["limit"]
    assert len(f["policy"]["groups"]) == 4


def test_cost_ratio_moves_the_cutoff_and_approvals_monotonically(c):
    pts = c.get("/api/fairness/curve").json()["points"]
    rates = [p["approval_rate"] for p in pts]
    assert rates == sorted(rates, reverse=True)  # dearer defaults -> stricter cut-off -> fewer approvals
    assert c.get("/api/fairness", params={"cost_ratio": 0.5}).status_code == 422


def test_gate_endpoint_passes(c):
    g = c.get("/api/gate").json()
    assert g["passed"] and {x["id"] for x in g["checks"]} >= {"G1", "G2", "NFR1", "NFR2", "NFR3", "R-4"}
    nfr3 = [x for x in g["checks"] if x["id"] == "NFR3"]
    assert all(x["interval"][1] <= 0.05 for x in nfr3)  # gated on the upper end


def test_recourse_not_needed_for_an_approval(c):
    r = c.post("/api/recourse", json=body(VALID)).json()
    assert r["needed"] is False and r["options"] == []


def test_recourse_options_really_get_approved(c):
    r = c.post("/api/recourse", json=body(RISKY)).json()
    assert r["needed"] is True
    for o in r["options"]:
        assert o["pd"] < o["cutoff"]
        s = c.post("/score", json=body({**RISKY, "AMT_CREDIT": o["loan"], "AMT_ANNUITY": o["annuity"]})).json()
        assert s["decision"] == "approve"


def test_recourse_validates_like_score(c):
    assert c.post("/api/recourse", json=body({})).status_code == 422


def test_frontier_contains_the_chosen_point(c):
    f = c.get("/api/frontier").json()
    ch = f["chosen"]
    assert any(p["method"] == ch["method"] and p["eps"] == ch["eps"] for p in f["points"])


def test_model_card_and_decision_log(c):
    m = c.get("/api/model-card").json()
    assert m["versions"]["model_sha256"] and m["limitations"]
    c.post("/score", json=body(VALID))
    d = c.get("/api/decisions", params={"limit": 5}).json()["decisions"]
    assert d and d[0]["decision"] == "approve"


@pytest.mark.skipif(not DEMO.exists(), reason="demo applicants not built (needs raw data)")
def test_demo_applicants_score_exactly_as_listed(c):
    lst = c.get("/api/applicants").json()
    assert lst["available"] and len(lst["applicants"]) == 400
    for a in lst["applicants"][:25]:
        req = c.get(f"/api/applicants/{a['id']}").json()["request"]
        s = c.post("/score", json=req).json()
        assert s["probability_of_default"] == pytest.approx(a["pd"], abs=1e-9)
        assert s["decision"] == a["decision"]


def test_unknown_applicant_is_404(c):
    assert c.get("/api/applicants/1").status_code == 404


def test_model_card_carries_model_metrics_for_the_deployed_policy(c):
    m = c.get("/api/model-card").json()
    for k in ("roc_auc", "ece", "eo_gap_gender", "eo_gap_age"):
        assert isinstance(m["test"][k], float), k
    assert all(r["auc"] is not None for r in m["ladder"])
