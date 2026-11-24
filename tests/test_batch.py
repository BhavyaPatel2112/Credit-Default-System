"""Improvement 8: batch scoring validates every row like POST /score, scores
the valid ones with the same service, explains each problem in plain words,
and logs every scored row like a single decision."""
import csv
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from test_api import RISKY, VALID  # noqa: E402


@pytest.fixture(scope="module")
def c(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    log = tmp_path_factory.mktemp("log") / "d.jsonl"
    mp.setenv("CAP501_DECISION_LOG", str(log))
    yield TestClient(app), log
    mp.undo()


def to_csv(rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: "" if v is None else v for k, v in r.items()})
    return out.getvalue()


def batch(c, text):
    return c[0].post("/api/batch", json={"csv": text, "filename": "t.csv"})


def test_valid_rows_score_exactly_like_single_requests(c):
    rows = [{"applicant_id": "a", "gender": "Female", "age_years": 40, **VALID},
            {"applicant_id": "b", "gender": "Male", "age_years": 28, **RISKY}]
    j = batch(c, to_csv(rows)).json()
    assert j["scored"] == 2 and j["rejected"] == 0 and j["approved"] + j["referred"] == 2
    for r, req in zip(j["results"], rows):
        single = c[0].post("/score", json={"application": {k: v for k, v in req.items()
                                                           if k not in ("applicant_id", "gender", "age_years")},
                                           "history": {}, "gender": req["gender"], "age_years": req["age_years"]}).json()
        assert r["probability_of_default"] == pytest.approx(single["probability_of_default"], abs=1e-6)
        assert r["decision"] == single["decision"]


def test_every_scored_row_is_logged_as_a_batch_decision(c):
    before = len(c[1].read_text().splitlines()) if c[1].exists() else 0
    j = batch(c, to_csv([{"gender": "Female", "age_years": 50, **VALID}] * 3)).json()
    lines = c[1].read_text().splitlines()[before:]
    recs = [json.loads(x) for x in lines]
    assert len(recs) == 3 and all(r["source"] == "batch" for r in recs)
    assert {r["decision_id"] for r in recs} == {r["decision_id"] for r in j["results"]}


def test_malformed_rows_get_a_precise_problem_each(c):
    no_income = {k: v for k, v in VALID.items() if k != "AMT_INCOME_TOTAL"}
    rows = [{"applicant_id": "ok", "gender": "Female", "age_years": 40, **VALID},
            {"applicant_id": "no-income", "gender": "Female", "age_years": 40, **no_income},
            {"applicant_id": "text", "gender": "Male", "age_years": 40, **VALID, "AMT_CREDIT": "lots"},
            {"applicant_id": "type", "gender": "Male", "age_years": 40, **VALID, "NAME_CONTRACT_TYPE": "Mortgage"},
            {"applicant_id": "age", "gender": "female", "age_years": 12, **VALID}]
    j = batch(c, to_csv(rows)).json()
    assert j["scored"] == 1 and j["rejected"] == 4
    by = {r["applicant_id"]: r for r in j["rejected_rows"]}
    assert any("Annual income (AMT_INCOME_TOTAL) is missing" in p for p in by["no-income"]["problems"])
    assert any("'lots' is not a number" in p for p in by["text"]["problems"])
    assert any("Mortgage" in p and "allowed: Cash loans, Revolving loans" in p for p in by["type"]["problems"])
    assert any("Gender" in p for p in by["age"]["problems"]) and any("Age" in p for p in by["age"]["problems"])
    assert by["no-income"]["line"] == 3  # header is line 1


def test_results_csv_and_group_rates(c):
    rows = [{"gender": g, "age_years": a, **VALID} for g in ("Female", "Male") for a in (25, 50)]
    j = batch(c, to_csv(rows)).json()
    out = list(csv.DictReader(io.StringIO(j["results_csv"])))
    assert len(out) == 4 and {"probability_of_default", "decision", "cutoff", "reasons"} <= set(out[0])
    assert len(j["groups"]) == 4 and all(g["scored"] == 1 for g in j["groups"])


def test_file_level_problems(c):
    assert batch(c, "").status_code == 422
    assert batch(c, "gender,age_years\n").status_code == 422
    assert batch(c, "gender,gender\nFemale,Female\n").status_code == 422
    j = batch(c, to_csv([{"gender": "Female", "age_years": 40, **VALID, "AMT_INCOM_TOTAL": 5}])).json()
    assert j["file_notes"]["unknown_columns"] == ["AMT_INCOM_TOTAL"] and j["rejected"] == 1


def test_sample_file_round_trips(c):
    text = c[0].get("/api/batch/sample").text
    j = batch(c, text).json()
    assert j["rejected"] == 4 and j["scored"] == j["rows"] - 4
