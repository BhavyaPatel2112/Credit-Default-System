"""The scoring service: it must refuse what it cannot score honestly, build
the model row exactly as training did, and explain referrals correctly."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api.app import CONTRACT, FEATURES, app  # noqa: E402
from api.features import derive, request_from_encoded  # noqa: E402
from data import SENTINEL, clean_application  # noqa: E402

VALID = {
    "AMT_INCOME_TOTAL": 202500, "AMT_CREDIT": 406597.5, "AMT_ANNUITY": 24700.5, "AMT_GOODS_PRICE": 351000,
    "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 1, "REGION_POPULATION_RELATIVE": 0.0188,
    "REGION_RATING_CLIENT": 2, "REGION_RATING_CLIENT_W_CITY": 2,
    "DAYS_REGISTRATION": -3648, "DAYS_EMPLOYED": -637, "FLAG_EMP_PHONE": 1,
    "REG_REGION_NOT_LIVE_REGION": 0, "REG_REGION_NOT_WORK_REGION": 0,
    "LIVE_REGION_NOT_WORK_REGION": 0, "REG_CITY_NOT_LIVE_CITY": 0, "REG_CITY_NOT_WORK_CITY": 0,
    "LIVE_CITY_NOT_WORK_CITY": 0,
    "NAME_CONTRACT_TYPE": "Cash loans", "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y",
    "NAME_INCOME_TYPE": "Working", "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_HOUSING_TYPE": "House / apartment", "ORGANIZATION_TYPE": "Business Entity Type 3",
    "EXT_SOURCE_1": 0.7, "EXT_SOURCE_2": 0.7, "EXT_SOURCE_3": 0.7,
}
RISKY = {**VALID, "EXT_SOURCE_1": 0.08, "EXT_SOURCE_2": 0.08, "EXT_SOURCE_3": 0.08, "DAYS_EMPLOYED": -200,
         "NAME_EDUCATION_TYPE": "Secondary / secondary special"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    log = tmp_path_factory.mktemp("log") / "decisions.jsonl"
    mp = pytest.MonkeyPatch()
    mp.setenv("CAP501_DECISION_LOG", str(log))
    yield TestClient(app), log
    mp.undo()


def post(client, application, gender="Female", age=40, history=None):
    return client[0].post("/score", json={"application": application, "history": history or {},
                                          "gender": gender, "age_years": age})


# ---------------------------------------------------------------- rejects
def test_empty_request_is_rejected_not_approved(client):
    r = post(client, {})
    assert r.status_code == 422
    errs = r.json()["detail"]
    assert "AMT_INCOME_TOTAL: required" in errs and any(e.startswith("NAME_EDUCATION_TYPE: required") for e in errs)


@pytest.mark.parametrize("field,value", [
    ("AMT_INCOME_TOTAL", -5e6), ("EXT_SOURCE_2", 50), ("DAYS_EMPLOYED", 400), ("FLAG_EMP_PHONE", 2),
    ("REGION_RATING_CLIENT", 7), ("AMT_ANNUITY", 9e9), ("CNT_FAM_MEMBERS", 0), ("AMT_CREDIT", "a lot"),
])
def test_out_of_range_values_are_rejected(client, field, value):
    assert post(client, {**VALID, field: value}).status_code == 422


@pytest.mark.parametrize("bad", [{"EXT_SOURCE2": 0.5}, {"CODE_GENDER": "F"}, {"DAYS_BIRTH": -12000},
                                 {"CREDIT_INCOME_RATIO": 2.0}, {"NAME_EDUCATION_TYPE": "PhD"}])
def test_unknown_protected_derived_or_invalid_fields_are_rejected(client, bad):
    r = post(client, {**VALID, **bad})
    assert r.status_code == 422
    assert any(k in " ".join(r.json()["detail"]) for k in bad)


def test_unknown_history_feature_and_unknown_group_rejected(client):
    assert post(client, VALID, history={"BUR_MADE_UP": 1}).status_code == 422
    assert post(client, VALID, gender="female").status_code == 422


def test_employment_may_be_null_but_not_absent(client):
    assert post(client, {**VALID, "DAYS_EMPLOYED": None, "NAME_INCOME_TYPE": "Pensioner",
                         "ORGANIZATION_TYPE": "XNA"}).status_code == 200
    assert post(client, {k: v for k, v in VALID.items() if k != "DAYS_EMPLOYED"}).status_code == 422


# ---------------------------------------------------------------- scoring
def test_valid_request_scores_and_logs(client):
    r = post(client, VALID)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["probability_of_default"] <= 1 and j["decision"] in {"approve", "refer"}
    assert j["versions"]["model_sha256"]
    rec = json.loads(client[1].read_text().strip().splitlines()[-1])
    assert rec["decision_id"] == j["decision_id"] and rec["model_sha256"] == j["versions"]["model_sha256"]
    assert "application" not in rec and "AMT_INCOME_TOTAL" not in json.dumps(rec)  # no raw personal data


def test_worse_credit_scores_mean_higher_risk(client):
    good = post(client, VALID).json()["probability_of_default"]
    bad = post(client, RISKY).json()["probability_of_default"]
    assert bad > good + 0.2


def test_protected_attributes_change_only_the_cutoff(client):
    out = [post(client, RISKY, g, a).json() for g in ("Female", "Male") for a in (28, 50)]
    assert len({round(o["probability_of_default"], 12) for o in out}) == 1
    assert len({o["cutoff_applied"] for o in out}) == 4


def test_referral_reasons_only_count_against_and_do_not_repeat(client):
    j = post(client, RISKY, "Female", 50).json()
    assert j["decision"] == "refer" and j["needs_human_review"]
    reasons = j["adverse_action_reasons"]
    assert 1 <= len(reasons) <= 4
    assert all(g["effect"] == "raises risk" and g["shap_logodds"] > 0 for g in reasons)
    assert len({g["reason"] for g in reasons}) == len(reasons)
    assert sum(g["reason"].startswith("External credit scores") for g in j["top_factors"]) == 1


def test_approvals_carry_no_adverse_reasons(client):
    j = post(client, VALID).json()
    assert j["decision"] == "approve" and j["adverse_action_reasons"] == []


# ---------------------------------------------------- training parity
def test_derived_features_match_training_code():
    raw = pd.DataFrame({
        "CODE_GENDER": ["F", "M", "F"], "DAYS_EMPLOYED": [-500, SENTINEL, -10],
        "AMT_INCOME_TOTAL": [1e5, 2e5, 5e4], "AMT_CREDIT": [3e5, 1e5, 2e5], "AMT_ANNUITY": [1.5e4, 9e3, 1e4],
        "AMT_GOODS_PRICE": [2.7e5, np.nan, 2e5], "DAYS_BIRTH": [-12000, -20000, -9000],
        "EXT_SOURCE_1": [0.5, np.nan, np.nan], "EXT_SOURCE_2": [0.6, 0.2, np.nan], "EXT_SOURCE_3": [0.7, 0.4, np.nan],
    })
    trained = clean_application(raw)
    for i in range(len(raw)):
        num = raw.iloc[i].to_dict()
        num["DAYS_EMPLOYED"] = None if num["DAYS_EMPLOYED"] == SENTINEL else num["DAYS_EMPLOYED"]
        num = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in num.items()}
        served = derive(num)
        for k, v in served.items():
            t = trained.iloc[i][k]
            assert (v is None and pd.isna(t)) or np.isclose(v, t), (i, k, v, t)


def test_encode_round_trips_a_training_row():
    """A row of the training matrix -> raw form -> encode must give the same row."""
    rng = np.random.default_rng(0)
    enc = {f: np.nan for f in FEATURES}
    enc.update({k: float(v) for k, v in VALID.items() if not isinstance(v, str)})
    for g, levels in CONTRACT.categories.items():
        pick = rng.choice(levels)
        for lv in levels:
            enc[f"{g}_{lv}"] = float(lv == pick)
    enc["BUR_COUNT"] = 3.0
    enc.update(derive({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in enc.items()}))
    enc = {k: (np.nan if v is None else v) for k, v in enc.items()}
    app_, hist = request_from_encoded(CONTRACT, enc)
    row, _ = CONTRACT.encode(app_, hist)
    want = np.array([enc[f] for f in FEATURES], dtype=float)
    assert np.allclose(row, want, equal_nan=True)


# ---------------------------------------------- improvement 5: dropped fields
def test_not_used_fields_match_training_code():
    import prepare
    from api import features as F
    assert F.NOT_USED == set(prepare.NOT_DEFENSIBLE)
    assert F.NOT_USED_PREFIXES == prepare.NOT_DEFENSIBLE_PREFIXES
    assert not any(F.not_used(f) for f in FEATURES), "the model must not use a dropped feature"


def test_dropped_fields_are_ignored_and_reported(client):
    legacy = {**VALID, "HOUR_APPR_PROCESS_START": 3, "WEEKDAY_APPR_PROCESS_START": "SUNDAY",
              "DAYS_LAST_PHONE_CHANGE": -1, "FLAG_DOCUMENT_3": 1, "OBS_30_CNT_SOCIAL_CIRCLE": 9}
    a, b = post(client, VALID).json(), post(client, legacy).json()
    assert a["probability_of_default"] == b["probability_of_default"]
    assert "HOUR_APPR_PROCESS_START" in b["data_quality"]["ignored_fields"]
    assert all(r["reason"] != "Time of application" for r in b["top_factors"])


def test_cutoff_follows_the_deployed_policy(client):
    sys.path.insert(0, str(ROOT / "src"))
    import policy as P
    from api.app import POLICY
    j = post(client, RISKY, "Male", 30).json()
    want = P.cutoff_one(POLICY, "Male|Under 35", RISKY["AMT_CREDIT"], RISKY["AMT_ANNUITY"])
    assert j["cutoff_applied"] == pytest.approx(want)
    if POLICY.get("type") == "expected_profit":
        # approve exactly when expected profit clears the group's offset
        assert (j["expected_profit"] >= POLICY["offsets"]["Male|Under 35"]) == (j["decision"] == "approve")
