"""The feature contract between training and serving.

Why this file exists
--------------------
The model was trained on a matrix built by src/data.py and src/prepare.py.
In that matrix some columns are NEVER missing: every one-hot column of a
categorical field (0 or 1), every flag, and most application fields. If the
service fills those with NaN, XGBoost sends the row down a branch it never
learned from, and the score is meaningless (an empty request used to come
back as "approve, PD 0.0").

So the service no longer accepts the 449 encoded columns. It accepts the
application form in raw form (numbers, and categories as text), validates
it, and builds the encoded row itself with the same formulas as training.
Credit-history aggregates are optional: a thin-file applicant has none, and
in training such applicants also had NaN there.

tests/test_api.py checks that `derive` here matches
`data.clean_application` exactly, so training and serving cannot drift apart
silently (training-serving skew).
"""
from __future__ import annotations

import numpy as np

HISTORY_PREFIXES = ("BUR_", "PREV_", "POS_", "INS_", "CC_")

# Never sent to the model (ETH1). Rejected if they appear in the application.
PROTECTED = {"CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", "EMPLOYED_BIRTH_RATIO"}

# Computed by the server from the raw fields. Rejected if the client sends
# them, so a client can never feed a ratio that disagrees with its inputs.
DERIVED = {"CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO",
           "EXT_SOURCE_MEAN", "EXT_SOURCE_PROD", "EXT_SOURCE_1_MISSING", "EXT_SOURCE_2_MISSING",
           "EXT_SOURCE_3_MISSING", "DAYS_EMPLOYED_ANOM"}

# Categorical form fields, as they were one-hot encoded in prepare.py.
# REQUIRED: the field was never missing in training (no "_nan" column).
# OPTIONAL: training had a "_nan" column, so "not given" is a known state.
CATEGORICAL_REQUIRED = ["NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
                        "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"]
CATEGORICAL_OPTIONAL = ["NAME_TYPE_SUITE", "OCCUPATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
                        "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]
CATEGORICAL = CATEGORICAL_REQUIRED + CATEGORICAL_OPTIONAL

# Improvement 5: form fields the model no longer uses because a lender could
# not defend them as a reason (time of application, contact details given,
# documents ticked, social-circle defaults). Must equal
# prepare.NOT_DEFENSIBLE / NOT_DEFENSIBLE_PREFIXES (checked in tests). A
# client that still sends them is not rejected: the fields are ignored and
# listed in data_quality.ignored_fields.
NOT_USED = {"HOUR_APPR_PROCESS_START", "WEEKDAY_APPR_PROCESS_START", "DAYS_LAST_PHONE_CHANGE",
            "DAYS_ID_PUBLISH", "FLAG_MOBIL", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", "FLAG_WORK_PHONE"}
NOT_USED_PREFIXES = ("FLAG_DOCUMENT_", "OBS_", "DEF_", "PREV_HOUR_APPR_PROCESS_START",
                     "PREV_NFLAG_LAST_APPL_IN_DAY", "PREV_SELLERPLACE_AREA")


def not_used(key: str) -> bool:
    return key in NOT_USED or key.startswith(NOT_USED_PREFIXES)

# Numeric form fields with no missing values in application_train (except
# AMT_ANNUITY: 12 rows, and CNT_FAM_MEMBERS: 2 rows, both negligible).
NUMERIC_REQUIRED = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "DAYS_REGISTRATION", "FLAG_EMP_PHONE", "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
]
# Must be present as a key, but null is allowed and meaningful:
# DAYS_EMPLOYED null = no current employment record (the 365243 placeholder
# in training, decoded to NaN with DAYS_EMPLOYED_ANOM = 1).
NUMERIC_NULLABLE_REQUIRED = ["DAYS_EMPLOYED"]


def mangle(col: str) -> str:
    """The exact column renaming applied in prepare.feature_matrix."""
    return col.replace(" ", "_").replace(",", "").replace("[", "(").replace("]", ")").replace("<", "lt")


class Contract:
    """Built from models/features.json, so it always matches the model."""

    def __init__(self, features: list[str]):
        self.features = features
        self.index = {f: i for i, f in enumerate(features)}
        self.history = [f for f in features if f.startswith(HISTORY_PREFIXES)]
        self.categories: dict[str, list[str]] = {}
        for g in CATEGORICAL:
            pre = g + "_"
            self.categories[g] = [f[len(pre):] for f in features if f.startswith(pre)]
        dummy_cols = {f"{g}_{v}" for g, vs in self.categories.items() for v in vs}
        known = set(NUMERIC_REQUIRED) | set(NUMERIC_NULLABLE_REQUIRED)
        # Everything else at application level is an optional number
        # (housing statistics, car age, social circle, bureau enquiries ...).
        self.numeric_optional = [f for f in features if f not in known and f not in dummy_cols
                                 and f not in DERIVED and f not in self.history]
        self.numeric_all = set(known) | set(self.numeric_optional)
        self.application_keys = self.numeric_all | set(CATEGORICAL)

    # ------------------------------------------------------------------
    def validate(self, app: dict, hist: dict) -> list[str]:
        """Every problem with the request, so the caller can fix them all at once."""
        err: list[str] = []
        for k in app:
            if not_used(k):
                continue
            if k in PROTECTED:
                err.append(f"{k}: protected attribute, must not be sent as a model input "
                           "(gender and age go in their own fields and only select the cut-off)")
            elif k in DERIVED:
                err.append(f"{k}: computed by the server from the raw fields, do not send it")
            elif k not in self.application_keys:
                err.append(f"{k}: unknown application field")
        for k in hist:
            if k not in self.history and not not_used(k):
                err.append(f"history.{k}: unknown credit-history feature")
        for k in NUMERIC_REQUIRED:
            if app.get(k) is None:
                err.append(f"{k}: required")
        for k in NUMERIC_NULLABLE_REQUIRED:
            if k not in app:
                err.append(f"{k}: required (send null if there is no current employment)")
        for g in CATEGORICAL_REQUIRED:
            if app.get(g) is None:
                err.append(f"{g}: required, one of {self.categories[g]}")
        for g in CATEGORICAL:
            v = app.get(g)
            if v is not None:
                if not isinstance(v, str):
                    err.append(f"{g}: must be text, one of {self.categories[g]}")
                elif mangle(v) not in self.categories[g] or mangle(v) == "nan":
                    err.append(f"{g}: unknown value {v!r}, one of {[c for c in self.categories[g] if c != 'nan']}")
        for k, v in list(app.items()) + list(hist.items()):
            if k in CATEGORICAL or v is None or not_used(k):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                err.append(f"{k}: must be a finite number")
        # Range checks run on every number that was supplied, so one response
        # lists missing fields and out-of-range values together.
        num = lambda d: {k: v for k, v in d.items()  # noqa: E731
                         if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                         and not not_used(k)}
        err += self._ranges(num(app), num(hist))
        return err

    def _ranges(self, app: dict, hist: dict) -> list[str]:
        err = []

        def check(k, ok, msg):
            v = app.get(k)
            if v is not None and not ok(v):
                err.append(f"{k}={v}: {msg}")

        for k in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
            check(k, lambda v: v > 0, "must be positive")
        if app.get("AMT_ANNUITY") and app.get("AMT_CREDIT") and app["AMT_ANNUITY"] > app["AMT_CREDIT"]:
            err.append("AMT_ANNUITY: instalment cannot exceed the loan amount")
        for k in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "REGION_POPULATION_RELATIVE"]:
            check(k, lambda v: 0 <= v <= 1, "must be between 0 and 1")
        for k in self.numeric_all:
            if k.startswith("DAYS_"):
                check(k, lambda v: v <= 0, "days are counted back from the application, so must be <= 0")
            if k.startswith(("FLAG_", "REG_", "LIVE_")):
                check(k, lambda v: v in (0, 1), "must be 0 or 1")
            if k.startswith(("CNT_", "OBS_", "DEF_", "AMT_REQ_", "OWN_CAR_AGE")):
                check(k, lambda v: v >= 0, "cannot be negative")
        for k in ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]:
            check(k, lambda v: v in (1, 2, 3), "must be 1, 2 or 3")
        if app.get("CNT_FAM_MEMBERS") is not None and app.get("CNT_CHILDREN") is not None \
                and app["CNT_FAM_MEMBERS"] < app["CNT_CHILDREN"] + 1:
            err.append("CNT_FAM_MEMBERS: must count the applicant plus their children")
        for k, v in hist.items():
            if v is None:
                continue
            if k.endswith("_SHARE") and not 0 <= v <= 1:
                err.append(f"history.{k}={v}: a share must be between 0 and 1")
            if k.endswith("_COUNT") and v < 0:
                err.append(f"history.{k}={v}: a count cannot be negative")
        return err

    # ------------------------------------------------------------------
    def encode(self, app: dict, hist: dict) -> tuple[np.ndarray, dict]:
        """Raw form -> the encoded row the model was trained on. Returns the
        row and a note of every default the server applied."""
        notes = {"ignored_fields": sorted(k for k in list(app) + list(hist) if not_used(k)),
                 "optional_categories_not_given": [], "history_tables_supplied": []}
        row = np.full(len(self.features), np.nan, dtype="float64")

        def put(name, val):
            if name in self.index:
                row[self.index[name]] = np.nan if val is None else float(val)

        num = {k: app.get(k) for k in self.numeric_all}
        num.update(derive(num))
        for k, v in num.items():
            put(k, v)

        cats = {g: app.get(g) for g in CATEGORICAL}
        for g, v in cats.items():
            levels = self.categories[g]
            for lv in levels:  # every level 0 first, exactly like get_dummies
                put(f"{g}_{lv}", 0)
            if v is None:  # only reachable for optional fields, which have a nan level
                put(f"{g}_nan", 1)
                notes["optional_categories_not_given"].append(g)
            else:
                put(f"{g}_{mangle(v)}", 1)

        for k, v in hist.items():
            if not not_used(k):
                put(k, v)
        notes["history_tables_supplied"] = sorted({k.split("_")[0] for k, v in hist.items()
                                                   if v is not None and not not_used(k)})
        return row, notes


def derive(num: dict) -> dict:
    """The application-level engineered features. Must equal
    data.clean_application (checked in tests/test_api.py)."""
    g = lambda k: np.nan if num.get(k) is None else float(num[k])  # noqa: E731
    inc, cr, ann, goods = g("AMT_INCOME_TOTAL"), g("AMT_CREDIT"), g("AMT_ANNUITY"), g("AMT_GOODS_PRICE")
    e1, e2, e3 = g("EXT_SOURCE_1"), g("EXT_SOURCE_2"), g("EXT_SOURCE_3")
    emp = g("DAYS_EMPLOYED")
    ext = [e for e in (e1, e2, e3) if not np.isnan(e)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "DAYS_EMPLOYED_ANOM": 1.0 if np.isnan(emp) else 0.0,
            "EXT_SOURCE_1_MISSING": float(np.isnan(e1)),
            "EXT_SOURCE_2_MISSING": float(np.isnan(e2)),
            "EXT_SOURCE_3_MISSING": float(np.isnan(e3)),
            "CREDIT_INCOME_RATIO": cr / inc,
            "ANNUITY_INCOME_RATIO": ann / inc,
            "CREDIT_TERM": ann / cr,
            "GOODS_CREDIT_RATIO": goods / cr,
            "EXT_SOURCE_MEAN": float(np.mean(ext)) if ext else np.nan,
            "EXT_SOURCE_PROD": e1 * e2 * e3,
        }
    return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in out.items()}


# ----------------------------------------------------------------------
# Reason codes: SHAP values are additive, so summing them over the columns
# that describe one thing (all three external scores and their mean, say)
# gives that thing's total effect. One reason per concept, no duplicates.
# ----------------------------------------------------------------------
REASON_GROUPS = [
    ("EXT_SOURCE", "External credit scores"),
    ("BUR_", "Credit bureau record (other lenders)"),
    ("INS_", "Past instalment payments"),
    ("PREV_", "Previous applications with this lender"),
    ("POS_", "Past point-of-sale and cash loans"),
    ("CC_", "Credit card usage"),
    ("AMT_REQ_CREDIT_BUREAU", "Recent credit enquiries"),
    (("AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"), "Loan and instalment relative to income"),
    (("AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CREDIT_TERM", "GOODS_CREDIT_RATIO", "NAME_CONTRACT_TYPE"),
     "Loan amount and repayment term"),
    (("DAYS_EMPLOYED", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "FLAG_EMP_PHONE"), "Employment"),
    ("NAME_EDUCATION_TYPE", "Education"),
    (("FLAG_OWN_CAR", "OWN_CAR_AGE"), "Car ownership"),
    (("NAME_HOUSING_TYPE", "FLAG_OWN_REALTY", "FONDKAPREMONT", "HOUSETYPE", "WALLSMATERIAL", "EMERGENCYSTATE",
      "TOTALAREA", "APARTMENTS", "BASEMENTAREA", "YEARS_BEGIN", "YEARS_BUILD", "COMMONAREA", "ELEVATORS",
      "ENTRANCES", "FLOORS", "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVING"), "Housing"),
    (("DAYS_REGISTRATION", "REGION_", "REG_", "LIVE_"), "Region and address"),
    (("CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_TYPE_SUITE"), "Household"),
]


def reason_group(feature: str) -> str:
    for pre, label in REASON_GROUPS:
        if feature.startswith(pre if isinstance(pre, tuple) else (pre,)):
            return label
    return "Other"


def request_from_encoded(c: Contract, enc: dict) -> tuple[dict, dict]:
    """Inverse of Contract.encode for a row of the training matrix: turns the
    encoded columns back into the raw form a client would send. Used by
    src/latency.py (real test applicants) and the round-trip test."""
    nan = lambda v: v is None or (isinstance(v, float) and np.isnan(v))  # noqa: E731
    app = {}
    for k in c.numeric_all:
        v = enc.get(k)
        if not nan(v):
            app[k] = float(v)
        elif k in NUMERIC_NULLABLE_REQUIRED:
            app[k] = None
    for g in CATEGORICAL:
        hot = [lv for lv in c.categories[g] if enc.get(f"{g}_{lv}") == 1]
        if hot and hot[0] != "nan":
            app[g] = hot[0]
    hist = {k: float(enc[k]) for k in c.history if not nan(enc.get(k))}
    return app, hist
