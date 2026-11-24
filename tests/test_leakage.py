"""R-2: the temporal cut-off is unit-tested. No record dated on or after the
application may survive, and observed outcome dates after it are nulled."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import RAW, SENTINEL, apply_cutoff  # noqa: E402


def test_monthly_tables_drop_current_and_future_months():
    df = pd.DataFrame({"SK_ID_CURR": [1, 1, 1, 1], "MONTHS_BALANCE": [-3, -1, 0, 2]})
    for t in ["POS_CASH_balance", "credit_card_balance", "bureau_balance"]:
        out = apply_cutoff(df, t)
        assert out.MONTHS_BALANCE.max() < 0
        assert len(out) == 2


def test_installments_drop_payments_after_application():
    df = pd.DataFrame({"SK_ID_CURR": [1, 1, 1], "DAYS_INSTALMENT": [-40, -10, -5],
                       "DAYS_ENTRY_PAYMENT": [-38, 3, np.nan]})
    out = apply_cutoff(df, "installments_payments")
    assert len(out) == 2                       # the payment made on day +3 is gone
    assert not (out.DAYS_ENTRY_PAYMENT >= 0).any()


def test_bureau_observed_dates_after_application_are_nulled():
    df = pd.DataFrame({"SK_ID_CURR": [1, 1], "DAYS_CREDIT": [-500, -100],
                       "DAYS_ENDDATE_FACT": [-50, 20], "DAYS_CREDIT_UPDATE": [-5, 7]})
    out = apply_cutoff(df, "bureau")
    assert out.DAYS_ENDDATE_FACT.isna().sum() == 1
    assert out.DAYS_CREDIT_UPDATE.isna().sum() == 1


def test_previous_application_sentinel_is_not_treated_as_a_date():
    df = pd.DataFrame({"SK_ID_CURR": [1], "DAYS_DECISION": [-300], "DAYS_LAST_DUE": [SENTINEL],
                       "DAYS_TERMINATION": [-10]})
    out = apply_cutoff(df, "previous_application")
    assert out.DAYS_LAST_DUE.iloc[0] == SENTINEL  # decoded separately in build_previous


@pytest.mark.skipif(not (RAW / "installments_payments.csv").exists(), reason="raw data not present")
def test_real_tables_have_nothing_after_application():
    for t, col in [("POS_CASH_balance", "MONTHS_BALANCE"), ("credit_card_balance", "MONTHS_BALANCE"),
                   ("installments_payments", "DAYS_ENTRY_PAYMENT")]:
        df = pd.read_csv(RAW / f"{t}.csv", usecols=["SK_ID_CURR", col], nrows=500_000)
        out = apply_cutoff(df, t)
        assert not (out[col] >= 0).any()
