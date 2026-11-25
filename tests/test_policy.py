"""Improvements 1 and 6: the expected-profit decision rule and the Basel
capital formula."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import policy as P  # noqa: E402


def test_cutoff_is_where_expected_profit_crosses_the_offset():
    pol = {"type": "expected_profit", "economics": P.ECONOMICS, "offsets": {"g": -2000.0}}
    credit, annuity = np.array([2e5, 6e5, 1.2e6]), np.array([1.2e4, 3e4, 4.5e4])
    cut = P.cutoffs(pol, ["g"] * 3, credit, annuity)
    prof = P.expected_profit(cut, credit, annuity)
    assert np.allclose(prof, -2000.0, atol=1e-6)            # break-even at the cut-off
    assert (P.expected_profit(cut - 0.01, credit, annuity) > -2000).all()
    assert (P.expected_profit(cut + 0.01, credit, annuity) < -2000).all()


def test_cutoff_depends_on_loan_size_through_the_fixed_cost():
    pol = {"type": "expected_profit", "economics": P.ECONOMICS, "offsets": {"g": 0.0}}
    small, large = P.cutoffs(pol, ["g", "g"], [5e4, 5e5], [5e3, 5e4])  # same term, 10x the size
    assert small < large


def test_threshold_policy_still_works():
    pol = {"type": "group_thresholds", "thresholds": {"a": 0.2, "b": 0.1}}
    assert list(P.approve(pol, [0.15, 0.15], ["a", "b"])) == [True, False]
    assert P.covers(pol, ["a", "c"]) == ["c"]


def test_irb_capital_matches_the_basel_formula():
    from scipy.stats import norm
    pd_, lgd = 0.02, 0.45
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    R = 0.03 * w + 0.16 * (1 - w)
    K = lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)) - pd_ * lgd
    assert P.irb_retail_capital([pd_], lgd)[0] == pytest.approx(K)
    k = P.irb_retail_capital(np.linspace(0.001, 0.2, 50))
    assert (k > 0).all() and (k < 0.45).all() and (np.diff(k) > 0).all()  # rises with PD in this range
    assert P.irb_retail_capital([0.0])[0] == P.irb_retail_capital([0.0005])[0]  # PD floor


def test_gross_interest_of_an_amortising_loan():
    # 24 monthly payments at 3.33%: total repaid / principal from the annuity formula
    r, n = P.MONTHLY_RATE, 24
    a = r / (1 - (1 + r) ** -n)
    assert P.gross_interest([1.0], [a])[0] == pytest.approx(n * a - 1, rel=1e-9)
    assert P.gross_interest([1.0], [np.nan])[0] == 0.0
