"""Decision policies and the loan economics behind them (improvements 1 and 6).

One module, used by training, the gate, the scoring service and the console,
so every part of the system turns a probability of default into a decision
the same way.

Policy files (models/fair_policy.json)
--------------------------------------
group_thresholds   {"thresholds": {group: cut-off}}  (or {"all": cut-off})
                   approve when PD < the group's cut-off. The 1.0-1.2 policy.
expected_profit    {"economics": {...}, "offsets": {group: h}}
                   approve when the loan's expected profit is at least h_g.

Expected profit of approving a loan (currency units)
    E[profit] = (1 - PD) * m * EAD  -  PD * LGD * EAD  -  c
EAD  exposure at default: the loan amount (AMT_CREDIT).
LGD  loss given default: 45%, the Basel foundation-IRB supervisory LGD for
     senior unsecured exposures (BCBS, Basel framework CRE32.6). Retail has no
     supervisory LGD; the report gives a 25% to 75% sensitivity range.
m    net lifetime interest margin per unit lent: the loan's own gross
     lifetime interest, from its instalment and the monthly rate implied by
     312,536 approved previous cash loans in previous_application (median
     3.33%), times the share the lender keeps after funding, running costs
     and interest lost on loans that default part-way (MARGIN_KEPT, stated
     assumption, sensitivity 15% to 40%).
c    fixed cost of originating and servicing one loan (stated assumption).
So the PD cut-off of loan i is
    PD*_i = (m_i * EAD_i - c - h_g) / ((m_i + LGD) * EAD_i)
which depends on the loan's size (through c) and its term (through m_i).
With fairness on top, h_g <= 0 for some groups: the lender accepts a slightly
lower expected profit on those applicants ("levelling up").
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# --- stated assumptions (report section 6 lists them with their sources) ----
LGD = 0.45
LGD_RANGE = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
MONTHLY_RATE = 0.0333        # median implied rate, approved previous cash loans
MARGIN_KEPT = 0.25           # share of gross interest the lender keeps
MARGIN_KEPT_RANGE = [0.15, 0.25, 0.40]
FIXED_COST = 5000.0          # per loan, about 1% of the median loan (513,531)
MAX_PAYMENTS = 60            # cap when the instalment barely covers interest

ECONOMICS = {"lgd": LGD, "monthly_rate": MONTHLY_RATE, "margin_kept": MARGIN_KEPT, "fixed_cost": FIXED_COST}


def group_key(gender, age_under_35) -> np.ndarray:
    """Intersectional policy group: 'Female|Under 35' etc."""
    return (np.asarray(gender).astype(str).astype(object) + "|" + np.asarray(age_under_35).astype(str)).astype(str)


def key_of(gender: str, age_years: float) -> str:
    return f"{gender}|{'Under 35' if age_years < 35 else '35 and over'}"


def gross_interest(credit, annuity, monthly_rate: float = MONTHLY_RATE) -> np.ndarray:
    """Lifetime interest per unit lent for an amortising loan with this
    instalment: n = -ln(1 - r/a) / ln(1 + r) payments of a = annuity/credit."""
    a = np.asarray(annuity, dtype=float) / np.asarray(credit, dtype=float)
    r = monthly_rate
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(a > r, -np.log1p(-r / a) / np.log1p(r), MAX_PAYMENTS)
    n = np.clip(np.nan_to_num(n, nan=MAX_PAYMENTS), 1, MAX_PAYMENTS)
    # no instalment on file (12 training rows; the service rejects them): no margin
    return np.nan_to_num(np.maximum(n * a - 1.0, 0.0), nan=0.0)


def margin(credit, annuity, econ: dict = ECONOMICS) -> np.ndarray:
    return econ["margin_kept"] * gross_interest(credit, annuity, econ["monthly_rate"])


def expected_profit(pd_, credit, annuity, econ: dict = ECONOMICS) -> np.ndarray:
    ead = np.asarray(credit, dtype=float)
    m = margin(credit, annuity, econ)
    p = np.asarray(pd_, dtype=float)
    return (1 - p) * m * ead - p * econ["lgd"] * ead - econ["fixed_cost"]


def realised_profit(y, credit, annuity, econ: dict = ECONOMICS) -> np.ndarray:
    """What approving the loan actually earned, given the observed outcome."""
    ead = np.asarray(credit, dtype=float)
    m = margin(credit, annuity, econ)
    y = np.asarray(y)
    return np.where(y == 1, -econ["lgd"] * ead, m * ead) - econ["fixed_cost"]


# ---------------------------------------------------------------- policies
def groups(policy: dict) -> set[str]:
    if policy.get("type") == "expected_profit":
        return set(policy["offsets"])
    return set(policy["thresholds"])


def covers(policy: dict, keys) -> list[str]:
    """Policy groups that are needed but missing (empty = every key covered)."""
    have = groups(policy)
    if "all" in have:
        return []
    return sorted({k for k in np.asarray(keys) if k not in have})


def cutoffs(policy: dict, keys, credit=None, annuity=None) -> np.ndarray:
    """The PD cut-off applied to each applicant (approve when PD < cut-off)."""
    keys = np.asarray(keys)
    if policy.get("type") == "expected_profit":
        econ = policy["economics"]
        ead = np.asarray(credit, dtype=float)
        m = margin(credit, annuity, econ)
        h = np.array([policy["offsets"][k] for k in keys], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (m * ead - econ["fixed_cost"] - h) / ((m + econ["lgd"]) * ead)
        return np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    th = policy["thresholds"]
    return np.array([th.get(k, th.get("all")) for k in keys], dtype=float)


def cutoff_one(policy: dict, key: str, credit: float, annuity: float) -> float:
    return float(cutoffs(policy, [key], [credit], [annuity])[0])


def approve(policy: dict, pd_, keys, credit=None, annuity=None) -> np.ndarray:
    return np.asarray(pd_) < cutoffs(policy, keys, credit, annuity)


# ---------------------------------------------------------------- capital
def irb_retail_capital(pd_, lgd: float = LGD, pd_floor: float = 0.0005) -> np.ndarray:
    """Basel IRB risk-weight function for 'other retail' exposures (BCBS,
    CRE31.15): capital requirement K per unit of EAD at 99.9% confidence,
    asset correlation between 3% and 16% falling with PD, no maturity
    adjustment for retail. PD floor 0.05% (CRE32.29 for retail). RWA =
    12.5 * K * EAD; minimum capital = 8% of RWA = K * EAD."""
    p = np.clip(np.asarray(pd_, dtype=float), pd_floor, 0.9999)
    w = (1 - np.exp(-35 * p)) / (1 - np.exp(-35))
    R = 0.03 * w + 0.16 * (1 - w)
    cond = norm.cdf((norm.ppf(p) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R))
    return np.maximum(lgd * cond - lgd * p, 0.0)
