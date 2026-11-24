# Parity: credit default prediction with fairness constraints (CAP 501)

A governed credit-scoring pipeline on the Home Credit Default Risk data. It covers the run ladder from the
mid-term plan: data assembly with leakage control, XGBoost, calibration, fairness mitigation, SHAP and DiCE,
a FastAPI service, a CI fairness gate, drift monitoring, and a web console for loan officers and reviewers.

Release 1.3 adds the eight improvements of the October 2026 plan: an expected-profit decision rule (each loan's
expected loss against its interest margin) in place of the fixed 5:1 cost ratio, a fairness method that needs no
protected attribute at decision time, a scorecard challenger, bootstrap intervals in the gate and calibration by
group, removal of indefensible features, a stress test with Basel IRB capital, reject inference, and batch
scoring.

## Quick start: run the console

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # if Anaconda is active, run `conda deactivate` first
pip install -r requirements.txt
python -m uvicorn api.app:app --port 8000
```

Open <http://127.0.0.1:8000>. The console is already built (`web/dist`), so Node is not needed to run it.

| Page | What it does |
| --- | --- |
| Home | Live decision on a real held-out applicant over the "risk landscape", the decision log, the test results |
| Assess | Pick one of 400 real test applicants or fill in a new application; see the decision, the notice the applicant receives, the factors, and the smallest change that gets it approved |
| Batch | Upload a CSV of applications: every row validated and scored like a single decision, rejected rows explained, approval by group, results CSV to download |
| Fairness | Move the cost of a default and watch the single cut-off and the deployed policy compare, group by group; every policy tried on one frontier |
| Model | The release gate run live with its intervals, the run ladder, the deployed policy and its economics, the scorecard challenger, calibration by group, stress test and capital, reject inference, limitations and the decision log |

The applicant picker needs `data/processed/demo_applicants.parquet` (built from the raw data, step 11 below).
Without it, the console still works with the new-application form.

## Setup

Python 3.12 or newer (xgboost 3.4.1 and shap 0.53.0 publish no wheels for 3.11).

```bash
pip install -r requirements.txt
# put the 7 Kaggle CSVs in data/raw/  (application_train, bureau, bureau_balance,
# previous_application, POS_CASH_balance, installments_payments, credit_card_balance)
```

## Run order

| Step | Command | Milestone | Writes |
| --- | --- | --- | --- |
| 1 | `python src/data.py` | M2/M4 | `data/processed/applicants.parquet`, `transformation_log.json` |
| 2 | `python src/prepare.py` | M2–M4 | EDA, proxy audit, features, frozen splits + checksums |
| 3 | `python src/eda_figures.py` | M2 | `reports/figures/eda_*.png` |
| 4 | `python src/train.py` | M5 | `models/xgb.json`, `models/calibrator.pkl`, `runs/r0–r2b*.json`, `models/manifest.json` |
| 5 | `python src/proxy_ablation.py` | M3 | `runs/r2c_without_flagged_proxies.json` |
| 6 | `python src/scorecard.py` | Imp. 3 | `models/scorecard.pkl`, `reports/scorecard_points.csv`, `runs/r9_scorecard.json` |
| 7 | `python src/profit_policy.py` | Imp. 1 | **`models/fair_policy.json` (deployed)**, `runs/r8_expected_profit.json`, `runs/expected_profit_sensitivity.csv` |
| 8 | `python src/fairness.py` | M6, Imp. 2 | `runs/frontier.csv`, `models/fair_policy_5to1.json`, `runs/r3_no_protected_at_decision.json`, `runs/r4_operating_point.json`, frontier figure (the reductions runs take about 10 minutes and are cached in `runs/reductions_cache_v2.pkl`) |
| 9 | `python src/calibration_groups.py` | Imp. 4 | `runs/r10_group_calibration.json` |
| 10 | `python src/stress_capital.py` | Imp. 6 | `runs/r11_stress_capital.json`, `reports/figures/stress_capital.png` |
| 11 | `python src/reject_inference.py` | Imp. 7 | `runs/r12_reject_inference.json` |
| 12 | `python src/explain.py` | FR2/FR3 | `runs/r5_explainability.json`, `runs/recourse.csv`, SHAP figure |
| 13 | `python src/monitor.py` | M8 | `runs/r6_monitoring.json`, drift figure |
| 14 | `python src/latency.py` | M7 | `runs/r7_latency.json` |
| 15 | `python src/make_gate_sample.py` | M8 | `data/processed/gate_sample.parquet` (enables gate check G3) |
| 16 | `python src/make_demo_applicants.py` | M7 | `data/processed/demo_applicants.parquet` (the console's applicant picker) |
| 17 | `python src/gate.py` | M8 | exit 0 = pass, 1 = fail |
| 18 | `python -m pytest -q tests` | R-2, M7, M8 | leakage, scoring-service, policy, batch, console and gate tests (60, all passing with the raw data present) |

Steps 4 to 14 retrain and rewrite committed results; the console and the gate only need steps 1, 2, 15 and 16.
Run them in this order: steps 6 and 7 write `runs/frontier_extra.csv`, which step 8 plots, and steps 9 to 16 read
the deployed policy written in step 7.
`gate_sample.parquet` and `demo_applicants.parquet` are derived from the Kaggle data: keep them out of any public repository.

`demo/run_gate_demo.sh` shows the gate passing the committed policy and rejecting a deliberately unfair one.
`.github/workflows/gate.yml` runs the tests, the gate and a container build on every pull request.

### What the gate checks

- **G1** `models/manifest.json`: the model, calibrator, feature list and saved predictions were produced together. Swapping the model without regenerating predictions fails.
- **G2** the committed calibrator reproduces the saved calibrated PDs.
- **G3** the committed booster re-scores a frozen sample of test applicants (needs `gate_sample.parquet`; reported as NOT RUN until it is committed).
- **POL** every gender x age group has a cut-off (a threshold, or an offset under the expected-profit policy).
- **NFR1-NFR3, R-4** AUC, ECE, gender and age equal-opportunity gaps, cell sizes. Since 1.3 each is checked on the
  unfavourable end of its bootstrap 95% interval (1,000 resamples, fixed seed): the lower bound for AUC, the upper bound
  for ECE and the gaps. The four-group gap is reported, not gated.

## Scoring service

`uvicorn api.app:app --port 8000`, interactive docs at `/docs`.

- `GET /schema` lists every field: required numbers, required categories and their allowed values, optional fields, and the credit-history features.
- `POST /score` takes the **raw application form** (`application`), optional credit-history aggregates (`history`, empty for a thin-file applicant), and `gender` / `age_years`, which only select the policy group's offset.
- The cut-off is the loan's own: approve when (1 − PD) × margin × loan − PD × 45% × loan − 5,000 is at least the group's offset (`src/policy.py`). The response gives `cutoff_applied` and `expected_profit`.
- Fields the model no longer uses (time of application, contact details, document flags, social-circle defaults) are accepted, ignored and listed in `data_quality.ignored_fields`.
- The service validates the form (required fields, ranges, unknown or protected or derived fields), then builds the 449-column row itself with the same formulas as training (`api/features.py`, checked against `src/data.py` in the tests). A request it cannot score honestly gets HTTP 422 with every problem listed.
- The response has the PD, band, decision (`approve` or `refer`, never decline), the cut-off, `adverse_action_reasons` (on a referral: up to four reasons that raised the risk, one per concept), `top_factors` for the loan officer, `data_quality` notes and the model / policy hashes.
- Every decision is appended to `logs/decisions.jsonl` (or `$CAP501_DECISION_LOG`) with the model and policy hashes and a hash of the input, never the raw application.

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d @demo/example_request.json
```

## Web console (`web/`)

React 19 and TypeScript, built with Vite; Tailwind CSS v4; Motion for animation; NumberFlow for animated figures;
Recharts for charts; Radix UI primitives (slider, switch); TanStack Query for server state; d3-contour and
simplex-noise for the contour background. Type: Instrument Serif and Geist.

```bash
cd web && npm install
npm run dev      # http://localhost:5173, forwards API calls to the service on :8000
npm run build    # writes web/dist, which the service serves at /
```

The console's endpoints live in `api/console.py`: `/api/applicants`, `/api/recourse`, `/api/fairness`,
`/api/fairness/curve`, `/api/frontier`, `/api/model-card`, `/api/gate`, `/api/decisions`; batch scoring in
`api/batch.py`: `POST /api/batch` (`{"csv": "<file text>"}`, up to 5,000 rows) and `GET /api/batch/sample`.

Batch CSV format: a header row of field names as in `GET /schema`, plus `gender` and `age_years`; optional
`applicant_id` (echoed back) and `TARGET` (a known outcome, used only to show repayer approval by group). An empty
cell means "not given". Each scored row is logged with `source: "batch"`.

## Split roles

- **train:** fits the booster. Early stopping uses a 15% inner hold-out of train.
- **valid:** fits the isotonic calibrator and the fairness cut-offs.
- **test:** used only for reporting.

Splits are stratified on default × gender, seed 42, and the SHA-256 of each split is in `data/processed/splits_manifest.json`.
