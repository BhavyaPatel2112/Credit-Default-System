// Typed client for the FastAPI service (api/app.py and api/console.py).

export type Application = Record<string, number | string | null>;
export type Request = {
  application: Application;
  history: Record<string, number | null>;
  gender: "Female" | "Male";
  age_years: number;
};

export type Factor = { reason: string; effect: "raises risk" | "lowers risk"; shap_logodds: number; features: string[] };

export type Decision = {
  decision_id: string;
  probability_of_default: number;
  band: string;
  decision: "approve" | "refer";
  cutoff_applied: number;
  expected_profit: number | null;
  needs_human_review: boolean;
  adverse_action_reasons: Factor[];
  top_factors: Factor[];
  data_quality: {
    ignored_fields: string[];
    optional_categories_not_given: string[];
    history_tables_supplied: string[];
  };
  versions: Record<string, string>;
  latency_ms: number;
};

export type ApplicantRow = {
  id: number; gender: "Female" | "Male"; age: number; group: string;
  loan: number; annuity: number; income: number; contract: string | null; income_type: string | null;
  education: string | null; thin_file: boolean; pd: number; band: string; decision: "approve" | "refer";
  actual: "repaid" | "defaulted";
};

export type RecourseOption = { loan: number; annuity: number; loan_cut_pct: number; term_change_pct: number; pd: number; cutoff: number };
export type Recourse = {
  needed: boolean; found?: boolean; note?: string | null;
  current: { loan: number; annuity: number; pd: number; cutoff: number; approved: boolean };
  options: RecourseOption[];
};

export type GroupStat = {
  group: string; applicants: number; repayers: number; approval_among_repayers: number;
  approval_rate: number; default_among_approved: number | null; default_rate: number;
};
export type PolicySummary = {
  approval_rate: number; default_among_approved: number; defaulters_declined: number; expected_cost: number;
  gap_gender: number; gap_age: number; gap_intersection: number; groups: GroupStat[];
};
export type Fairness = {
  cost_ratio: number; cutoff: number; limit: number; n: number;
  single: PolicySummary; policy: PolicySummary & { thresholds: Record<string, number> };
  added_by_policy: { count: number; default_rate: number | null; mean_pd: number | null };
  removed_by_policy: number;
};
export type CurvePoint = { cost_ratio: number; cutoff: number; approval_rate: number; gap_gender: number; gap_age: number; default_among_approved: number };
export type FrontierPoint = { method: string; eps: number | null; gap: number; cost: number; approval: number; auc: number; uses_protected_at_decision: boolean };

export type ModelCard = {
  versions: Record<string, string>; n_features: number; policy: Record<string, number>;
  test: Record<string, number>; baseline: Record<string, number>;
  ladder: { run: string; label: string; auc: number; pr_auc: number; ece: number; gap_gender: number | null; gap_age: number | null }[];
  drivers: { reason: string; importance: number }[];
  latency: { p50_ms: number; p95_ms: number; p99_ms: number; n: number };
  data: { applicants: number; defaults: number; base_rate: number; has_bureau_history: number;
          by_gender: Record<string, { n: number; default_rate: number }>; by_age: Record<string, { n: number; default_rate: number }> };
  limitations: string[];
  policy_type?: string;
  economics?: { lgd: number; monthly_rate: number; margin_kept: number; fixed_cost: number };
  offsets?: Record<string, number>;
  challenger?: { method?: string; gate_passed?: boolean; test: ModelBlock; champion_xgboost: ModelBlock };
  no_protected_at_decision?: { approval_rate: number; roc_auc_of_mixture_score: number; eo_gap_gender: number; eo_gap_age: number; expected_cost: number } | null;
  group_calibration?: { group: string; mean_pd: number; observed: number; observed_ci: [number, number]; outside_ci: boolean }[];
  stress?: ({ scenario: string; mean_pd: number } & Record<string, { capital: number; approval_rate: number; gap: number } | string | number>)[];
  reject_inference?: { accept_vs_reject_auc: number | null; models: { model: string; test_auc: number; rejects_mean_pd: number }[] | null };
};
export type ModelBlock = { roc_auc: number; ece: number; approval_rate: number; eo_gap_gender: number; eo_gap_age: number; expected_cost_5to1: number };
export type GateCheck = { id: string; name: string; value: string | number; interval?: [number, number] | null; status: "pass" | "fail" | "skip" | "info" };

export type BatchRow = {
  row: number; applicant_id: string | null; status: "scored" | "rejected"; group?: string;
  probability_of_default?: number; band?: string; decision?: "approve" | "refer"; cutoff?: number; reasons?: string; problems?: string;
};
export type BatchResult = {
  rows: number; scored: number; approved: number; referred: number; rejected: number;
  file_notes: { unknown_columns: string[]; ignored_columns: string[]; missing_columns: string[] };
  rejected_rows: { row: number; line: number; applicant_id: string | null; problems: string[] }[];
  rejected_rows_truncated: boolean;
  groups: { group: string; scored: number; approved: number; approval_rate: number; repayers_known: number; approval_among_repayers: number | null; mean_pd: number }[];
  approval_rate_gap: number | null; repayer_approval_gap: number | null;
  results: BatchRow[]; results_csv: string;
};
export type LoggedDecision = {
  decision_id: string; time_utc: string; policy_group: string; probability_of_default: number;
  cutoff: number; decision: "approve" | "refer"; adverse_action_reasons: string[]; model_sha256: string;
};
export type Schema = {
  required_numbers: string[]; required_but_nullable: string[];
  required_categories: Record<string, string[]>; optional_categories: Record<string, string[]>;
  optional_numbers: string[]; history_features: string[]; policy_groups: string[];
};

export class ApiError extends Error {
  problems: string[];
  constructor(status: number, problems: string[]) {
    super(problems[0] ?? `Request failed (${status})`);
    this.problems = problems;
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { "content-type": "application/json" }, ...init });
  } catch {
    throw new ApiError(0, ["The scoring service is not reachable. Start it with: python -m uvicorn api.app:app --port 8000"]);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = body?.detail;
    const problems = Array.isArray(d) ? d.map((x: unknown) => (typeof x === "string" ? x : JSON.stringify(x))) : [String(d ?? res.statusText)];
    throw new ApiError(res.status, problems);
  }
  return body as T;
}

export const api = {
  health: () => call<{ status: string; versions: Record<string, string> }>("/health"),
  schema: () => call<Schema>("/schema"),
  score: (r: Request) => call<Decision>("/score", { method: "POST", body: JSON.stringify(r) }),
  recourse: (r: Request) => call<Recourse>("/api/recourse", { method: "POST", body: JSON.stringify(r) }),
  applicants: () => call<{ available: boolean; note: string; applicants: ApplicantRow[] }>("/api/applicants"),
  applicant: (id: number) => call<{ id: number; actual: "repaid" | "defaulted"; request: Request }>(`/api/applicants/${id}`),
  fairness: (r: number) => call<Fairness>(`/api/fairness?cost_ratio=${r}`),
  curve: () => call<{ limit: number; points: CurvePoint[] }>("/api/fairness/curve"),
  frontier: () => call<{ points: FrontierPoint[]; chosen: { method: string; eps: number }; limit: number }>("/api/frontier"),
  modelCard: () => call<ModelCard>("/api/model-card"),
  gate: () => call<{ passed: boolean; checks: GateCheck[] }>("/api/gate"),
  decisions: (n = 25) => call<{ decisions: LoggedDecision[] }>(`/api/decisions?limit=${n}`),
  batch: (csv: string, filename?: string) => call<BatchResult>("/api/batch", { method: "POST", body: JSON.stringify({ csv, filename }) }),
  batchSample: async () => {
    const r = await fetch("/api/batch/sample");
    if (!r.ok) throw new ApiError(r.status, ["Could not fetch the sample file"]);
    return r.text();
  },
};

// A complete, valid application used when no real applicants are available
// (same as demo/example_request.json).
export const EXAMPLE: Request = {
  application: {
    AMT_INCOME_TOTAL: 202500, AMT_CREDIT: 406597.5, AMT_ANNUITY: 24700.5, AMT_GOODS_PRICE: 351000,
    CNT_CHILDREN: 0, CNT_FAM_MEMBERS: 1, REGION_POPULATION_RELATIVE: 0.0188,
    REGION_RATING_CLIENT: 2, REGION_RATING_CLIENT_W_CITY: 2, DAYS_REGISTRATION: -3648,
    DAYS_EMPLOYED: -637, FLAG_EMP_PHONE: 1, REG_REGION_NOT_LIVE_REGION: 0, REG_REGION_NOT_WORK_REGION: 0, LIVE_REGION_NOT_WORK_REGION: 0,
    REG_CITY_NOT_LIVE_CITY: 0, REG_CITY_NOT_WORK_CITY: 0, LIVE_CITY_NOT_WORK_CITY: 0,
    NAME_CONTRACT_TYPE: "Cash loans", FLAG_OWN_CAR: "N", FLAG_OWN_REALTY: "Y", NAME_INCOME_TYPE: "Working",
    NAME_EDUCATION_TYPE: "Higher education", NAME_HOUSING_TYPE: "House / apartment",
    ORGANIZATION_TYPE: "Business Entity Type 3", EXT_SOURCE_1: 0.7, EXT_SOURCE_2: 0.7, EXT_SOURCE_3: 0.7,
  },
  history: {},
  gender: "Female",
  age_years: 29,
};
