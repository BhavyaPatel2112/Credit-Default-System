import { motion } from "motion/react";
import { useQuery } from "@tanstack/react-query";
import { Check, Minus, X } from "lucide-react";
import { api } from "../lib/api";
import { groupLabel, money, pct, timeAgo } from "../lib/format";
import type { ModelCard } from "../lib/api";
import { Headline } from "../components/Headline";

const GATE_TEXT: Record<string, string> = {
  G1: "The model files are the ones that produced the saved predictions",
  G2: "The calibrator reproduces the saved probabilities",
  G3: "The model re-scores a frozen sample of real applicants identically",
  POL: "Every gender and age group has a cut-off",
  NFR1: "Ranks defaulters above repayers well enough, even at the low end of its 95% interval",
  NFR2: "Predicted risk matches observed default, even at the high end of its 95% interval",
  "R-4": "Every group has enough repayers to measure",
  INFO: "Gap across the four gender and age groups",
};

export function Model() {
  const m = useQuery({ queryKey: ["model-card"], queryFn: api.modelCard });
  const g = useQuery({ queryKey: ["gate"], queryFn: api.gate });
  const d = m.data;
  return (
    <div className="mx-auto max-w-[1240px] px-4 pt-16 sm:px-6 sm:pt-20">
      <Headline quiet="How the model was built," loud="and how it is held to account." size="lg" className="max-w-4xl" />

      <section className="mt-16 grid gap-12 lg:grid-cols-[1fr_1.1fr] [&>*]:min-w-0">
        <div>
          <div className="flex items-baseline justify-between">
            <h2 className="text-[15px] font-medium">Release gate, run now</h2>
            {g.data && <span className={`text-[14px] ${g.data.passed ? "text-moss" : "text-mustard"}`}>{g.data.passed ? "Passed" : "Failed"}</span>}
          </div>
          <p className="mt-1 text-[14px] text-soft">The same checks run on every pull request. A failure blocks the merge. Each requirement
            is checked on the unfavourable end of a 1,000-resample bootstrap interval, so a pass cannot be luck of the sample.</p>
          <ul className="mt-5 divide-y divide-line border-y border-line">
            {(g.data?.checks ?? []).map((c, i) => {
              const text = c.id === "NFR3" ? (c.name.includes("gender") ? "Repayer approval gap between genders, 5 points or less at the top of its interval" : "Repayer approval gap between age groups, 5 points or less at the top of its interval") : GATE_TEXT[c.id] ?? c.name;
              const gap = c.id === "NFR3" || c.id === "INFO";
              const f = (v: number) => (gap ? (v * 100).toFixed(1) : v.toFixed(3));
              const val = typeof c.value === "number"
                ? (c.id === "R-4" ? c.value.toLocaleString("en-IN") : `${f(c.value)}${gap ? " pts" : ""}${c.interval ? ` (${f(c.interval[0])} to ${f(c.interval[1])})` : ""}`)
                : "";
              return (
                <motion.li key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.04 }} className="flex items-center gap-3 py-3 text-[14px]">
                  <Status s={c.status} />
                  <span className="flex-1">{text}</span>
                  <span className="text-soft tnum">{val}</span>
                </motion.li>
              );
            })}
            {g.isLoading && <li className="py-6 text-soft">Running the checks</li>}
          </ul>
        </div>

        <div>
          <h2 className="text-[15px] font-medium">How each step changed the model</h2>
          <p className="mt-1 text-[14px] text-soft">On the 46,127 held-out applicants. Gaps are in points of repayer approval.</p>
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[520px] text-left text-[14px]">
              <thead className="text-[12px] text-mute">
                <tr><th className="pb-2 font-normal">Step</th><th className="pb-2 text-right font-normal">Area under curve</th><th className="pb-2 text-right font-normal">Calibration error</th><th className="pb-2 text-right font-normal">Gender gap</th><th className="pb-2 text-right font-normal">Age gap</th></tr>
              </thead>
              <tbody>
                {(d?.ladder ?? []).map((r) => (
                  <tr key={r.run} className={`border-t border-line ${r.run === "r8" ? "text-ink" : "text-soft"}`}>
                    <td className="py-3 pr-3"><span className={r.run === "r8" ? "font-medium" : ""}>{r.label}</span></td>
                    <td className="py-3 text-right tnum">{r.auc?.toFixed(3)}</td>
                    <td className={`py-3 text-right tnum ${r.ece > 0.05 ? "text-mustard" : ""}`}>{r.ece?.toFixed(3)}</td>
                    <td className="py-3 text-right tnum">{r.run === "r2" || r.gap_gender == null ? "Not shown" : (r.gap_gender * 100).toFixed(1)}</td>
                    <td className={`py-3 text-right tnum ${r.gap_age != null && r.gap_age > 0.05 && r.run !== "r2" ? "text-mustard" : ""}`}>{r.run === "r2" || r.gap_age == null ? "Not shown" : (r.gap_age * 100).toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[13px] text-mute">Class weighting kept the ranking but broke the probabilities; isotonic calibration restored them. Gaps for that step are measured on broken probabilities, so they are not shown.</p>
        </div>
      </section>

      <section className="mt-20 grid gap-12 lg:grid-cols-2 [&>*]:min-w-0">
        <div>
          <h2 className="text-[15px] font-medium">What the model leans on</h2>
          <p className="mt-1 text-[14px] text-soft">Average influence across 4,000 test applicants, grouped by concept.</p>
          <ul className="mt-5 space-y-3">
            {(d?.drivers ?? []).slice(0, 10).map((x, i, all) => (
              <li key={x.reason} className="grid grid-cols-[1fr_1.1fr] items-center gap-4 text-[14px]">
                <span>{x.reason}</span>
                <div className="h-2 rounded-full bg-stone-2">
                  <motion.div className="h-2 rounded-full bg-ink" initial={{ width: 0 }} animate={{ width: `${(x.importance / all[0].importance) * 100}%` }} transition={{ duration: 0.8, delay: 0.2 + i * 0.04 }} />
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-[13px] text-mute">The external credit scores dominate. They come from undocumented third parties, which limits how far an applicant can act on them.</p>
        </div>

        <div>
          <h2 className="text-[15px] font-medium">The data</h2>
          <p className="mt-1 text-[14px] text-soft">
            {d?.data.applicants?.toLocaleString("en-IN")} Home Credit applications across seven tables; {pct(d?.data.base_rate)} defaulted.
            Risk differs by group before any model is involved.
          </p>
          <div className="mt-5 space-y-4">
            {d?.data.by_age && ["<25", "25-34", "35-44", "45-54", "55+"].filter((k) => d.data.by_age[k]).map((k) => [k, d.data.by_age[k]] as const).map(([k, v]) => <Bar key={k} label={k === "<25" ? "Under 25" : k === "55+" ? "55 and over" : k.replace("-", " to ")} v={v.default_rate} />)}
            <div className="h-px bg-line" />
            {d?.data.by_gender && Object.entries(d.data.by_gender).map(([k, v]) => <Bar key={k} label={k === "Female" ? "Women" : "Men"} v={v.default_rate} />)}
          </div>
        </div>
      </section>

      <section className="mt-20 grid gap-12 lg:grid-cols-2 [&>*]:min-w-0">
        <div>
          <h2 className="text-[15px] font-medium">Deployed policy</h2>
          <p className="mt-1 text-[14px] text-soft">
            Approve when the loan's expected profit, (1 − PD) × margin × loan − PD × loss given default × loan − fixed cost,
            is at least the group's offset; otherwise refer to a person. So every loan has its own cut-off; the median per group is shown.
          </p>
          <dl className="mt-5 divide-y divide-line border-y border-line text-[14px]">
            {d && Object.entries(d.policy).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4 py-3"><dt>{groupLabel(k)}</dt>
                <dd className="text-right tnum">{pct(v)}{d.offsets?.[k] != null && <span className="ml-3 text-soft">offset {money(d.offsets[k])}</span>}</dd></div>
            ))}
          </dl>
          {d?.economics && (
            <p className="mt-3 text-[13px] text-mute">
              Loss given default {pct(d.economics.lgd, 0)} (Basel supervisory value for unsecured loans), monthly rate {pct(d.economics.monthly_rate, 2)} implied
              by previous loans, {pct(d.economics.margin_kept, 0)} of interest kept after costs, fixed cost {money(d.economics.fixed_cost)} per loan.
            </p>
          )}
          <h2 className="mt-10 text-[15px] font-medium">Fingerprints of this release</h2>
          <dl className="mt-3 divide-y divide-line border-y border-line text-[14px]">
            {d && Object.entries(d.versions).map(([k, v]) => (
              <div key={k} className="flex justify-between py-3"><dt className="capitalize">{k.replace("_sha256", "")}</dt><dd className="text-soft tnum">{v}</dd></div>
            ))}
          </dl>
        </div>
        <div>
          <h2 className="text-[15px] font-medium">Known limitations</h2>
          <ul className="mt-5 space-y-4">
            {(d?.limitations ?? []).map((l) => (
              <li key={l} className="flex gap-3 text-[15px] leading-relaxed text-soft"><span className="mt-2.5 h-1 w-3 shrink-0 bg-mustard" />{l}</li>
            ))}
          </ul>
        </div>
      </section>

      {d && <Challengers d={d} />}

      <Recent />
    </div>
  );
}

function Status({ s }: { s: "pass" | "fail" | "skip" | "info" }) {
  const cls = { pass: "bg-moss text-white", fail: "bg-mustard text-white", skip: "bg-stone-3 text-soft", info: "bg-stone-2 text-soft" }[s];
  const Icon = s === "pass" ? Check : s === "fail" ? X : Minus;
  return <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full ${cls}`} title={s}><Icon className="h-3 w-3" strokeWidth={3} /></span>;
}

function Bar({ label, v }: { label: string; v: number }) {
  return (
    <div className="grid grid-cols-[110px_1fr_52px] items-center gap-3 text-[14px]">
      <span>{label}</span>
      <div className="h-2 rounded-full bg-stone-2"><motion.div className="h-2 rounded-full bg-ink/80" initial={{ width: 0 }} animate={{ width: `${(v / 0.14) * 100}%` }} transition={{ duration: 0.8, delay: 0.2 }} /></div>
      <span className="text-right text-soft tnum">{pct(v)}</span>
    </div>
  );
}

function Recent() {
  const q = useQuery({ queryKey: ["decisions", 25], queryFn: () => api.decisions(25), refetchInterval: 10000 });
  const rows = q.data?.decisions ?? [];
  return (
    <section className="mt-20">
      <h2 className="text-[15px] font-medium">Decision log</h2>
      <p className="mt-1 text-[14px] text-soft">The latest decisions from logs/decisions.jsonl. Refreshes every ten seconds.</p>
      {rows.length === 0 ? (
        <p className="mt-6 text-soft">No decisions logged yet. <a href="#/assess" className="text-ink underline underline-offset-4">Assess an applicant</a> to add one.</p>
      ) : (
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-[14px]">
            <thead className="text-[12px] text-mute"><tr>
              <th className="pb-2 font-normal">When</th><th className="pb-2 font-normal">Group</th><th className="pb-2 text-right font-normal">Probability</th><th className="pb-2 text-right font-normal">Cut-off</th><th className="pb-2 font-normal pl-6">Decision</th><th className="pb-2 font-normal">Main reason</th><th className="pb-2 text-right font-normal">Model</th>
            </tr></thead>
            <tbody>
              {rows.map((d) => (
                <tr key={d.decision_id} className="border-t border-line">
                  <td className="py-3 text-soft">{timeAgo(d.time_utc)}</td>
                  <td className="py-3">{groupLabel(d.policy_group)}</td>
                  <td className="py-3 text-right tnum">{pct(d.probability_of_default)}</td>
                  <td className="py-3 text-right text-soft tnum">{pct(d.cutoff)}</td>
                  <td className={`py-3 pl-6 ${d.decision === "approve" ? "text-moss" : "text-mustard"}`}>{d.decision === "approve" ? "Approved" : "Referred"}</td>
                  <td className="py-3 text-soft">{d.adverse_action_reasons[0] ?? ""}</td>
                  <td className="py-3 text-right text-soft tnum">{d.model_sha256.slice(0, 8)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Challengers({ d }: { d: ModelCard }) {
  const ch = d.challenger;
  const policies = ["single cut-off 1/6", "5:1 group cut-offs (1.2)", "expected profit, no fairness", "expected profit + offsets (deployed)"];
  const cell = (s: Record<string, unknown>, n: string) => s[n] as { capital: number; approval_rate: number; gap: number } | undefined;
  return (
    <>
      <section className="mt-20 grid gap-12 lg:grid-cols-2 [&>*]:min-w-0">
        {ch?.test && ch?.champion_xgboost && (
          <div>
            <h2 className="text-[15px] font-medium">Challenger: a traditional bank scorecard</h2>
            <p className="mt-1 text-[14px] text-soft">Weight-of-evidence bins and logistic regression in points (600 at 50:1 odds, 20 points to double them), same splits, same gate.</p>
            <table className="mt-5 w-full text-left text-[14px]">
              <thead className="text-[12px] text-mute"><tr><th className="pb-2 font-normal" /><th className="pb-2 text-right font-normal">XGBoost (champion)</th><th className="pb-2 text-right font-normal">Scorecard</th></tr></thead>
              <tbody>
                {([["Area under curve", "roc_auc", 3], ["Calibration error", "ece", 3], ["Approved", "approval_rate", -1], ["Gender gap, points", "eo_gap_gender", -2], ["Age gap, points", "eo_gap_age", -2], ["Cost per applicant (5:1)", "expected_cost_5to1", 3]] as const).map(([k, f, dp]) => {
                  const fmt = (v: number) => (dp === -1 ? pct(v) : dp === -2 ? (v * 100).toFixed(1) : v.toFixed(dp));
                  return <tr key={k} className="border-t border-line"><td className="py-2.5">{k}</td><td className="py-2.5 text-right tnum">{fmt(ch.champion_xgboost[f])}</td><td className="py-2.5 text-right text-soft tnum">{fmt(ch.test[f])}</td></tr>;
                })}
              </tbody>
            </table>
            <p className="mt-3 text-[13px] text-mute">The scorecard {ch.gate_passed ? "passes" : "fails"} the same gate. Its points table is published in reports/scorecard_points.csv.</p>
          </div>
        )}
        <div>
          {d.no_protected_at_decision && (
            <>
              <h2 className="text-[15px] font-medium">Fair without gender or age at decision time</h2>
              <p className="mt-1 text-[14px] text-soft">Fairlearn's exponentiated gradient, trained with the fairness constraint and the 5:1 cost, then deciding from the application alone.</p>
              <dl className="mt-5 divide-y divide-line border-y border-line text-[14px]">
                <div className="flex justify-between py-2.5"><dt>Approved</dt><dd className="tnum">{pct(d.no_protected_at_decision.approval_rate)}</dd></div>
                <div className="flex justify-between py-2.5"><dt>Gender gap</dt><dd className="tnum">{(d.no_protected_at_decision.eo_gap_gender * 100).toFixed(1)} pts</dd></div>
                <div className="flex justify-between py-2.5"><dt>Age gap</dt><dd className="tnum">{(d.no_protected_at_decision.eo_gap_age * 100).toFixed(1)} pts</dd></div>
                <div className="flex justify-between py-2.5"><dt>Cost per applicant (5:1)</dt><dd className="tnum">{d.no_protected_at_decision.expected_cost.toFixed(3)}</dd></div>
              </dl>
            </>
          )}
          {d.group_calibration && d.group_calibration.length > 0 && (
            <>
              <h2 className="mt-10 text-[15px] font-medium">Calibration within each group</h2>
              <p className="mt-1 text-[14px] text-soft">Mean predicted risk against the observed default rate, test split.</p>
              <ul className="mt-4 divide-y divide-line border-y border-line text-[14px]">
                {d.group_calibration.map((g) => (
                  <li key={g.group} className="flex justify-between gap-4 py-2.5">
                    <span>{groupLabel(g.group)}</span>
                    <span className={`tnum ${g.outside_ci ? "text-mustard" : "text-soft"}`}>{pct(g.mean_pd)} predicted, {pct(g.observed)} observed</span>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-[13px] text-mute">Amber: the prediction is outside the 95% interval of the observed rate. A per-group correction is analysed in the report.</p>
            </>
          )}
        </div>
      </section>

      {d.stress && d.stress.length > 0 && (
        <section className="mt-20">
          <h2 className="text-[15px] font-medium">Stress test and regulatory capital</h2>
          <p className="mt-1 text-[14px] text-soft">Basel IRB capital for other retail exposures on each policy's approved loans, and the larger fairness gap when the same rule meets recession-level risk.</p>
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-[14px]">
              <thead className="text-[12px] text-mute"><tr><th className="pb-2 font-normal">Scenario</th>{policies.map((n) => <th key={n} className="pb-2 text-right font-normal">{n}</th>)}</tr></thead>
              <tbody>
                {d.stress.map((s) => (
                  <tr key={s.scenario as string} className="border-t border-line">
                    <td className="py-2.5">{s.scenario as string}<span className="block text-[12px] text-mute">mean PD {pct(s.mean_pd as number)}</span></td>
                    {policies.map((n) => { const c = cell(s, n); return (
                      <td key={n} className="py-2.5 text-right tnum">
                        {c ? <>{(c.capital / 1e9).toFixed(2)} bn<span className={`block text-[12px] ${c.gap > 0.05 ? "text-mustard" : "text-mute"}`}>gap {(c.gap * 100).toFixed(1)} pts</span></> : ""}
                      </td>
                    ); })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {d.reject_inference?.models && (
        <section className="mt-16 max-w-3xl">
          <h2 className="text-[15px] font-medium">Reject inference</h2>
          <p className="mt-1 text-[14px] leading-relaxed text-soft">
            Outcomes exist only for granted loans. Refused applications differ clearly from approved ones (a classifier tells them apart with an area
            under the curve of {d.reject_inference.accept_vs_reject_auc?.toFixed(2)}), yet neither parcelling nor augmentation changes the model reliably:
          </p>
          <ul className="mt-4 divide-y divide-line border-y border-line text-[14px]">
            {d.reject_inference.models.map((m) => (
              <li key={m.model} className="flex justify-between gap-4 py-2.5"><span>{m.model}</span><span className="text-soft tnum">AUC {m.test_auc.toFixed(3)} · rejects scored {pct(m.rejects_mean_pd)}</span></li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}
