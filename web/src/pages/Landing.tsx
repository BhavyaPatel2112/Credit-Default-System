import { useEffect, useMemo, useRef, useState } from "react";
import { motion, useScroll, useTransform, useReducedMotion } from "motion/react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import NumberFlow from "@number-flow/react";
import { toast } from "sonner";
import { api, EXAMPLE, type Decision, type Request } from "../lib/api";
import { groupLabel, money, pct, timeAgo } from "../lib/format";
import { plain } from "../lib/reasons";
import { go } from "../lib/router";
import { Headline } from "../components/Headline";
import { Terrain } from "../components/Terrain";
import { RiskScale } from "../components/RiskScale";
import { Button, LinkButton } from "../components/Button";

export function Landing() {
  return (
    <>
      <Hero />
      <Stage />
      <Activity />
      <Flow />
      <Results />
    </>
  );
}

function Hero() {
  const [id, setId] = useState("");
  const list = useQuery({ queryKey: ["applicants"], queryFn: api.applicants });
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const n = Number(id.trim());
    if (!id.trim()) return go({ page: "assess" });
    if (!Number.isInteger(n)) return toast.error("An applicant ID is a whole number, for example 395231.");
    if (list.data?.available && !list.data.applicants.some((a) => a.id === n))
      return toast.error(`Applicant ${n} is not in the demo set. Pick one from the list instead.`);
    go({ page: "assess", id: n });
  };
  const sample = list.data?.applicants?.[0]?.id;

  return (
    <section className="mx-auto max-w-[1240px] px-4 pt-20 text-center sm:px-6 sm:pt-28">
      <Headline quiet="Credit decisions" loud="you can explain." reveal />
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9, duration: 0.8 }}
        className="mx-auto mt-7 max-w-[34rem] text-pretty text-[17px] leading-relaxed text-soft sm:text-lg"
      >
        Score a loan application, see the reasons in plain language, and check that the policy treats every group of
        applicants fairly before it goes live.
      </motion.p>
      <motion.form
        onSubmit={submit}
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.05, duration: 0.7 }}
        className="mx-auto mt-10 flex max-w-xl flex-col items-stretch gap-3 sm:flex-row sm:items-center"
      >
        <div className="flex flex-1 items-center rounded-full bg-white p-1.5 pl-5 shadow-[0_1px_0_rgba(0,0,0,0.03),0_10px_30px_-18px_rgba(25,25,24,0.35)] ring-1 ring-line/70 focus-within:ring-ink/30">
          <label htmlFor="hero-id" className="sr-only">Applicant ID</label>
          <input
            id="hero-id" inputMode="numeric" value={id} onChange={(e) => setId(e.target.value)}
            placeholder={sample ? `Applicant ID, for example ${sample}` : "Applicant ID"}
            className="min-w-0 flex-1 bg-transparent text-[15px] outline-none placeholder:text-mute"
          />
          <Button type="submit" className="shrink-0">Assess</Button>
        </div>
        <LinkButton href="#/assess/new" variant="secondary" size="lg" className="h-[52px]">New application</LinkButton>
      </motion.form>
    </section>
  );
}

// The product itself on a dark stage over the risk landscape. It scores real
// applicants live; the viewer steps through them.
function Stage() {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "start 30%"] });
  const rotate = useTransform(scrollYProgress, [0, 1], [reduce ? 0 : 10, 0]);
  const y = useTransform(scrollYProgress, [0, 1], [reduce ? 0 : 40, 0]);

  const list = useQuery({ queryKey: ["applicants"], queryFn: api.applicants });
  const picks = useMemo(() => {
    const a = list.data?.applicants ?? [];
    const refer = a.filter((x) => x.decision === "refer" && x.pd < 0.45);
    const appr = a.filter((x) => x.decision === "approve" && x.pd > 0.06);
    return [refer[0], appr[0], refer[3], appr[5], refer[8]].filter(Boolean).map((x) => x!.id);
  }, [list.data]);
  const [i, setI] = useState(0);
  const id = picks.length ? picks[i % picks.length] : undefined;

  const req = useQuery({
    queryKey: ["applicant", id], enabled: id !== undefined || list.isSuccess,
    queryFn: async () => (id !== undefined ? (await api.applicant(id)).request : EXAMPLE),
  });
  const qc = useQueryClient();
  const dec = useQuery({
    queryKey: ["score", req.data], enabled: !!req.data,
    queryFn: async () => { const d = await api.score(req.data!); qc.invalidateQueries({ queryKey: ["decisions"] }); return d; },
  });

  return (
    <section ref={ref} className="mx-auto mt-20 max-w-[1240px] px-3 sm:mt-24 sm:px-6">
      <div className="grain relative overflow-hidden rounded-[28px] bg-stage px-4 pb-16 pt-12 sm:px-10 sm:pb-20 sm:pt-20" style={{ perspective: 1400 }}>
        <div className="absolute inset-0"><Terrain /></div>
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-stage/0 via-stage/10 to-stage/70" />
        <motion.div style={{ rotateX: rotate, y, transformOrigin: "50% 0%" }} className="relative mx-auto max-w-[880px]">
          <LiveCard request={req.data} decision={dec.data} loading={req.isLoading || dec.isLoading} error={dec.error || req.error} />
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3 text-[13px] text-stone/60">
            <span>{id ? `Real applicant ${id} from the held-out test set, scored live` : "Example application, scored live"}</span>
            {picks.length > 1 && (
              <div className="flex gap-2">
                <Button size="sm" variant="light" className="!bg-white/10 !text-stone hover:!bg-white/20" onClick={() => setI((v) => (v + picks.length - 1) % picks.length)}>Previous</Button>
                <Button size="sm" variant="light" onClick={() => setI((v) => v + 1)}>Next applicant</Button>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function LiveCard({ request, decision, loading, error }: { request?: Request; decision?: Decision; loading: boolean; error: unknown }) {
  if (error) return (
    <div className="rounded-[22px] bg-stone p-8 text-ink">
      <p className="display text-3xl">The scoring service is not running.</p>
      <p className="mt-3 text-soft">Start it from the cap501 folder with <code className="rounded bg-stone-2 px-1.5 py-0.5 text-[13px]">python -m uvicorn api.app:app --port 8000</code> and reload this page.</p>
    </div>
  );
  const refer = decision?.decision === "refer";
  const group = request ? `${request.gender}|${request.age_years < 35 ? "Under 35" : "35 and over"}` : "";
  return (
    <div className="overflow-hidden rounded-[22px] bg-stone text-ink shadow-[0_40px_80px_-30px_rgba(0,0,0,0.6)]">
      <div className="flex items-center justify-between border-b border-line px-5 py-3 text-[13px] text-soft sm:px-7">
        <span>Assess</span>
        <span className="tnum">{request ? `${groupLabel(group)}, loan ${money(Number(request.application.AMT_CREDIT))}` : "Loading"}</span>
      </div>
      <div className={`grid gap-8 p-5 transition-opacity sm:p-7 md:grid-cols-[1.1fr_1fr] ${loading ? "opacity-50" : ""}`}>
        <div>
          <p className="display text-[2.4rem] leading-none sm:text-[2.9rem]">{decision ? (refer ? "Refer." : "Approve.") : " "}</p>
          <p className="mt-3 text-soft">
            Probability of default{" "}
            <span className="text-ink tnum">
              <NumberFlow value={(decision?.probability_of_default ?? 0) * 100} format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} suffix="%" />
            </span>
          </p>
          <div className="mt-6">{decision && <RiskScale pd={decision.probability_of_default} cutoff={decision.cutoff_applied} />}</div>
        </div>
        <div>
          <p className="text-[13px] text-mute">{refer ? "Main reasons, as the applicant reads them" : "What counted in favour"}</p>
          <ul className="mt-3 space-y-3">
            {(refer ? decision!.adverse_action_reasons.slice(0, 3) : decision?.top_factors.filter((f) => f.effect === "lowers risk").slice(0, 3) ?? []).map((f) => (
              <motion.li key={f.reason} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3 text-[14px] leading-snug">
                <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${refer ? "bg-mustard" : "bg-moss"}`} />
                {plain(f.reason, f.effect)}
              </motion.li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function Activity() {
  const q = useQuery({ queryKey: ["decisions", 5], queryFn: () => api.decisions(5), refetchInterval: 8000 });
  const [, tick] = useState(0);
  useEffect(() => { const t = setInterval(() => tick((v) => v + 1), 15000); return () => clearInterval(t); }, []);
  const rows = q.data?.decisions ?? [];
  return (
    <section className="mx-auto mt-28 max-w-[1240px] px-4 sm:px-6">
      <div className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr]">
        <div>
          <Headline as="h2" size="md" quiet="Every decision" loud="is written down." />
          <p className="mt-5 max-w-sm leading-relaxed text-soft">
            Each score is logged with the exact model, calibrator and policy that produced it, so any decision can be
            traced later. The application itself is kept only as a fingerprint.
          </p>
        </div>
        <div className="rounded-[22px] bg-white/70 p-2 ring-1 ring-line/60">
          {rows.length === 0 ? (
            <p className="p-6 text-soft">No decisions yet. Assess an applicant and it appears here.</p>
          ) : (
            <table className="w-full text-left text-[14px]">
              <caption className="sr-only">Most recent decisions</caption>
              <thead className="text-[12px] text-mute">
                <tr><th className="hidden px-4 py-2 font-normal sm:table-cell">When</th><th className="px-4 py-2 font-normal">Group</th><th className="px-4 py-2 text-right font-normal">Probability</th><th className="px-4 py-2 text-right font-normal">Decision</th></tr>
              </thead>
              <tbody>
                {rows.map((d) => (
                  <motion.tr key={d.decision_id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="border-t border-line/70">
                    <td className="hidden px-4 py-3 text-soft sm:table-cell">{timeAgo(d.time_utc)}</td>
                    <td className="px-4 py-3">{groupLabel(d.policy_group)}</td>
                    <td className="px-4 py-3 text-right tnum">{pct(d.probability_of_default)}</td>
                    <td className="px-4 py-3 text-right">
                      <span className={`inline-flex items-center gap-1.5 ${d.decision === "approve" ? "text-moss" : "text-mustard"}`}>
                        <span className="h-1.5 w-1.5 rounded-full bg-current" />{d.decision === "approve" ? "Approved" : "Referred"}
                      </span>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </section>
  );
}

const STEPS = [
  { title: "Score", body: "Gradient-boosted trees over 449 features from the application and six credit-history tables, calibrated so a 20% score means about one in five default.", link: "#/assess", cta: "Assess an applicant" },
  { title: "Explain", body: "Every referral lists up to four reasons in plain language, and searches for the smallest change to the loan that would get it approved.", link: "#/assess", cta: "See a referral" },
  { title: "Govern", body: "Cut-offs are set so repayers are approved at the same rate in every gender and age group, and a release gate blocks any change that breaks that.", link: "#/fairness", cta: "Explore the trade-off" },
];

function Flow() {
  return (
    <section className="mx-auto mt-32 max-w-[1240px] px-4 sm:px-6">
      <Headline as="h2" size="lg" quiet="From application" loud="to an answer you can defend." className="max-w-3xl" />
      <div className="mt-14 grid gap-px overflow-hidden rounded-[22px] bg-line md:grid-cols-3">
        {STEPS.map((s) => (
          <div key={s.title} className="flex flex-col bg-stone p-7">
            <h3 className="text-xl font-medium tracking-[-0.01em]">{s.title}</h3>
            <p className="mt-3 flex-1 leading-relaxed text-soft">{s.body}</p>
            <a href={s.link} className="mt-8 self-start text-[15px] text-ink underline decoration-line underline-offset-[6px] transition-colors hover:decoration-ink">{s.cta}</a>
          </div>
        ))}
      </div>
    </section>
  );
}

function Results() {
  const m = useQuery({ queryKey: ["model-card"], queryFn: api.modelCard });
  const t = m.data?.test;
  const rows = t ? [
    { k: "Ranks defaulters above repayers", v: t.roc_auc.toFixed(3), req: "0.75 or higher", note: "Area under the curve" },
    { k: "Predicted risk matches observed default", v: t.ece.toFixed(3), req: "0.05 or lower", note: "Expected calibration error" },
    { k: "Repayer approval gap between genders", v: pts(t.eo_gap_gender), req: "5 points or less", note: "Equal opportunity" },
    { k: "Repayer approval gap, under 35 and 35 and over", v: pts(t.eo_gap_age), req: "5 points or less", note: "Equal opportunity" },
    { k: "Time to score and explain, 95th percentile", v: `${m.data!.latency.p95_ms.toFixed(1)} ms`, req: "50 ms or less", note: "Measured on 1,000 applicants" },
  ] : [];
  return (
    <section className="mx-auto mt-32 max-w-[1240px] px-4 sm:px-6">
      <div className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr]">
        <div>
          <Headline as="h2" size="md" quiet="Tested on 46,127" loud="applicants it never saw." />
          <p className="mt-5 max-w-sm leading-relaxed text-soft">
            Each requirement was set before the model was built. The release gate checks all of them again on every change.
          </p>
          <LinkButton href="#/model" variant="secondary" className="mt-7">Open the model card</LinkButton>
        </div>
        <table className="w-full text-left text-[15px]">
          <caption className="sr-only">Results on the held-out test set</caption>
          <thead className="text-[12px] text-mute"><tr><th className="py-2 font-normal">Requirement</th><th className="py-2 text-right font-normal">Result</th><th className="hidden py-2 text-right font-normal sm:table-cell">Required</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.k} className="border-t border-line">
                <td className="py-4 pr-4"><span className="block">{r.k}</span><span className="text-[13px] text-mute">{r.note}</span></td>
                <td className="py-4 text-right text-lg tnum">{r.v}</td>
                <td className="hidden py-4 text-right text-soft sm:table-cell">{r.req}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const pts = (v: number) => `${(v * 100).toFixed(1)} points`;
