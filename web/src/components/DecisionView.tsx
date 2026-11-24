import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import NumberFlow from "@number-flow/react";
import { useMutation } from "@tanstack/react-query";
import * as Switch from "@radix-ui/react-switch";
import type { Decision, Request } from "../lib/api";
import { api } from "../lib/api";
import { groupLabel, money, pct } from "../lib/format";
import { NOT_ACTIONABLE, plain } from "../lib/reasons";
import { RiskScale } from "./RiskScale";
import { Factors } from "./Factors";
import { Button } from "./Button";

const keyOf = (r: Request) => `${r.gender}|${r.age_years < 35 ? "Under 35" : "35 and over"}`;

export function DecisionView({
  decision, request, actual, onTry, busy,
}: {
  decision: Decision; request: Request; actual?: "repaid" | "defaulted"; onTry: (r: Request) => void; busy?: boolean;
}) {
  const refer = decision.decision === "refer";
  const group = keyOf(request);

  return (
    <div className="space-y-12">
      {/* the verdict */}
      <section aria-live="polite">
        <AnimatePresence mode="wait">
          <motion.h2
            key={decision.decision}
            initial={{ opacity: 0, filter: "blur(8px)" }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0, filter: "blur(8px)" }}
            transition={{ duration: 0.45 }}
            className="display text-[2.8rem] sm:text-[3.6rem]"
          >
            {refer ? "Refer to a loan officer." : "Approve."}
          </motion.h2>
        </AnimatePresence>
        <p className="mt-3 text-lg text-soft">
          Probability of default{" "}
          <span className="text-ink tnum">
            <NumberFlow value={decision.probability_of_default * 100} format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} suffix="%" />
          </span>
          , band <span className="text-ink">{decision.band.charAt(0)}</span>.{" "}
          {refer
            ? `Above the ${pct(decision.cutoff_applied)} cut-off, so a person decides. The service never declines on its own.`
            : `Below the ${pct(decision.cutoff_applied)} cut-off.`}
        </p>
        <div className="mt-8">
          <RiskScale pd={decision.probability_of_default} cutoff={decision.cutoff_applied} group={group} />
        </div>
        {actual && <Outcome actual={actual} />}
      </section>

      {/* referral: what the applicant is told, and what would change the answer */}
      {refer && (
        <div className="grid items-start gap-6 lg:grid-cols-2">
          <Notice decision={decision} request={request} />
          <RecoursePanel request={request} onTry={onTry} busy={busy} />
        </div>
      )}

      <section>
        <div className="mb-2 flex items-baseline justify-between gap-4">
          <h3 className="text-[15px] font-medium">What drove this score</h3>
          <span className="text-[13px] text-mute">For the loan officer</span>
        </div>
        <Factors factors={decision.top_factors} />
      </section>

      <Details decision={decision} />
    </div>
  );
}

function Outcome({ actual }: { actual: "repaid" | "defaulted" }) {
  const [show, setShow] = useState(false);
  return (
    <div className="mt-6 flex flex-wrap items-center gap-3 rounded-2xl bg-stone-2/70 px-4 py-3 text-sm">
      <Switch.Root checked={show} onCheckedChange={setShow} id="outcome"
        className="relative h-5 w-9 shrink-0 rounded-full bg-stone-3 transition-colors data-[state=checked]:bg-ink">
        <Switch.Thumb className="block h-4 w-4 translate-x-0.5 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[18px]" />
      </Switch.Root>
      <label htmlFor="outcome" className="cursor-pointer text-soft">Show what actually happened</label>
      <AnimatePresence>
        {show && (
          <motion.span initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                       className={actual === "repaid" ? "text-moss" : "text-mustard"}>
            This applicant {actual === "repaid" ? "repaid the loan" : "defaulted"}.
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  );
}

// The adverse-action notice, as the applicant would receive it.
function Notice({ decision, request }: { decision: Decision; request: Request }) {
  return (
    <section className="rounded-[22px] bg-white p-6 shadow-[0_1px_0_rgba(25,25,24,0.04),0_12px_32px_-16px_rgba(25,25,24,0.18)] ring-1 ring-line/60 sm:p-7">
      <p className="text-[13px] text-mute">What the applicant is told</p>
      <p className="display mt-4 text-[1.7rem] leading-tight">Your application needs a second look.</p>
      <p className="mt-3 text-[15px] leading-relaxed text-soft">
        A loan officer will review your application for {money(Number(request.application.AMT_CREDIT))} before a decision is
        made. The main reasons it was not approved automatically:
      </p>
      <ol className="mt-4 space-y-3">
        {decision.adverse_action_reasons.map((r, i) => (
          <li key={r.reason} className="flex gap-3 text-[15px] leading-snug">
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-stone-2 text-[11px] text-soft tnum">{i + 1}</span>
            <span>
              {plain(r.reason, "raises risk")}
              {NOT_ACTIONABLE.has(r.reason.replace(/: none on file$/, "")) && (
                <span className="mt-1 block text-[12px] text-mustard">Not something the applicant can change. The officer should weigh this.</span>
              )}
            </span>
          </li>
        ))}
      </ol>
      <p className="mt-5 text-[13px] leading-relaxed text-mute">
        Reasons are listed in order of influence, at most four. Your gender and age were not used to score you.
      </p>
    </section>
  );
}

function RecoursePanel({ request, onTry, busy }: { request: Request; onTry: (r: Request) => void; busy?: boolean }) {
  const m = useMutation({ mutationFn: () => api.recourse(request) });
  const loan = Number(request.application.AMT_CREDIT), inst = Number(request.application.AMT_ANNUITY);
  return (
    <section className="rounded-[22px] bg-stone-2/70 p-6 sm:p-7">
      <p className="text-[13px] text-mute">Path to approval</p>
      <p className="display mt-4 text-[1.7rem] leading-tight">What would change the answer?</p>
      <p className="mt-3 text-[15px] leading-relaxed text-soft">
        Searches smaller loans and terms up to 43% longer, keeping everything else about the applicant fixed.
        Now: {money(loan)} at {money(inst)} per instalment.
      </p>
      {!m.data && (
        <Button className="mt-5" onClick={() => m.mutate()} disabled={m.isPending}>
          {m.isPending ? "Searching" : "Find options"}
        </Button>
      )}
      {m.isError && <p className="mt-4 text-sm text-mustard">{(m.error as Error).message}</p>}
      {m.data && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="mt-5 space-y-2.5">
          {m.data.options.length === 0 && <p className="text-[15px] text-soft">{m.data.note}</p>}
          {m.data.options.map((o, i) => (
            <div key={i} className="flex items-center justify-between gap-4 rounded-2xl bg-white/80 px-4 py-3 ring-1 ring-line/60">
              <div className="text-[14px] leading-snug">
                <p className="text-ink">
                  {o.loan_cut_pct > 0 ? `Borrow ${money(o.loan)} (${o.loan_cut_pct.toFixed(0)}% less)` : `Keep ${money(o.loan)}`}
                  {o.term_change_pct > 0.5 ? `, ${o.term_change_pct.toFixed(0)}% longer term` : ", same term"}
                </p>
                <p className="text-soft tnum">Instalment {money(o.annuity)}. Probability of default {pct(o.pd)}</p>
              </div>
              <Button size="sm" variant="secondary" disabled={busy}
                onClick={() => onTry({ ...request, application: { ...request.application, AMT_CREDIT: o.loan, AMT_ANNUITY: o.annuity } })}>
                Apply
              </Button>
            </div>
          ))}
        </motion.div>
      )}
    </section>
  );
}

function Details({ decision }: { decision: Decision }) {
  const q = decision.data_quality;
  return (
    <details className="group rounded-2xl border border-line px-5 py-4 text-sm">
      <summary className="cursor-pointer list-none text-soft transition-colors hover:text-ink">
        <span className="inline-block transition-transform group-open:rotate-90">›</span>&nbsp; Record of this decision
      </summary>
      <dl className="mt-4 grid gap-x-8 gap-y-2.5 text-[13px] sm:grid-cols-2">
        <Row k="Decision id" v={decision.decision_id.slice(0, 12)} />
        <Row k="Model" v={decision.versions.model_sha256} />
        <Row k="Calibrator" v={decision.versions.calibrator_sha256} />
        <Row k="Policy" v={decision.versions.policy_sha256} />
        <Row k="Credit history supplied" v={q.history_tables_supplied.length ? q.history_tables_supplied.join(", ") : "None (thin file)"} />
        <Row k="Fields sent but not used" v={q.ignored_fields.length ? String(q.ignored_fields.length) : "None"} />
        {decision.expected_profit != null && <Row k="Expected profit of the loan" v={money(decision.expected_profit)} />}
        <Row k="Scored in" v={`${decision.latency_ms.toFixed(1)} ms`} />
      </dl>
      <p className="mt-4 text-[12px] text-mute">Logged to the decision log with these fingerprints. The application itself is stored only as a hash.</p>
    </details>
  );
}

const Row = ({ k, v }: { k: string; v: string }) => (
  <div className="flex justify-between gap-4 border-b border-line/60 pb-2">
    <dt className="text-mute">{k}</dt><dd className="text-right text-ink tnum">{v}</dd>
  </div>
);

export { groupLabel };
