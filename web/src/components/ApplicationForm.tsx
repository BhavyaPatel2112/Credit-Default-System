import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import * as Slider from "@radix-ui/react-slider";
import type { Application, Request } from "../lib/api";
import { api } from "../lib/api";
import { Button } from "./Button";

// The application form, grouped the way an officer reads a file. Days are
// entered in years and converted; categories come from GET /schema so they
// always match what the model was trained on.
const YEAR = 365.25;

// Server problems name model fields; show them with the labels on this form.
const LABELS: Record<string, string> = {
  AMT_INCOME_TOTAL: "Annual income", AMT_CREDIT: "Loan amount", AMT_ANNUITY: "Instalment", AMT_GOODS_PRICE: "Price of goods",
  NAME_CONTRACT_TYPE: "Loan type", NAME_INCOME_TYPE: "Income type", DAYS_EMPLOYED: "Years in current job",
  ORGANIZATION_TYPE: "Employer type", OCCUPATION_TYPE: "Occupation", NAME_EDUCATION_TYPE: "Education",
  CNT_CHILDREN: "Children", CNT_FAM_MEMBERS: "Family members", NAME_HOUSING_TYPE: "Housing",
  FLAG_OWN_REALTY: "Owns property", FLAG_OWN_CAR: "Owns a car", EXT_SOURCE_1: "Agency 1 score",
  EXT_SOURCE_2: "Agency 2 score", EXT_SOURCE_3: "Agency 3 score", DAYS_REGISTRATION: "Years at registered address",
  REGION_RATING_CLIENT: "Region rating",
  REGION_RATING_CLIENT_W_CITY: "Region rating, including city", REGION_POPULATION_RELATIVE: "Region population",
};
export function friendly(problem: string) {
  const m = problem.match(/^([A-Z0-9_]+)(=[^:]*)?:\s*(.*)$/);
  if (!m || !LABELS[m[1]]) return problem;
  const label = LABELS[m[1]];
  let msg = m[3];
  if (/^required/.test(msg)) return `${label} is required`;
  if (/^days are counted back/.test(msg)) return `${label} cannot be in the future`;
  if (/^(must|cannot)/.test(msg)) return `${label} ${msg}`;
  msg = msg.charAt(0).toUpperCase() + msg.slice(1);
  return msg.toLowerCase().startsWith(label.toLowerCase().split(" ")[0]) ? msg : `${label}: ${msg}`;
}
const pretty = (v: string) => v.replace(/_/g, " ");

export function ApplicationForm({ initial, onSubmit, busy, problems }: {
  initial: Request; onSubmit: (r: Request) => void; busy?: boolean; problems?: string[];
}) {
  const [r, setR] = useState<Request>(initial);
  const alertRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (problems?.length) alertRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }); }, [problems]);
  const schema = useQuery({ queryKey: ["schema"], queryFn: api.schema, staleTime: Infinity });
  const a = r.application;
  const set = (k: string, v: number | string | null) => setR((p) => ({ ...p, application: { ...p.application, [k]: v } }));
  const cats = { ...schema.data?.required_categories, ...schema.data?.optional_categories } as Record<string, string[]>;

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); onSubmit(r); }}
      className="space-y-10"
      noValidate
    >
      {problems && problems.length > 0 && (
        <div ref={alertRef} role="alert" className="scroll-mt-28 rounded-2xl bg-mustard-2/70 px-5 py-4 text-[14px]">
          <p className="font-medium text-ink">The service could not score this application. Fix these and assess again:</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-soft">{problems.map((p) => <li key={p}>{friendly(p)}</li>)}</ul>
        </div>
      )}

      <Section title="Loan" hint="What the applicant is asking for.">
        <Select label="Type" k="NAME_CONTRACT_TYPE" a={a} set={set} options={cats.NAME_CONTRACT_TYPE} />
        <Num label="Loan amount" k="AMT_CREDIT" a={a} set={set} step={1000} />
        <Num label="Instalment" k="AMT_ANNUITY" a={a} set={set} step={100} />
        <Num label="Price of goods" k="AMT_GOODS_PRICE" a={a} set={set} step={1000} optional hint="Leave empty for a revolving loan" />
      </Section>

      <Section title="Applicant" hint="Income, work and household.">
        <Num label="Annual income" k="AMT_INCOME_TOTAL" a={a} set={set} step={1000} />
        <Select label="Income type" k="NAME_INCOME_TYPE" a={a} set={set} options={cats.NAME_INCOME_TYPE} />
        <Years label="Years in current job" k="DAYS_EMPLOYED" a={a} set={set} nullable nullLabel="Not employed" />
        <Select label="Employer type" k="ORGANIZATION_TYPE" a={a} set={set} options={cats.ORGANIZATION_TYPE} />
        <Select label="Occupation" k="OCCUPATION_TYPE" a={a} set={set} options={cats.OCCUPATION_TYPE} optional />
        <Select label="Education" k="NAME_EDUCATION_TYPE" a={a} set={set} options={cats.NAME_EDUCATION_TYPE} />
        <Num label="Children" k="CNT_CHILDREN" a={a} set={set} step={1} />
        <Num label="Family members, including the applicant" k="CNT_FAM_MEMBERS" a={a} set={set} step={1} />
        <Select label="Housing" k="NAME_HOUSING_TYPE" a={a} set={set} options={cats.NAME_HOUSING_TYPE} />
        <Select label="Owns property" k="FLAG_OWN_REALTY" a={a} set={set} options={["Y", "N"]} labels={{ Y: "Yes", N: "No" }} />
        <Select label="Owns a car" k="FLAG_OWN_CAR" a={a} set={set} options={["Y", "N"]} labels={{ Y: "Yes", N: "No" }} />
      </Section>

      <Section title="External credit scores" hint="From three credit agencies, 0 is worst and 1 is best. Switch off any the applicant does not have.">
        {(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] as const).map((k, i) => (
          <Score key={k} label={`Agency ${i + 1}`} k={k} a={a} set={set} />
        ))}
      </Section>

      <Section title="Policy group" hint="Used only to choose the cut-off, never as a model input.">
        <label className="block">
          <span className="text-[13px] text-soft">Gender</span>
          <select value={r.gender} onChange={(e) => setR({ ...r, gender: e.target.value as Request["gender"] })} className={field}>
            <option value="Female">Female</option><option value="Male">Male</option>
          </select>
        </label>
        <label className="block">
          <span className="text-[13px] text-soft">Age</span>
          <input type="number" min={18} max={100} value={r.age_years} onChange={(e) => setR({ ...r, age_years: Number(e.target.value) })} className={field} />
        </label>
      </Section>

      <details className="group rounded-2xl border border-line px-5 py-4">
        <summary className="cursor-pointer list-none text-[15px] text-soft hover:text-ink">
          <span className="inline-block transition-transform group-open:rotate-90">›</span>&nbsp; Address, employer contact and region
          <span className="ml-2 text-[13px] text-mute">prefilled, change if known</span>
        </summary>
        <div className="mt-6 grid gap-x-6 gap-y-5 sm:grid-cols-2">
          <Years label="Years at registered address" k="DAYS_REGISTRATION" a={a} set={set} />
          <Select label="Region rating" k="REGION_RATING_CLIENT" a={a} set={set} options={[1, 2, 3]} numeric labels={{ 1: "1, best", 2: "2", 3: "3, worst" }} />
          <Select label="Region rating, including city" k="REGION_RATING_CLIENT_W_CITY" a={a} set={set} options={[1, 2, 3]} numeric labels={{ 1: "1, best", 2: "2", 3: "3, worst" }} />
          <Num label="Region population, relative" k="REGION_POPULATION_RELATIVE" a={a} set={set} step={0.001} />
          <div className="grid grid-cols-2 gap-2 sm:col-span-2">
            {[
              ["FLAG_EMP_PHONE", "Gave an employer phone"],
              ["REG_CITY_NOT_LIVE_CITY", "Lives outside the registered city"], ["REG_CITY_NOT_WORK_CITY", "Works outside the registered city"],
              ["LIVE_CITY_NOT_WORK_CITY", "Works outside the city lived in"], ["REG_REGION_NOT_LIVE_REGION", "Lives outside the registered region"],
              ["REG_REGION_NOT_WORK_REGION", "Works outside the registered region"], ["LIVE_REGION_NOT_WORK_REGION", "Works outside the region lived in"],
            ].map(([k, label]) => (
              <label key={k} className="flex cursor-pointer items-center gap-2.5 rounded-xl px-2 py-1.5 text-[14px] hover:bg-stone-2/60">
                <input type="checkbox" checked={a[k] === 1} onChange={(e) => set(k, e.target.checked ? 1 : 0)} className="h-4 w-4 accent-[var(--color-ink)]" />
                {label}
              </label>
            ))}
          </div>
        </div>
      </details>

      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" size="lg" disabled={busy}>{busy ? "Assessing" : "Assess application"}</Button>
        <Button type="button" variant="ghost" onClick={() => setR(initial)}>Reset</Button>
        {Object.keys(r.history).length > 0 && <span className="text-[13px] text-mute">Credit history from the applicant's file is kept as it is.</span>}
      </div>
    </form>
  );
}

const field = "mt-1.5 h-11 w-full rounded-xl bg-white px-3.5 text-[15px] text-ink ring-1 ring-line outline-none transition focus:ring-2 focus:ring-ink/40 tnum";

function Section({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <fieldset className="grid gap-x-6 gap-y-5 border-t border-line pt-6 sm:grid-cols-2">
      <legend className="sr-only">{title}</legend>
      <div className="sm:col-span-2">
        <p className="text-[17px] font-medium">{title}</p>
        <p className="mt-1 text-[14px] text-soft">{hint}</p>
      </div>
      {children}
    </fieldset>
  );
}

type FP = { label: string; k: string; a: Application; set: (k: string, v: number | string | null) => void };

function Num({ label, k, a, set, step, optional, hint }: FP & { step: number; optional?: boolean; hint?: string }) {
  const v = a[k];
  return (
    <label className="block">
      <span className="text-[13px] text-soft">{label}{optional && <span className="text-mute">, optional</span>}</span>
      <input type="number" step={step} min={0} value={v == null ? "" : String(v)} placeholder={hint}
        onChange={(e) => set(k, e.target.value === "" ? null : Number(e.target.value))} className={field} />
    </label>
  );
}

function Years({ label, k, a, set, nullable, nullLabel }: FP & { nullable?: boolean; nullLabel?: string }) {
  const v = a[k];
  const none = v === null;
  return (
    <label className="block">
      <span className="flex items-center justify-between text-[13px] text-soft">
        {label}
        {nullable && (
          <span className="flex items-center gap-1.5">
            <input type="checkbox" checked={none} onChange={(e) => set(k, e.target.checked ? null : -365)} className="h-3.5 w-3.5 accent-[var(--color-ink)]" />
            {nullLabel}
          </span>
        )}
      </span>
      <input type="number" step={0.1} min={0} disabled={none}
        value={none || v === undefined ? "" : (-Number(v) / YEAR).toFixed(1)}
        onChange={(e) => set(k, e.target.value === "" ? 0 : -Math.round(Number(e.target.value) * YEAR))}
        className={`${field} disabled:opacity-40`} />
    </label>
  );
}

function Select({ label, k, a, set, options, labels, optional, numeric }: FP & {
  options?: (string | number)[]; labels?: Record<string | number, string>; optional?: boolean; numeric?: boolean;
}) {
  const v = a[k];
  return (
    <label className="block">
      <span className="text-[13px] text-soft">{label}{optional && <span className="text-mute">, optional</span>}</span>
      <select value={v == null ? "" : String(numeric ? v : String(v).replace(/ /g, "_"))}
        onChange={(e) => set(k, e.target.value === "" ? null : numeric ? Number(e.target.value) : pretty(e.target.value))}
        className={field}>
        {optional && <option value="">Not given</option>}
        {(options ?? []).map((o) => <option key={o} value={String(o)}>{labels?.[o] ?? pretty(String(o))}</option>)}
      </select>
    </label>
  );
}

function Score({ label, k, a, set }: FP) {
  const v = a[k];
  const on = v != null;
  return (
    <div className="sm:col-span-2 grid grid-cols-[110px_1fr_56px] items-center gap-4">
      <label className="flex items-center gap-2 text-[14px]">
        <input type="checkbox" checked={on} onChange={(e) => set(k, e.target.checked ? 0.5 : null)} className="h-4 w-4 accent-[var(--color-ink)]" />
        {label}
      </label>
      <Slider.Root className={`relative flex h-6 touch-none items-center ${on ? "" : "opacity-30"}`} disabled={!on}
        value={[on ? Number(v) : 0.5]} min={0} max={1} step={0.01} onValueChange={([x]) => set(k, x)} aria-label={`${label} score`}>
        <Slider.Track className="relative h-1.5 grow rounded-full bg-stone-3">
          <Slider.Range className="absolute h-full rounded-full bg-ink" />
        </Slider.Track>
        <Slider.Thumb className="block h-5 w-5 rounded-full bg-white shadow ring-1 ring-line focus-visible:outline-2 focus-visible:outline-moss" />
      </Slider.Root>
      <span className="text-right text-[14px] tnum">{on ? Number(v).toFixed(2) : "None"}</span>
    </div>
  );
}
