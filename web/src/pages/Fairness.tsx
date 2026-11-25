import { useDeferredValue, useState } from "react";
import { motion } from "motion/react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import * as Slider from "@radix-ui/react-slider";
import NumberFlow from "@number-flow/react";
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { api, type Fairness as FairnessData, type GroupStat } from "../lib/api";
import { groupLabel, pct } from "../lib/format";
import { Headline } from "../components/Headline";

const INK = "#191918", MOSS = "#3e5b41", MUSTARD = "#a87717", MUTE = "#86857f", LINE = "#d6d5cf";

export function Fairness() {
  const [ratio, setRatio] = useState(5);
  const r = useDeferredValue(ratio);
  const f = useQuery({ queryKey: ["fairness", r], queryFn: () => api.fairness(r), placeholderData: keepPreviousData });
  const d = f.data;

  return (
    <div className="mx-auto max-w-[1240px] px-4 pt-16 sm:px-6 sm:pt-20">
      <Headline quiet="One cut-off for everyone" loud="approves young repayers less often." size="lg" className="max-w-4xl" />
      <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-soft">
        Younger applicants default more often, so a single cut-off turns away more of the young people who would have
        repaid. The deployed policy weighs each loan's own expected loss against the interest it would earn, then gives
        each gender and age group a small offset, chosen so repayers are approved at the same rate everywhere, and never
        lowers anyone's chance of approval to get there.
      </p>

      {/* the control */}
      <section className="mt-14 rounded-[24px] bg-white/70 p-6 ring-1 ring-line/60 sm:p-8">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div className="max-w-md">
            <p className="text-[15px] font-medium">How much worse is a default than a lost customer?</p>
            <p className="mt-1 text-[14px] text-soft">
              Approving someone who defaults costs <span className="text-ink tnum">{ratio}</span> times as much as turning away someone who would have repaid.
              That sets the single cut-off at <span className="text-ink tnum">{pct(1 / (1 + ratio))}</span>. Release 1.2 assumed 5;
              the deployed policy no longer needs this number.
            </p>
          </div>
          <div className="w-full max-w-md">
            <Slider.Root className="relative flex h-8 touch-none items-center" value={[ratio]} min={1} max={15} step={0.5}
              onValueChange={([v]) => setRatio(v)} aria-label="Cost of a default relative to a lost customer">
              <Slider.Track className="relative h-1.5 grow rounded-full bg-stone-3">
                <Slider.Range className="absolute h-full rounded-full bg-ink" />
              </Slider.Track>
              <Slider.Thumb className="block h-6 w-6 rounded-full bg-white shadow-md ring-1 ring-line focus-visible:outline-2 focus-visible:outline-moss" />
            </Slider.Root>
            <div className="mt-1 flex justify-between text-[12px] text-mute tnum"><span>1</span><span>5, the 1.2 assumption</span><span>15</span></div>
          </div>
        </div>
      </section>

      {d && (
        <>
          <section className="mt-10 grid gap-10 lg:grid-cols-[1.15fr_1fr] [&>*]:min-w-0">
            <div>
              <h2 className="text-[15px] font-medium">Repayers approved, by group</h2>
              <p className="mt-1 text-[14px] text-soft">Hollow: one cut-off at this cost. Filled: the deployed policy.</p>
              <Dumbbells single={d.single.groups} policy={d.policy.groups} thresholds={d.policy.thresholds} />
            </div>
            <Compare d={d} />
          </section>

          <section className="mt-12 rounded-[24px] bg-stone-2/70 p-6 sm:p-8">
            <p className="display text-[1.9rem] leading-tight sm:text-[2.3rem]">
              Fairness here is a cost the lender chooses to carry.
            </p>
            <p className="mt-4 max-w-3xl text-[16px] leading-relaxed text-soft">
              At this cost ratio the deployed policy approves <span className="text-ink tnum">{d.added_by_policy.count.toLocaleString("en-IN")}</span> test
              applicants that one cut-off would refer
              {d.removed_by_policy > 0 && <> and refers <span className="text-ink tnum">{d.removed_by_policy.toLocaleString("en-IN")}</span> it would approve</>}.
              They go on to default at <span className="text-ink tnum">{pct(d.added_by_policy.default_rate)}</span>, close to the{" "}
              <span className="text-ink tnum">{pct(d.added_by_policy.mean_pd)}</span> the model predicted for them. The model is not wrong about them;
              the policy accepts that risk so that repayers in every group have the same chance. The deployed policy does not
              use this ratio: each loan has its own cut-off, from its size, its term and a 45% loss given default.
            </p>
          </section>

          <section className="mt-16 grid gap-12 lg:grid-cols-2 [&>*]:min-w-0">
            <GapCurve ratio={ratio} />
            <Frontier />
          </section>
        </>
      )}
      {f.isError && <p className="mt-10 text-mustard">{(f.error as Error).message}</p>}
    </div>
  );
}

function Dumbbells({ single, policy, thresholds }: { single: GroupStat[]; policy: GroupStat[]; thresholds: Record<string, number> }) {
  const lo = 0.6, hi = 1;
  const x = (v: number) => `${((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="mt-6 space-y-6">
      {single.map((s) => {
        const p = policy.find((q) => q.group === s.group)!;
        const a = s.approval_among_repayers, b = p.approval_among_repayers;
        return (
          <div key={s.group}>
            <div className="flex items-baseline justify-between text-[14px]">
              <span>{groupLabel(s.group)}</span>
              <span className="text-soft tnum">
                <NumberFlow value={a * 100} format={{ maximumFractionDigits: 1, minimumFractionDigits: 1 }} suffix="%" /> to{" "}
                <span className="text-ink"><NumberFlow value={b * 100} format={{ maximumFractionDigits: 1, minimumFractionDigits: 1 }} suffix="%" /></span>
              </span>
            </div>
            <div className="relative mt-2 h-5">
              <div className="absolute inset-x-0 top-1/2 h-px bg-line" />
              <motion.div className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-ink/30" initial={false}
                animate={{ left: x(Math.min(a, b)), width: `calc(${x(Math.max(a, b))} - ${x(Math.min(a, b))})` }} />
              <motion.div className="absolute top-1/2 -ml-2 h-4 w-4 -translate-y-1/2 rounded-full border-2 border-ink bg-stone" initial={false} animate={{ left: x(a) }}
                transition={{ type: "spring", stiffness: 160, damping: 22 }} />
              <motion.div className="absolute top-1/2 -ml-2 h-4 w-4 -translate-y-1/2 rounded-full bg-moss" initial={false} animate={{ left: x(b) }} />
            </div>
            <p className="mt-1 text-[12px] text-mute tnum">Median deployed cut-off {pct(thresholds[s.group])}. Repayers in test set {s.repayers.toLocaleString("en-IN")}.</p>
          </div>
        );
      })}
      <div className="flex justify-between text-[12px] text-mute tnum"><span>60%</span><span>80%</span><span>100%</span></div>
    </div>
  );
}

function Compare({ d }: { d: FairnessData }) {
  const rows: { k: string; a: number; b: number; fmt: "pct" | "gap" | "num"; limit?: boolean }[] = [
    { k: "Applicants approved", a: d.single.approval_rate, b: d.policy.approval_rate, fmt: "pct" },
    { k: "Approved loans that default", a: d.single.default_among_approved, b: d.policy.default_among_approved, fmt: "pct" },
    { k: "Defaulters caught", a: d.single.defaulters_declined, b: d.policy.defaulters_declined, fmt: "pct" },
    { k: "Cost per applicant", a: d.single.expected_cost, b: d.policy.expected_cost, fmt: "num" },
    { k: "Gender gap", a: d.single.gap_gender, b: d.policy.gap_gender, fmt: "gap", limit: true },
    { k: "Age gap", a: d.single.gap_age, b: d.policy.gap_age, fmt: "gap", limit: true },
  ];
  const show = (v: number, fmt: string) => fmt === "num"
    ? <NumberFlow value={v} format={{ minimumFractionDigits: 3, maximumFractionDigits: 3 }} />
    : <NumberFlow value={v * 100} format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} suffix={fmt === "gap" ? " pts" : "%"} />;
  return (
    <table className="w-full self-start text-left text-[15px]">
      <caption className="sr-only">One cut-off compared with the deployed policy</caption>
      <thead className="text-[12px] text-mute">
        <tr><th className="pb-3 font-normal" /><th className="pb-3 text-right font-normal">One cut-off</th><th className="pb-3 text-right font-normal">Deployed</th></tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const over = (v: number) => r.limit && v > d.limit;
          return (
            <tr key={r.k} className="border-t border-line">
              <td className="py-3.5 pr-3">{r.k}{r.limit && <span className="block text-[12px] text-mute">Repayer approval, limit 5 points</span>}{r.k === "Cost per applicant" && <span className="block text-[12px] text-mute">In units of one lost customer</span>}</td>
              <td className={`py-3.5 text-right tnum ${over(r.a) ? "text-mustard" : "text-soft"}`}>{show(r.a, r.fmt)}</td>
              <td className={`py-3.5 text-right tnum ${over(r.b) ? "text-mustard" : "text-ink"}`}>{show(r.b, r.fmt)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function GapCurve({ ratio }: { ratio: number }) {
  const c = useQuery({ queryKey: ["curve"], queryFn: api.curve });
  const data = (c.data?.points ?? []).map((p) => ({ ...p, age: p.gap_age * 100, gender: p.gap_gender * 100 }));
  return (
    <figure>
      <figcaption>
        <p className="text-[15px] font-medium">With one cut-off, the gaps grow as defaults get dearer</p>
        <p className="mt-1 text-[14px] text-soft">Repayer approval gap in points. The dashed line is the 5-point limit.</p>
      </figcaption>
      <div className="mt-5 h-72">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: -8 }}>
            <CartesianGrid stroke={LINE} strokeDasharray="0" vertical={false} />
            <XAxis dataKey="cost_ratio" type="number" domain={[1, 15]} ticks={[1, 3, 5, 7, 9, 11, 13, 15]} tick={{ fill: MUTE, fontSize: 12 }} axisLine={{ stroke: LINE }} tickLine={false}
              label={{ value: "Cost of a default, relative", position: "insideBottom", offset: -10, fill: MUTE, fontSize: 12 }} />
            <YAxis tick={{ fill: MUTE, fontSize: 12 }} axisLine={false} tickLine={false} unit="" width={40} />
            <Tooltip contentStyle={{ borderRadius: 12, border: `1px solid ${LINE}`, fontSize: 13 }} formatter={(v, n) => [`${Number(v).toFixed(1)} points`, n === "age" ? "Age gap" : "Gender gap"]} labelFormatter={(l) => `Cost ratio ${l}`} />
            <ReferenceLine y={5} stroke={MUSTARD} strokeDasharray="4 4" />
            <ReferenceLine x={ratio} stroke={INK} strokeOpacity={0.25} />
            <Line type="monotone" dataKey="age" stroke={INK} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="gender" stroke={MOSS} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-5 text-[13px] text-soft">
        <span className="flex items-center gap-2"><span className="h-0.5 w-5 bg-ink" />Age gap</span>
        <span className="flex items-center gap-2"><span className="h-0.5 w-5 bg-moss" />Gender gap</span>
      </div>
    </figure>
  );
}

function Frontier() {
  const f = useQuery({ queryKey: ["frontier"], queryFn: api.frontier });
  const pts = (f.data?.points ?? []).filter((p) => p.auc > 0.6).map((p) => ({ ...p, gapPts: p.gap * 100 }));
  const ch = f.data?.chosen;
  const isChosen = (p: { method: string; eps: number | null }) => ch && p.method === ch.method && p.eps === ch.eps;
  const single = pts.filter((p) => p.method.startsWith("single"));
  const chosen = pts.filter(isChosen);
  const blind = pts.filter((p) => !isChosen(p) && !p.method.startsWith("single") && !p.uses_protected_at_decision);
  const rest = pts.filter((p) => !isChosen(p) && !p.method.startsWith("single") && p.uses_protected_at_decision);
  return (
    <figure>
      <figcaption>
        <p className="text-[15px] font-medium">Every policy the project tried</p>
        <p className="mt-1 text-[14px] text-soft">Lower and further left is better. Green is deployed, black is one cut-off,
          hollow squares need no gender or age when deciding. Cost is at the 5:1 ratio so every policy sits on the same axes.</p>
      </figcaption>
      <div className="mt-5 h-72">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: -8 }}>
            <CartesianGrid stroke={LINE} vertical={false} />
            <XAxis dataKey="gapPts" type="number" tick={{ fill: MUTE, fontSize: 12 }} axisLine={{ stroke: LINE }} tickLine={false}
              label={{ value: "Larger of the two gaps, points", position: "insideBottom", offset: -10, fill: MUTE, fontSize: 12 }} />
            <YAxis dataKey="cost" type="number" domain={["dataMin - 0.005", "dataMax + 0.005"]} tick={{ fill: MUTE, fontSize: 12 }} axisLine={false} tickLine={false} width={48} tickFormatter={(v) => v.toFixed(3)} />
            <ZAxis range={[46, 46]} />
            <ReferenceLine x={5} stroke={MUSTARD} strokeDasharray="4 4" />
            <Tooltip cursor={false} contentStyle={{ borderRadius: 12, border: `1px solid ${LINE}`, fontSize: 13 }}
              formatter={(v, n) => (n === "cost" ? [Number(v).toFixed(3), "Cost per applicant"] : [`${Number(v).toFixed(1)} points`, "Gap"])} />
            <Scatter data={rest} fill={MUTE} fillOpacity={0.45} isAnimationActive={false} />
            <Scatter data={single} fill={INK} isAnimationActive={false} />
            <Scatter data={blind} fill={MUSTARD} isAnimationActive={false} shape={(p: { cx?: number; cy?: number }) => (
              <rect x={(p.cx ?? 0) - 5} y={(p.cy ?? 0) - 5} width={10} height={10} fill="none" stroke={MUSTARD} strokeWidth={1.6} />
            )} />
            <Scatter data={chosen} fill={MOSS} isAnimationActive={false} shape={(p: { cx?: number; cy?: number }) => (
              <g><circle cx={p.cx} cy={p.cy} r={11} fill="none" stroke={MOSS} strokeWidth={1.5} /><circle cx={p.cx} cy={p.cy} r={5} fill={MOSS} /></g>
            )} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[13px] text-soft">The deployed policy maximises each loan's expected profit, not this 5:1 cost, so it sits higher here
        while earning more (see the Model page). Fairlearn's reductions method now minimises the same cost as every other policy; in 1.2 it
        minimised plain error and collapsed to approving almost everyone.</p>
    </figure>
  );
}
