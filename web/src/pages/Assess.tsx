import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError, EXAMPLE, type ApplicantRow, type Decision, type Request } from "../lib/api";
import { groupLabel, money, pct } from "../lib/format";
import { go, type Route } from "../lib/router";
import { Headline } from "../components/Headline";
import { DecisionView } from "../components/DecisionView";
import { ApplicationForm, friendly } from "../components/ApplicationForm";
import { Button, LinkButton } from "../components/Button";

type Filter = "all" | "refer" | "approve";

export function Assess({ route }: { route: Extract<Route, { page: "assess" }> }) {
  const list = useQuery({ queryKey: ["applicants"], queryFn: api.applicants });
  return (
    <div className="mx-auto max-w-[1240px] px-4 pt-10 sm:px-6 sm:pt-14">
      <div className="grid gap-10 lg:grid-cols-[340px_1fr] lg:gap-14">
        <aside className="lg:sticky lg:top-24 lg:h-[calc(100vh-7.5rem)]">
          <Picker rows={list.data?.applicants ?? []} available={list.data?.available} note={list.data?.note}
                  loading={list.isLoading} error={list.error} selected={route.id} />
        </aside>
        <main className="min-w-0 pb-10">
          {route.mode === "new" ? <NewApplication key="new" /> :
           route.id ? <Existing key={route.id} id={route.id} /> : <Empty available={!!list.data?.available} />}
        </main>
      </div>
    </div>
  );
}

function Picker({ rows, available, note, loading, error, selected }: {
  rows: ApplicantRow[]; available?: boolean; note?: string; loading: boolean; error: unknown; selected?: number;
}) {
  const [q, setQ] = useState("");
  const [f, setF] = useState<Filter>("all");
  const shown = useMemo(() => rows.filter((r) =>
    (f === "all" || r.decision === f) && (!q || String(r.id).includes(q.trim()) || groupLabel(r.group).toLowerCase().includes(q.toLowerCase())),
  ), [rows, q, f]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-baseline justify-between">
        <h1 className="text-[15px] font-medium">Applicants</h1>
        <a href="#/assess/new" className="text-[14px] text-soft underline decoration-line underline-offset-4 hover:text-ink">New application</a>
      </div>
      {error ? (
        <p className="mt-4 text-sm text-mustard">{(error as Error).message}</p>
      ) : available === false ? (
        <p className="mt-4 text-[14px] leading-relaxed text-soft">{note}</p>
      ) : (
        <>
          <label className="mt-4 flex h-10 items-center gap-2 rounded-full bg-white px-4 ring-1 ring-line focus-within:ring-ink/30">
            <Search className="h-4 w-4 text-mute" aria-hidden />
            <span className="sr-only">Search by ID or group</span>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by ID or group" className="min-w-0 flex-1 bg-transparent text-[14px] outline-none placeholder:text-mute" />
          </label>
          <div className="mt-3 flex gap-1 rounded-full bg-stone-2/80 p-1 text-[13px]" role="tablist" aria-label="Filter by decision">
            {(["all", "refer", "approve"] as Filter[]).map((k) => (
              <button key={k} role="tab" aria-selected={f === k} onClick={() => setF(k)}
                className={`relative flex-1 rounded-full py-1.5 transition-colors ${f === k ? "text-ink" : "text-soft hover:text-ink"}`}>
                {f === k && <motion.span layoutId="filter-pill" className="absolute inset-0 rounded-full bg-white shadow-sm" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
                <span className="relative">{{ all: "All", refer: "Referred", approve: "Approved" }[k]}</span>
              </button>
            ))}
          </div>
          <ul className="mt-3 max-h-[46vh] flex-1 overflow-y-auto pr-1 lg:max-h-none" aria-label="Applicants">
            {loading && Array.from({ length: 8 }).map((_, i) => <li key={i} className="my-1 h-14 animate-pulse rounded-xl bg-stone-2/70" />)}
            {shown.map((r) => (
              <li key={r.id}>
                <a href={`#/assess/${r.id}`} aria-current={selected === r.id ? "true" : undefined}
                   className={`flex items-center justify-between gap-3 rounded-xl px-3 py-2.5 transition-colors ${selected === r.id ? "bg-white ring-1 ring-line" : "hover:bg-stone-2/70"}`}>
                  <span className="min-w-0">
                    <span className="block text-[14px] tnum">{r.id}</span>
                    <span className="block truncate text-[12px] text-soft">{groupLabel(r.group)}, {money(r.loan)}</span>
                  </span>
                  <span className="flex shrink-0 items-center gap-2 text-[13px] tnum">
                    {pct(r.pd)}
                    <span className={`h-2 w-2 rounded-full ${r.decision === "approve" ? "bg-moss" : "bg-mustard"}`} title={r.decision === "approve" ? "Approved" : "Referred"} />
                  </span>
                </a>
              </li>
            ))}
            {!loading && shown.length === 0 && <li className="px-3 py-6 text-[14px] text-soft">No applicants match. Clear the search or the filter.</li>}
          </ul>
          {note && <p className="mt-3 text-[12px] leading-relaxed text-mute">{note}</p>}
        </>
      )}
    </div>
  );
}

function Empty({ available }: { available: boolean }) {
  return (
    <div className="flex min-h-[60vh] flex-col justify-center">
      <Headline quiet="Pick an applicant," loud="or start a new application." size="lg" />
      <p className="mt-5 max-w-lg leading-relaxed text-soft">
        {available
          ? "The list holds real applicants the model never saw during training. Open one to see the decision, the reasons, and what would change it."
          : "Fill in an application to score it. The list of real applicants appears once the demo set is built from the raw data."}
      </p>
      <div className="mt-8 flex gap-3">
        <LinkButton href="#/assess/new">New application</LinkButton>
      </div>
    </div>
  );
}

function useScoring() {
  const [problems, setProblems] = useState<string[]>([]);
  const qc = useQueryClient();
  const m = useMutation({
    mutationFn: (r: Request) => api.score(r),
    onMutate: () => setProblems([]),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["decisions"] }),
    onError: (e) => {
      const p = e instanceof ApiError ? e.problems : [String(e)];
      setProblems(p);
      toast.error(p.length > 1 ? `${p.length} problems with the application` : friendly(p[0]));
    },
  });
  return { ...m, problems };
}

function Summary({ r, id }: { r: Request; id?: number }) {
  const a = r.application;
  const items = [
    ["Group", groupLabel(`${r.gender}|${r.age_years < 35 ? "Under 35" : "35 and over"}`)],
    ["Loan", money(Number(a.AMT_CREDIT))],
    ["Instalment", money(Number(a.AMT_ANNUITY))],
    ["Income", money(Number(a.AMT_INCOME_TOTAL))],
    ["Credit history", Object.keys(r.history).some((k) => k.startsWith("BUR_")) ? "On file" : "None with other lenders"],
  ];
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-3 border-y border-line py-4 text-[14px] sm:grid-cols-5">
      {id && <div className="col-span-2 sm:col-span-5"><dt className="sr-only">Applicant</dt><dd className="text-soft">Applicant <span className="text-ink tnum">{id}</span>, held-out test set</dd></div>}
      {items.map(([k, v]) => <div key={k}><dt className="text-[12px] text-mute">{k}</dt><dd className="mt-0.5 tnum">{v}</dd></div>)}
    </dl>
  );
}

function Existing({ id }: { id: number }) {
  const q = useQuery({ queryKey: ["applicant", id], queryFn: () => api.applicant(id), retry: false });
  const [req, setReq] = useState<Request | null>(null);
  const [editing, setEditing] = useState(false);
  const score = useScoring();
  const active = req ?? q.data?.request;
  const first = useQuery({ queryKey: ["score", active], enabled: !!active && !score.data, queryFn: () => api.score(active!) });
  const decision: Decision | undefined = score.data ?? first.data;
  const changed = !!req;

  if (q.isError) return (
    <div className="py-20">
      <Headline quiet={`Applicant ${id}`} loud="is not in the demo set." size="md" />
      <p className="mt-4 text-soft">Pick someone from the list, or start a new application.</p>
    </div>
  );

  const run = (r: Request) => { setReq(r); setEditing(false); score.mutate(r); };

  return (
    <div className="space-y-8">
      {active && <Summary r={active} id={id} />}
      {changed && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-moss-2/70 px-4 py-3 text-[14px]">
          <span>Showing a changed version of this application.</span>
          <Button size="sm" variant="ghost" onClick={() => { setReq(null); score.reset(); }}>Back to the original</Button>
        </div>
      )}
      <AnimatePresence mode="wait">
        {editing && active ? (
          <motion.div key="form" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <ApplicationForm initial={active} onSubmit={run} busy={score.isPending} problems={score.problems} />
          </motion.div>
        ) : decision && active ? (
          <motion.div key="decision" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <DecisionView decision={decision} request={active} actual={changed ? undefined : q.data?.actual} onTry={run} busy={score.isPending} />
            <div className="mt-8"><Button variant="secondary" onClick={() => setEditing(true)}>Edit this application</Button></div>
          </motion.div>
        ) : (
          <Loading key="loading" />
        )}
      </AnimatePresence>
    </div>
  );
}

function NewApplication() {
  const score = useScoring();
  const [req, setReq] = useState<Request>(EXAMPLE);
  const [editing, setEditing] = useState(true);
  const run = (r: Request) => { setReq(r); score.mutate(r, { onSuccess: () => setEditing(false) }); };

  return (
    <div className="space-y-8">
      <div>
        <Headline quiet="New application." loud="Fill in what the applicant gave you." size="md" />
        <p className="mt-4 max-w-xl text-soft">Prefilled with an example so you can assess straight away. Everything you change is checked by the service before it is scored.</p>
      </div>
      <AnimatePresence mode="wait">
        {editing || !score.data ? (
          <motion.div key="form" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <ApplicationForm initial={req} onSubmit={run} busy={score.isPending} problems={score.problems} />
          </motion.div>
        ) : (
          <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Summary r={req} />
            <div className="mt-8">
              <DecisionView decision={score.data} request={req} onTry={run} busy={score.isPending} />
            </div>
            <div className="mt-8 flex gap-3">
              <Button variant="secondary" onClick={() => setEditing(true)}>Edit this application</Button>
              <Button variant="ghost" onClick={() => { setReq(EXAMPLE); score.reset(); setEditing(true); go({ page: "assess", mode: "new" }); }}>Start over</Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Loading() {
  return (
    <div className="space-y-4 py-6" aria-busy>
      <div className="h-14 w-2/3 animate-pulse rounded-2xl bg-stone-2" />
      <div className="h-5 w-1/2 animate-pulse rounded-full bg-stone-2" />
      <div className="mt-8 h-16 animate-pulse rounded-2xl bg-stone-2" />
      <div className="h-40 animate-pulse rounded-2xl bg-stone-2" />
    </div>
  );
}
