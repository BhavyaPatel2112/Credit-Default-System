import { useRef, useState } from "react";
import { motion } from "motion/react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import NumberFlow from "@number-flow/react";
import { Download, FileUp } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError, type BatchResult } from "../lib/api";
import { groupLabel, pct } from "../lib/format";
import { Headline } from "../components/Headline";
import { Button } from "../components/Button";

// Improvement 8: lenders score applications in batches. Every row of the
// file goes through the same validation and scoring as a single decision
// (api/batch.py), and every scored row is written to the decision log.
export function Batch() {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [name, setName] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const run = useMutation({
    mutationFn: ({ text, filename }: { text: string; filename: string }) => api.batch(text, filename),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["decisions"] }),
    onError: (e) => toast.error(e instanceof ApiError ? e.problems.join("; ") : String(e)),
  });

  const readFile = async (f: File) => {
    if (!f.name.toLowerCase().endsWith(".csv") && f.type !== "text/csv") {
      toast.error("Choose a .csv file: one application per row, a header row of field names first.");
      return;
    }
    setName(f.name);
    run.mutate({ text: await f.text(), filename: f.name });
  };
  const useSample = async () => {
    try {
      const text = await api.batchSample();
      setName("batch_sample.csv");
      run.mutate({ text, filename: "batch_sample.csv" });
    } catch (e) {
      toast.error(e instanceof ApiError ? e.problems[0] : "The scoring service is not reachable.");
    }
  };
  const downloadSample = async () => {
    try { save(await api.batchSample(), "batch_sample.csv"); } catch { toast.error("The scoring service is not reachable."); }
  };

  const d = run.data;
  return (
    <div className="mx-auto max-w-[1240px] px-4 pt-16 sm:px-6 sm:pt-20">
      <Headline quiet="Score a whole file" loud="the same way as one application." size="lg" className="max-w-4xl" />
      <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-soft">
        Upload a CSV of applications. Every row is checked and scored by the same service as a single decision, and
        written to the decision log. Rows that cannot be scored honestly are set aside with the problem in plain words,
        instead of being scored wrongly.
      </p>

      <section
        className={`mt-12 rounded-[24px] border-2 border-dashed p-8 text-center transition-colors sm:p-12 ${drag ? "border-moss bg-moss-2/40" : "border-line bg-white/60"}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) readFile(f); }}
      >
        <FileUp className="mx-auto h-8 w-8 text-mute" strokeWidth={1.5} aria-hidden />
        <p className="mt-4 text-[17px]">{run.isPending ? `Scoring ${name ?? "the file"}` : "Drop a CSV file here"}</p>
        <p className="mt-1 text-[14px] text-soft">
          One application per row. Column names as in the service schema, plus <code className="text-ink">gender</code> and{" "}
          <code className="text-ink">age_years</code>. Up to 5,000 rows.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <input ref={input} type="file" accept=".csv,text/csv" className="sr-only" aria-label="Choose a CSV file"
                 onChange={(e) => { const f = e.target.files?.[0]; if (f) readFile(f); e.target.value = ""; }} />
          <Button onClick={() => input.current?.click()} disabled={run.isPending}>Choose a file</Button>
          <Button variant="secondary" onClick={useSample} disabled={run.isPending}>Try the sample file</Button>
          <Button variant="ghost" onClick={downloadSample}><Download className="h-4 w-4" />Download the sample</Button>
        </div>
        <p className="mt-4 text-[13px] text-mute">The sample holds 200 real held-out applicants, half of whom defaulted (balanced for demonstration), and four deliberately broken rows.</p>
      </section>

      {d && <Results d={d} name={name} />}
    </div>
  );
}

function Results({ d, name }: { d: BatchResult; name: string | null }) {
  const notes = d.file_notes;
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <section className="mt-14 grid gap-6 sm:grid-cols-4">
        <Stat label="Rows in the file" v={d.rows} />
        <Stat label="Approved" v={d.approved} tone="text-moss" sub={d.scored ? pct(d.approved / d.scored) + " of scored" : undefined} />
        <Stat label="Referred to a person" v={d.referred} tone="text-mustard" sub={d.scored ? pct(d.referred / d.scored) + " of scored" : undefined} />
        <Stat label="Rejected by validation" v={d.rejected} sub={d.rejected ? "Not scored, see below" : "Every row was valid"} />
      </section>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <Button onClick={() => save(d.results_csv, (name ?? "batch").replace(/\.csv$/i, "") + "_decisions.csv")}>
          <Download className="h-4 w-4" />Download the results
        </Button>
        <span className="text-[13px] text-mute">Probability of default, decision, cut-off and reasons for every row; problems for every rejected row.</span>
      </div>

      {(notes.unknown_columns.length > 0 || notes.missing_columns.length > 0 || notes.ignored_columns.length > 0) && (
        <section className="mt-8 space-y-2 rounded-2xl bg-mustard-2/60 px-5 py-4 text-[14px]">
          {notes.missing_columns.length > 0 && <p><span className="font-medium">Missing columns:</span> {notes.missing_columns.join(", ")}. Every row needs them.</p>}
          {notes.unknown_columns.length > 0 && <p><span className="font-medium">Columns the service does not know:</span> {notes.unknown_columns.join(", ")}. Check the spelling against the schema.</p>}
          {notes.ignored_columns.length > 0 && <p className="text-soft"><span className="font-medium text-ink">Ignored columns:</span> {notes.ignored_columns.length} field(s) the model no longer uses, such as the time of application or contact details.</p>}
        </section>
      )}

      <section className="mt-14 grid gap-12 lg:grid-cols-[1fr_1.15fr] [&>*]:min-w-0">
        <div>
          <h2 className="text-[15px] font-medium">Approval by group in this batch</h2>
          <p className="mt-1 text-[14px] text-soft">
            {d.repayer_approval_gap != null
              ? <>The file carries known outcomes, so repayer approval is shown too. Gap between groups: <span className="text-ink tnum">{(d.repayer_approval_gap * 100).toFixed(1)} points</span> (limit 5).</>
              : <>Share of scored applications approved. Without known outcomes this is not the gated measure, but it shows the batch at a glance.</>}
          </p>
          <ul className="mt-5 space-y-4">
            {d.groups.map((g) => {
              return (
                <li key={g.group}>
                  <div className="flex items-baseline justify-between text-[14px]">
                    <span>{groupLabel(g.group)}</span>
                    <span className="text-soft tnum">{g.approved} of {g.scored} approved · {pct(g.approval_rate)}</span>
                  </div>
                  <div className="mt-2 h-2 rounded-full bg-stone-2">
                    <motion.div className="h-2 rounded-full bg-moss" initial={{ width: 0 }} animate={{ width: `${g.approval_rate * 100}%` }} transition={{ duration: 0.7 }} />
                  </div>
                  {g.approval_among_repayers != null && (
                    <p className="mt-1 text-[12px] text-mute tnum">Repayers approved {pct(g.approval_among_repayers)} of {g.repayers_known}. Mean probability of default {pct(g.mean_pd)}.</p>
                  )}
                </li>
              );
            })}
            {d.groups.length === 0 && <li className="text-soft">No row could be scored.</li>}
          </ul>
        </div>

        <div>
          <h2 className="text-[15px] font-medium">Rows set aside by validation</h2>
          <p className="mt-1 text-[14px] text-soft">Line numbers count the header as line 1, as a spreadsheet does.</p>
          {d.rejected_rows.length === 0 ? (
            <p className="mt-5 text-soft">None. Every row was complete and in range.</p>
          ) : (
            <ul className="mt-5 divide-y divide-line border-y border-line">
              {d.rejected_rows.map((r) => (
                <li key={r.row} className="py-3 text-[14px]">
                  <p className="font-medium">Line {r.line}{r.applicant_id ? <span className="font-normal text-soft"> · {r.applicant_id}</span> : null}</p>
                  <ul className="mt-1 list-disc space-y-0.5 pl-5 text-soft">{r.problems.map((p, i) => <li key={i}>{p}</li>)}</ul>
                </li>
              ))}
            </ul>
          )}
          {d.rejected_rows_truncated && <p className="mt-2 text-[13px] text-mute">Showing the first 500. The results file lists every one.</p>}
        </div>
      </section>

      <section className="mt-14">
        <h2 className="text-[15px] font-medium">Decisions</h2>
        <p className="mt-1 text-[14px] text-soft">The first {Math.min(d.results.length, 200)} rows. The download has all of them.</p>
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-[14px]">
            <thead className="text-[12px] text-mute"><tr>
              <th className="pb-2 font-normal">Row</th><th className="pb-2 font-normal">Applicant</th><th className="pb-2 font-normal">Group</th>
              <th className="pb-2 text-right font-normal">Probability</th><th className="pb-2 text-right font-normal">Cut-off</th>
              <th className="pb-2 pl-6 font-normal">Decision</th><th className="pb-2 font-normal">Main reason or problem</th>
            </tr></thead>
            <tbody>
              {d.results.map((r) => (
                <tr key={r.row} className="border-t border-line">
                  <td className="py-2.5 text-soft tnum">{r.row}</td>
                  <td className="py-2.5 text-soft tnum">{r.applicant_id ?? ""}</td>
                  <td className="py-2.5">{r.group ? groupLabel(r.group) : ""}</td>
                  <td className="py-2.5 text-right tnum">{r.probability_of_default != null ? pct(r.probability_of_default) : ""}</td>
                  <td className="py-2.5 text-right text-soft tnum">{r.cutoff != null ? pct(r.cutoff) : ""}</td>
                  <td className={`py-2.5 pl-6 ${r.status === "rejected" ? "text-soft" : r.decision === "approve" ? "text-moss" : "text-mustard"}`}>
                    {r.status === "rejected" ? "Not scored" : r.decision === "approve" ? "Approved" : "Referred"}
                  </td>
                  <td className="max-w-[360px] truncate py-2.5 text-soft" title={r.problems || r.reasons}>{r.status === "rejected" ? r.problems : (r.reasons ?? "").split("; ")[0]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </motion.div>
  );
}

function Stat({ label, v, tone = "text-ink", sub }: { label: string; v: number; tone?: string; sub?: string }) {
  return (
    <div className="rounded-2xl bg-white/70 p-5 ring-1 ring-line/60">
      <p className="text-[13px] text-soft">{label}</p>
      <p className={`display mt-2 text-[2.6rem] ${tone}`}><NumberFlow value={v} /></p>
      {sub && <p className="mt-1 text-[12px] text-mute">{sub}</p>}
    </div>
  );
}

function save(text: string, filename: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
