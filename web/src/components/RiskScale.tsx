import { motion } from "motion/react";
import { BANDS, groupLabel, pct } from "../lib/format";

// Probability of default on a square-root scale (0 to 60%), so the crowded
// low-risk end is readable. The marker springs to the new value; the tick is
// the cut-off for this applicant's policy group.
const MAX = 0.6;
const pos = (p: number) => Math.sqrt(Math.min(Math.max(p, 0), MAX) / MAX) * 100;

export function RiskScale({ pd, cutoff, group, dark = false }: { pd: number; cutoff: number; group?: string; dark?: boolean }) {
  const refer = pd >= cutoff;
  let prev = 0;
  return (
    <div className="select-none">
      <div className="relative h-14">
        {/* band track */}
        <div className="absolute inset-x-0 top-5 flex h-2 overflow-hidden rounded-full">
          {BANDS.map((b, i) => {
            const left = pos(prev), right = pos(Math.min(b.upTo, MAX));
            prev = b.upTo;
            return (
              <div key={b.id} style={{ width: `${right - left}%` }}
                   className={dark ? ["bg-white/12", "bg-white/20", "bg-white/28", "bg-white/38", "bg-white/50"][i]
                                   : ["bg-stone-3", "bg-[#cbcac3]", "bg-[#b3b2aa]", "bg-[#8f8e86]", "bg-[#5f5e58]"][i]} />
            );
          })}
        </div>
        {/* cut-off */}
        <div className="absolute top-2.5 h-7 w-px" style={{ left: `${pos(cutoff)}%` }}>
          <div className={`h-full w-px ${dark ? "bg-stone" : "bg-ink"}`} />
          <span className={`absolute -top-3 left-1/2 -translate-x-1/2 whitespace-nowrap text-[11px] ${dark ? "text-stone/70" : "text-soft"}`}>
            cut-off {pct(cutoff)}
          </span>
        </div>
        {/* marker */}
        <motion.div
          className="absolute top-[13px] -ml-2.5"
          initial={false}
          animate={{ left: `${pos(pd)}%` }}
          transition={{ type: "spring", stiffness: 140, damping: 20 }}
        >
          <div className={`h-5 w-5 rounded-full border-[3px] shadow-sm ${dark ? "border-stage" : "border-stone"} ${refer ? "bg-mustard" : "bg-moss"}`} />
        </motion.div>
      </div>
      <div className={`flex justify-between text-[11px] tnum ${dark ? "text-stone/50" : "text-mute"}`}>
        <span>0%</span>
        <span className="hidden sm:inline">Bands A to E</span>
        <span>60% and above</span>
      </div>
      {group && (
        <p className={`mt-2 text-[13px] ${dark ? "text-stone/70" : "text-soft"}`}>
          Cut-off for {groupLabel(group).toLowerCase()}. Gender and age choose the cut-off only; the model never sees them.
        </p>
      )}
    </div>
  );
}
