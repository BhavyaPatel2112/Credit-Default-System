import { motion } from "motion/react";
import type { Factor } from "../lib/api";
import { plain } from "../lib/reasons";

// What drove the score, for the loan officer: each concept's total effect,
// risk-raising to the right in ink, risk-lowering to the left in moss.
export function Factors({ factors }: { factors: Factor[] }) {
  const max = Math.max(...factors.map((f) => Math.abs(f.shap_logodds)), 0.01);
  return (
    <ul className="divide-y divide-line">
      {factors.map((f, i) => {
        const up = f.effect === "raises risk";
        const w = (Math.abs(f.shap_logodds) / max) * 50;
        return (
          <li key={f.reason} className="grid grid-cols-1 gap-2 py-3.5 sm:grid-cols-[1fr_200px] sm:items-center sm:gap-6">
            <div>
              <p className="text-[15px] text-ink">{f.reason}</p>
              <p className="mt-0.5 text-[13px] leading-snug text-soft">{plain(f.reason, f.effect)}</p>
            </div>
            <div className="relative h-6" aria-label={`${f.effect}, strength ${Math.abs(f.shap_logodds).toFixed(2)}`}>
              <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${w}%` }}
                transition={{ duration: 0.7, delay: i * 0.05, ease: [0.22, 1, 0.36, 1] }}
                className={`absolute top-1.5 h-3 rounded-full ${up ? "left-1/2 bg-ink" : "right-1/2 bg-moss"}`}
              />
              <span className={`absolute top-0 text-[11px] text-mute ${up ? "left-0" : "right-0"}`}>
                {up ? "raises risk" : "lowers risk"}
              </span>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
