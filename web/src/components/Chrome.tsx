import { motion } from "motion/react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { href, type Route } from "../lib/router";
import { LinkButton } from "./Button";

export const PRODUCT = "Parity";

export function Mark({ className = "h-7 w-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="8" fill="currentColor" />
      <g fill="none" stroke="var(--color-stone)" strokeWidth="1.6" strokeLinecap="round">
        <path d="M7 21c3-1 5-6 9-6s6 4 9 3" />
        <path d="M7 16.5c3-1 5-5 9-5s6 3 9 2.5" opacity=".6" />
        <path d="M7 25c3-.6 5-4 9-4s6 2.6 9 2" opacity=".35" />
      </g>
    </svg>
  );
}

// Top strip: the release's real gate status, not marketing.
export function StatusBar() {
  const gate = useQuery({ queryKey: ["gate"], queryFn: api.gate, staleTime: 60_000 });
  const passed = gate.data?.passed;
  const text =
    gate.isLoading ? "Checking the release gate" :
    gate.isError ? "The scoring service is not running" :
    passed ? "passed" : "failed";
  return (
    <a href={href({ page: "model" })} className="block bg-stone-2/70 text-[13px] text-soft transition-colors hover:text-ink">
      <div className="mx-auto flex h-9 max-w-[1240px] items-center justify-center gap-2 px-4">
        <span className={`h-1.5 w-1.5 rounded-full ${gate.isError || passed === false ? "bg-mustard" : "bg-moss"}`} />
        <span>{text === "passed" ? <>This release passed its <span className="hidden sm:inline">accuracy, calibration and fairness </span>gate</> : text === "failed" ? "This release failed its gate" : text}</span>
        {!gate.isLoading && !gate.isError && <span className="hidden text-mute underline decoration-line underline-offset-4 sm:inline">See the checks</span>}
      </div>
    </a>
  );
}

const LINKS: { route: Route; label: string }[] = [
  { route: { page: "assess" }, label: "Assess" },
  { route: { page: "batch" }, label: "Batch" },
  { route: { page: "fairness" }, label: "Fairness" },
  { route: { page: "model" }, label: "Model" },
];

export function Nav({ route }: { route: Route }) {
  return (
    <header className="sticky top-0 z-40 border-b border-line/70 bg-stone/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-[1240px] items-center justify-between gap-4 px-4 sm:px-6">
        <a href="#/" className="flex items-center gap-2.5 text-ink" aria-label={`${PRODUCT} home`}>
          <Mark />
          <span className="text-[17px] font-semibold tracking-[-0.01em]">{PRODUCT}</span>
        </a>
        <nav className="flex items-center gap-1" aria-label="Main">
          {LINKS.map((l) => {
            const active = route.page === l.route.page;
            return (
              <a key={l.label} href={href(l.route)} aria-current={active ? "page" : undefined}
                 className={`relative rounded-full px-3.5 py-1.5 text-sm transition-colors ${active ? "text-ink" : "text-soft hover:text-ink"}`}>
                {active && (
                  <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-full bg-stone-2"
                               transition={{ type: "spring", stiffness: 420, damping: 34 }} />
                )}
                <span className="relative">{l.label}</span>
              </a>
            );
          })}
        </nav>
        <LinkButton href="#/assess/new" size="sm" className="hidden sm:inline-flex">New application</LinkButton>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="mt-28 border-t border-line">
      <div className="mx-auto grid max-w-[1240px] gap-8 px-4 py-12 text-sm text-soft sm:grid-cols-[1.4fr_1fr_1fr] sm:px-6">
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-ink"><Mark className="h-6 w-6" /><span className="font-semibold">{PRODUCT}</span></div>
          <p className="max-w-sm leading-relaxed">
            Credit default prediction with fairness constraints. A student project for CAP 501 Project Lifecycle
            Management, ATLAS SkillTech University.
          </p>
        </div>
        <div className="space-y-2">
          <p className="text-ink">Built on</p>
          <p className="leading-relaxed">Home Credit Default Risk data, XGBoost with isotonic calibration, SHAP explanations, Fairlearn.</p>
        </div>
        <div className="space-y-2">
          <p className="text-ink">For decision support only</p>
          <p className="leading-relaxed">Trained on public competition data. Not for decisions about real people.</p>
        </div>
      </div>
    </footer>
  );
}
