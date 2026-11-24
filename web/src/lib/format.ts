const nf0 = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

export const money = (v: number | null | undefined) => (v == null ? "Not given" : nf0.format(Math.round(v)));
export const pct = (v: number | null | undefined, d = 1) => (v == null ? "None" : `${(v * 100).toFixed(d)}%`);
export const pts = (v: number) => `${(v * 100).toFixed(1)} points`;

// "Female|Under 35" -> "Women under 35"
export function groupLabel(key: string) {
  const [g, a] = key.split("|");
  const who = g === "Female" ? "Women" : g === "Male" ? "Men" : g;
  return `${who} ${a === "Under 35" ? "under 35" : "35 and over"}`;
}

export const BANDS = [
  { id: "A", upTo: 0.05, label: "A", note: "Low risk" },
  { id: "B", upTo: 0.1, label: "B", note: "" },
  { id: "C", upTo: 0.2, label: "C", note: "" },
  { id: "D", upTo: 0.35, label: "D", note: "" },
  { id: "E", upTo: 1, label: "E", note: "High risk" },
];
export const bandLetter = (band: string) => band.trim().charAt(0);

export function timeAgo(iso: string) {
  const s = Math.max(1, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}
