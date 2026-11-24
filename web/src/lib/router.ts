import { useEffect, useState } from "react";

// Hash routing (#/assess/123) so FastAPI can serve the console as plain
// static files without server-side route handling.
export type Route =
  | { page: "home" }
  | { page: "assess"; id?: number; mode?: "new" }
  | { page: "fairness" }
  | { page: "batch" }
  | { page: "model" };

export function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "assess") {
    if (parts[1] === "new") return { page: "assess", mode: "new" };
    const id = Number(parts[1]);
    return Number.isFinite(id) && parts[1] ? { page: "assess", id } : { page: "assess" };
  }
  if (parts[0] === "fairness") return { page: "fairness" };
  if (parts[0] === "model") return { page: "model" };
  if (parts[0] === "batch") return { page: "batch" };
  return { page: "home" };
}

export const href = (r: Route) =>
  r.page === "home" ? "#/" : r.page === "assess" ? `#/assess${r.mode ? "/new" : r.id ? `/${r.id}` : ""}` : `#/${r.page}`;

export function go(r: Route) {
  window.location.hash = href(r);
}

export function useRoute() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => {
      setRoute(parse(window.location.hash));
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
