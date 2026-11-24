import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { AnimatePresence, motion, MotionConfig } from "motion/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import "./styles.css";
import { useRoute } from "./lib/router";
import { Footer, Nav, StatusBar } from "./components/Chrome";
import { Landing } from "./pages/Landing";
import { Assess } from "./pages/Assess";
// The chart-heavy pages load on demand, so the landing page stays light.
const Fairness = lazy(() => import("./pages/Fairness").then((m) => ({ default: m.Fairness })));
const Model = lazy(() => import("./pages/Model").then((m) => ({ default: m.Model })));
const Batch = lazy(() => import("./pages/Batch").then((m) => ({ default: m.Batch })));

const client = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } } });

function App() {
  const route = useRoute();
  const key = route.page;
  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-full focus:bg-ink focus:px-4 focus:py-2 focus:text-stone">Skip to content</a>
      <StatusBar />
      <Nav route={route} />
      <AnimatePresence mode="wait">
        <motion.div id="main" key={key} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
          {route.page === "home" && <Landing />}
          {route.page === "assess" && <Assess route={route} />}
          <Suspense fallback={<div className="min-h-[70vh]" />}>
            {route.page === "fairness" && <Fairness />}
            {route.page === "model" && <Model />}
            {route.page === "batch" && <Batch />}
          </Suspense>
        </motion.div>
      </AnimatePresence>
      <Footer />
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <MotionConfig reducedMotion="user">
        <App />
        <Toaster position="bottom-center" toastOptions={{ style: { borderRadius: 14, fontFamily: "var(--font-sans)", fontSize: 14 } }} />
      </MotionConfig>
    </QueryClientProvider>
  </StrictMode>,
);
