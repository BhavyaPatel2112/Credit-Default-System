import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// `npm run dev` serves the UI on :5173 and forwards API calls to the
// FastAPI service on :8000. `npm run build` writes web/dist, which FastAPI
// serves at "/" so no Node is needed to run the finished console.
const api = "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": api, "/score": api, "/schema": api, "/health": api } },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
});
