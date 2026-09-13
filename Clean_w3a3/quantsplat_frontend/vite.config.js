import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Reuse the actively curated demo pack. Vite copies this directory into dist,
// allowing the deployed React site to serve the research explorer without Gradio.
export default defineConfig({
  plugins: [react()],
  publicDir: "../quantsplat_demo/data",
});
