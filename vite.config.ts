import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // base must stay "/" for Netlify (a subpath base would blank the page)
  base: "/",
});
