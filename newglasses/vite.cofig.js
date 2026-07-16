// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react"; // or "@vitejs/plugin-react-swc"

export default defineConfig({
  plugins: [react()], // Make sure this plugin is active
  // ... other config
});
