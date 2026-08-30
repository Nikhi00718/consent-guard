import { defineConfig } from "@playwright/test";

const apiURL = "http://127.0.0.1:7861";
const frontendURL = "http://127.0.0.1:5174";
const python = process.env.CONSENTGUARD_PYTHON ?? (
  process.platform === "win32" ? "../../.venv/Scripts/python.exe" : "../../.venv/bin/python"
);

export default defineConfig({
  testDir: "./e2e",
  use: {
    baseURL: frontendURL,
    trace: "on-first-retry",
  },
  webServer: [
    {
      command: `"${python}" ../scripts/stage_05_review_export/run_e2e_fixture_api.py --port 7861`,
      url: `${apiURL}/health`,
      reuseExistingServer: false,
    },
    {
      command: "npm run dev -- --port 5174 --strictPort",
      url: frontendURL,
      reuseExistingServer: false,
      env: { CONSENTGUARD_API_TARGET: apiURL },
    },
  ],
});
