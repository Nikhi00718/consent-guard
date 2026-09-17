import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App, { reviewValidationMessage } from "./App";

const config = {
  providers: [{ key: "global", label: "Global segmentation", available: true }],
  privacy_groups: ["Face", "License plate"],
  upload_max_bytes: 25 * 1024 * 1024,
  upload_max_pixels: 40_000_000,
  session_ttl_seconds: 3600,
};

describe("reviewer shell", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("loads safe defaults and keeps analysis disabled before upload", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => config,
    }));
    render(<App />);
    expect(await screen.findByText("Drop a still image here")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run local analysis/i })).toBeDisabled();
    expect(screen.getByText(/download stays blocked unless consent/i)).toBeInTheDocument();
  });

  it("explains why an incomplete consent assertion cannot be verified", () => {
    expect(reviewValidationMessage({
      consentState: "GRANTED",
      subjectRef: "subject-01",
      audience: "Project team",
      purpose: "",
      reviewCompleted: true,
    })).toBe("Add a release purpose before verification.");
  });
});
