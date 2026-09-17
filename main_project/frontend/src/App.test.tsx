import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App, { reviewValidationMessage, warningMessage } from "./App";

const config = {
  providers: [{ key: "global", label: "Global segmentation", available: true }],
  privacy_groups: ["Face", "License plate"],
  upload_max_bytes: 25 * 1024 * 1024,
  upload_max_pixels: 40_000_000,
  session_ttl_seconds: 3600,
};

const personalConfig = {
  ...config,
  policy_mode: "personal",
  default_privacy_groups: ["Face"],
  group_reliability: { Face: "reliable", "License plate": "check_manually" },
};

function stubConfig(payload: unknown) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => payload,
  }));
}

describe("reviewer shell", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("loads safe defaults and keeps analysis disabled before upload", async () => {
    stubConfig(config);
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

describe("personal mode shell", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("offers one-click erasing and warns that detection is incomplete", async () => {
    stubConfig(personalConfig);
    render(<App />);
    expect(await screen.findByRole("button", { name: /erase and save/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /check it myself first/i })).toBeInTheDocument();
    expect(screen.getByText(/it will miss things/i)).toBeInTheDocument();
    expect(screen.queryByText(/download stays blocked unless consent/i)).not.toBeInTheDocument();
  });

  it("translates reason codes into plain language", () => {
    expect(warningMessage("WARNING_EXPERIMENTAL_DETECTION_PROFILE")).toMatch(/experimental/i);
    expect(warningMessage("WARNING_PROVIDER_UNAVAILABLE_ZXINGCPP")).toMatch(/did not run/i);
    expect(warningMessage("WARNING_RESIDUAL_OCR_DETECTED")).toMatch(/still detectable/i);
    expect(warningMessage("WARNING_BARCODE_NOT_VERIFIED")).toMatch(/could not re-check/i);
  });
});
