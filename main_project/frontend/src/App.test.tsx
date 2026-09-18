import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App, { checkCopy, classLabel, decisionTitle, reviewValidationMessage, warningMessage } from "./App";

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
    expect(await screen.findByText("Drop a photo here")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run local analysis/i })).toBeDisabled();
    expect(screen.getByText(/download stays blocked unless consent/i)).toBeInTheDocument();
  });

  it("offers a retry when the local app cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Failed to fetch")));
    render(<App />);
    expect(await screen.findByRole("button", { name: /try again/i })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/could not reach the consentguard app/i);
    expect(screen.queryByText("Drop a photo here")).not.toBeInTheDocument();
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

  it("names every check and decision in plain words", () => {
    expect(checkCopy({ name: "attack_face", status: "PASS", reason_code: "FACE_ATTACK_PASSED" })).toEqual({
      title: "Faces",
      detail: "Re-scanned the saved file: no faces found.",
    });
    expect(checkCopy({ name: "metadata", status: "FAIL", reason_code: "METADATA_PRESENT" }).detail).toMatch(/hidden file data/i);
    expect(checkCopy({ name: "attack_plate", status: "UNCERTAIN", reason_code: "PLATE_ATTACK_UNCERTAIN" }).detail).toMatch(/check by eye/i);
    expect(checkCopy({ name: "something_new", status: "PASS", reason_code: "SOMETHING_NEW_OK" })).toEqual({ title: "Something New", detail: "Something New Ok" });
    expect(decisionTitle("ALLOW_REDACTED")).toBe("Ready to download");
    expect(decisionTitle("REJECT_EXPORT")).toBe("Download blocked");
  });

  it("reads detector class ids as the thing they cover", () => {
    expect(classLabel("a108_license_plate_all")).toBe("License plate");
    expect(classLabel("printed_text")).toBe("Text");
    expect(classLabel("a43_medicine")).toBe("Medicine");
  });
});
