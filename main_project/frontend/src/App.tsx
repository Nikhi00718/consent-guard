import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Check, CheckCircle, CircleNotch, DownloadSimple, LockKey, MinusCircle, Question, SlidersHorizontal, Trash, UploadSimple, Warning, X, XCircle } from "@phosphor-icons/react";
import { api } from "./api";
import type { MaskEditorHandle } from "./MaskEditor";
import type { Analysis, AppConfig, Asset, AssuranceCheck, ConsentState, PolicyMode, RenderResult, Session } from "./types";

type Phase = "upload" | "ready" | "analyzing" | "review" | "verifying" | "result";

type ReviewFields = {
  consentState: ConsentState;
  subjectRef: string;
  audience: string;
  purpose: string;
  reviewCompleted: boolean;
};

const steps = ["Add photo", "Find", "Check", "Save"];
const MaskEditor = lazy(() => import("./MaskEditor"));

function readable(value: string): string {
  return value.toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

/** Detector class ids such as "a108_license_plate_all" read as the thing they cover. */
const CLASS_LABELS: Record<string, string> = {
  face: "Face",
  license_plate: "License plate",
  person_body: "Person",
  nudity: "Nudity",
  handwriting: "Handwriting",
  printed_text: "Text",
  disability_physical: "Physical disability",
  medicine: "Medicine",
  fingerprint: "Fingerprint",
  signature: "Signature",
  barcode: "Barcode / QR",
};

export function classLabel(value: string): string {
  const key = value.replace(/^a\d+_/, "").replace(/_all$/, "");
  return CLASS_LABELS[key] ?? readable(key);
}

type CheckCopy = { title: string; PASS: string; FAIL: string; UNCERTAIN: string; NOT_RUN: string };

function rescan(thing: string, found: string): Omit<CheckCopy, "title"> {
  return {
    PASS: `Re-scanned the saved file: no ${thing} found.`,
    FAIL: `The re-scan still found ${found}. Fix the cover before sharing.`,
    UNCERTAIN: `The ${thing} re-scan couldn't give a clear answer. Check by eye.`,
    NOT_RUN: `The ${thing} re-scan didn't run, so check by eye.`,
  };
}

const CHECK_COPY: Record<string, CheckCopy> = {
  render: { title: "Cover applied", PASS: "The cover was burned into the pixels.", FAIL: "The cover could not be applied.", UNCERTAIN: "Couldn't confirm the cover was applied.", NOT_RUN: "The cover step didn't run." },
  pixel_decode: { title: "Opens as a normal image", PASS: "Re-opened at full size without errors.", FAIL: "The saved file didn't open at the right size.", UNCERTAIN: "Couldn't confirm the file opens correctly.", NOT_RUN: "The file wasn't re-opened." },
  output_hash: { title: "Written as a new file", PASS: "Encoded from scratch, and unchanged since it was written.", FAIL: "The file wasn't freshly encoded, or changed after it was written.", UNCERTAIN: "Couldn't confirm the file was freshly encoded.", NOT_RUN: "The file's fingerprint wasn't checked." },
  metadata: { title: "Location and camera data", PASS: "None left in the file.", FAIL: "Some hidden file data is still in there.", UNCERTAIN: "Couldn't read the file's hidden data.", NOT_RUN: "Hidden file data wasn't checked." },
  attack_face: { title: "Faces", ...rescan("faces", "a face") },
  attack_plate: { title: "License plates", ...rescan("plates", "a plate") },
  attack_ocr: { title: "Readable text", ...rescan("text", "readable text") },
  attack_barcode: { title: "QR codes and barcodes", ...rescan("codes", "a code") },
};

const STATUS_WORD: Record<AssuranceCheck["status"], string> = { PASS: "Clear", FAIL: "Problem", UNCERTAIN: "Unsure", NOT_RUN: "Skipped" };

export function checkCopy(check: Pick<AssuranceCheck, "name" | "status" | "reason_code">): { title: string; detail: string } {
  const copy = CHECK_COPY[check.name];
  if (!copy) return { title: readable(check.name), detail: readable(check.reason_code) };
  return { title: copy.title, detail: copy[check.status] ?? readable(check.reason_code) };
}

export function decisionTitle(action: string): string {
  if (action === "ALLOW_REDACTED") return "Ready to download";
  if (action === "ALLOW_PIXELS_UNCHANGED") return "Ready to download, nothing covered";
  if (action === "HOLD_FOR_REVIEW") return "Needs your review";
  if (action === "HOLD_FOR_CONSENT") return "Waiting for consent";
  if (action === "HOLD_PROFILE_NOT_RELEASE_READY") return "Held: detector settings not approved";
  if (action === "REJECT_EXPORT") return "Download blocked";
  return readable(action);
}

function statusIcon(status: string) {
  if (status === "PASS") return <CheckCircle weight="fill" aria-hidden="true" />;
  if (status === "FAIL") return <XCircle weight="fill" aria-hidden="true" />;
  if (status === "NOT_RUN") return <MinusCircle weight="fill" aria-hidden="true" />;
  return <Question weight="fill" aria-hidden="true" />;
}

function activeStep(phase: Phase): number {
  if (phase === "upload" || phase === "ready") return 0;
  if (phase === "analyzing") return 1;
  if (phase === "review") return 2;
  return 3;
}

function percentOf(pixels: number, width: number, height: number): string {
  const share = (pixels / Math.max(1, width * height)) * 100;
  if (share < 1) return "under 1% of the photo";
  return `${Math.round(share)}% of the photo`;
}

export function reviewValidationMessage(fields: ReviewFields): string {
  if (fields.consentState !== "GRANTED") return "Set Consent state to Granted before verification.";
  if (!fields.subjectRef.trim()) return "Add a non-PII subject reference before verification.";
  if (!fields.audience.trim()) return "Add the intended audience before verification.";
  if (!fields.purpose.trim()) return "Add a release purpose before verification.";
  if (!fields.reviewCompleted) return "Confirm that you inspected the full image before verification.";
  return "";
}

/** Plain-language summary of a reason code, so the result screen never shows bare enums. */
export function warningMessage(code: string): string {
  if (code === "WARNING_EXPERIMENTAL_DETECTION_PROFILE") return "Detection settings are experimental, not calibrated for guarantees.";
  if (code === "WARNING_DETECTED_REGIONS_LEFT_VISIBLE") return "You left detected regions visible in this export.";
  if (code.startsWith("WARNING_RESIDUAL_")) return `Something was still detectable in the saved file: ${readable(code.replace("WARNING_RESIDUAL_", "").replace("_DETECTED", ""))}.`;
  if (code.startsWith("WARNING_PROVIDER_UNAVAILABLE_")) return `A detector did not run: ${readable(code.replace("WARNING_PROVIDER_UNAVAILABLE_", ""))}.`;
  if (code.endsWith("_NOT_VERIFIED")) return `Could not re-check the saved file for ${readable(code.replace("WARNING_", "").replace("_NOT_VERIFIED", ""))}.`;
  return readable(code.replace("WARNING_", ""));
}

function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <rect x="6.5" y="9.5" width="11" height="5" fill="var(--marker)" />
    </svg>
  );
}

function CropCorners() {
  return <><span className="corner tl" /><span className="corner tr" /><span className="corner br" /><span className="corner bl" /></>;
}

export default function App() {
  const editorRef = useRef<MaskEditorHandle>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const pageHeading = useRef<HTMLHeadingElement>(null);
  const consentInput = useRef<HTMLSelectElement>(null);
  const subjectInput = useRef<HTMLInputElement>(null);
  const audienceInput = useRef<HTMLInputElement>(null);
  const purposeInput = useRef<HTMLTextAreaElement>(null);
  const reviewInput = useRef<HTMLInputElement>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [configFailed, setConfigFailed] = useState(false);
  const [session, setSession] = useState<Session | null>(null);
  const [asset, setAsset] = useState<Asset | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [result, setResult] = useState<RenderResult | null>(null);
  const [fileName, setFileName] = useState("");
  const [phase, setPhase] = useState<Phase>("upload");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [providerKeys, setProviderKeys] = useState<string[]>([]);
  const [privacyGroups, setPrivacyGroups] = useState<string[]>([]);
  const [consentState, setConsentState] = useState<ConsentState>("UNKNOWN");
  const [subjectRef, setSubjectRef] = useState("");
  const [operation, setOperation] = useState("share");
  const [audience, setAudience] = useState("");
  const [purpose, setPurpose] = useState("");
  const [reviewCompleted, setReviewCompleted] = useState(false);
  const [reviewError, setReviewError] = useState("");

  const mode: PolicyMode = config?.policy_mode ?? "research";
  const personal = mode === "personal";

  const loadConfig = useCallback(() => {
    setConfigFailed(false);
    setError("");
    api.config()
      .then((next) => {
        setConfig(next);
        setProviderKeys(next.providers.filter((provider) => provider.available).map((provider) => provider.key));
        setPrivacyGroups(next.default_privacy_groups?.length ? next.default_privacy_groups : next.privacy_groups);
      })
      .catch((reason: Error) => {
        setConfigFailed(true);
        setError(`Could not reach the ConsentGuard app running on this computer. ${reason.message}`);
      });
  }, []);

  useEffect(loadConfig, [loadConfig]);

  // Moving to a new screen puts keyboard and screen-reader focus on its heading.
  useEffect(() => {
    if (phase === "review" || phase === "result") pageHeading.current?.focus();
  }, [phase]);

  // The staged copy of a private photo should not survive the tab.
  useEffect(() => {
    if (!session) return undefined;
    const sessionId = session.session_id;
    const abandon = () => api.abandonSession(sessionId);
    window.addEventListener("pagehide", abandon);
    return () => window.removeEventListener("pagehide", abandon);
  }, [session]);

  const progress = activeStep(phase);
  const reliability = config?.group_reliability ?? {};
  const unreliableSelected = useMemo(
    () => privacyGroups.filter((group) => (reliability[group] ?? "reliable") !== "reliable"),
    [privacyGroups, reliability],
  );

  const reset = async () => {
    if (session) await api.deleteSession(session.session_id).catch(() => undefined);
    setSession(null);
    setAsset(null);
    setAnalysis(null);
    setResult(null);
    setFileName("");
    setPhase("upload");
    setError("");
    setConsentState("UNKNOWN");
    setSubjectRef("");
    setAudience("");
    setPurpose("");
    setReviewCompleted(false);
    setReviewError("");
  };

  const upload = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("That isn't a photo. Choose a JPEG, PNG, or WebP image.");
      return;
    }
    setError("");
    setUploading(true);
    try {
      if (session) await api.deleteSession(session.session_id).catch(() => undefined);
      const nextSession = await api.createSession();
      const nextAsset = await api.upload(nextSession.session_id, file);
      setSession(nextSession);
      setAsset(nextAsset);
      setFileName(file.name || "Pasted image");
      setPhase("ready");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The image could not be ingested.");
    } finally {
      setUploading(false);
    }
  };

  // Screenshots usually live on the clipboard, so pasting is as good as dropping.
  const onIngestScreen = phase === "upload" || phase === "ready";
  const uploadRef = useRef(upload);
  uploadRef.current = upload;
  const hasConfig = Boolean(config);
  useEffect(() => {
    if (!onIngestScreen || !hasConfig) return undefined;
    const onPaste = (event: ClipboardEvent) => {
      const file = Array.from(event.clipboardData?.files ?? []).find((item) => item.type.startsWith("image/"));
      if (!file) return;
      event.preventDefault();
      void uploadRef.current(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [onIngestScreen, hasConfig]);

  const runAnalysis = async (): Promise<Analysis | null> => {
    if (!session || !providerKeys.length) return null;
    setPhase("analyzing");
    setError("");
    try {
      const next = await api.analyze(session.session_id, providerKeys, privacyGroups);
      setAnalysis(next);
      return next;
    } catch (reason) {
      setPhase("ready");
      setError(reason instanceof Error ? reason.message : "Analysis failed.");
      return null;
    }
  };

  const analyze = async () => {
    const next = await runAnalysis();
    if (next) setPhase("review");
  };

  /** One click: detect everything, erase it, verify the file, offer the download. */
  const eraseAndSave = async () => {
    const next = await runAnalysis();
    if (!next || !session) return;
    setPhase("verifying");
    try {
      setResult(await api.autoRedact(session.session_id));
      setPhase("result");
    } catch (reason) {
      setPhase("review");
      setError(reason instanceof Error ? reason.message : "Redaction failed.");
    }
  };

  const render = async () => {
    if (!session) return;
    if (!editorRef.current) {
      setReviewError("The editing tools are still loading. Try again in a moment.");
      return;
    }
    if (!personal) {
      const validationMessage = reviewValidationMessage({ consentState, subjectRef, audience, purpose, reviewCompleted });
      if (validationMessage) {
        setReviewError(validationMessage);
        if (consentState !== "GRANTED") consentInput.current?.focus();
        else if (!subjectRef.trim()) subjectInput.current?.focus();
        else if (!audience.trim()) audienceInput.current?.focus();
        else if (!purpose.trim()) purposeInput.current?.focus();
        else reviewInput.current?.focus();
        return;
      }
    }
    setReviewError("");
    setPhase("verifying");
    setError("");
    try {
      const mask = await editorRef.current.exportMask();
      const next = personal
        ? await api.render(session.session_id, mask)
        : await api.render(session.session_id, mask, { consentState, subjectRef, operation, audience, purpose, reviewCompleted });
      setResult(next);
      setPhase("result");
    } catch (reason) {
      setPhase("review");
      setError(reason instanceof Error ? reason.message : "Verification failed.");
    }
  };

  const toggle = (value: string, values: string[], setter: (next: string[]) => void) => {
    setter(values.includes(value) ? values.filter((item) => item !== value) : [...values, value]);
  };

  const canStart = Boolean(asset) && providerKeys.length > 0 && privacyGroups.length > 0 && !uploading;
  const status = uploading
    ? "Opening your photo."
    : phase === "ready" && asset ? `Photo added: ${fileName}.`
      : phase === "analyzing" ? "Looking for private details. Big photos take longer."
        : phase === "verifying" ? "Saving and checking the file."
          : "";
  const chooseFile = () => fileInput.current?.click();
  const unavailable = analysis?.unavailable_providers ?? [];

  return (
    <div className="app-shell">
      <header className="rebate">
        <div className="brand">
          <BrandMark />
          <strong>ConsentGuard</strong>
        </div>
        {config && (
          <ol className="frames" aria-label="Progress">
            {steps.map((step, index) => {
              const state = index < progress ? "done" : index === progress ? "current" : "next";
              return (
                <li key={step} className={`frame ${state}`} aria-current={state === "current" ? "step" : undefined}>
                  <span className="frame-number">{state === "done" ? <Check weight="bold" aria-hidden="true" /> : index + 1}</span>
                  <span className="frame-label">{step}{state === "done" && <span className="visually-hidden"> (done)</span>}</span>
                </li>
              );
            })}
          </ol>
        )}
        <button className="quiet-button start-over" onClick={reset} disabled={!session}><Trash aria-hidden="true" /> Start over</button>
      </header>

      <main id="main-content" className="workspace">
        <p className="visually-hidden" role="status">{status}</p>
        {error && (
          <div className="alert alert-error" role="alert">
            <XCircle weight="fill" aria-hidden="true" />
            <p>{error}</p>
            {configFailed
              ? <button className="secondary-button compact" onClick={loadConfig}>Try again</button>
              : <button className="icon-button" onClick={() => setError("")} aria-label="Dismiss message"><X aria-hidden="true" /></button>}
          </div>
        )}
        {unavailable.length > 0 && phase !== "upload" && phase !== "ready" && (
          <div className="alert alert-warn" role="status">
            <Warning weight="fill" aria-hidden="true" />
            <p>
              {unavailable.length} detector{unavailable.length === 1 ? "" : "s"} didn't run ({unavailable.map(readable).join(", ")}), so those kinds of things weren't checked at all.
            </p>
          </div>
        )}

        {!config && !configFailed && (
          <section className="ingest loading-shell" aria-busy="true" aria-label="Loading">
            <div className="ingest-intro"><span className="skeleton-line wide" /><span className="skeleton-line wide" /><span className="skeleton-line" /><span className="skeleton-line short" /></div>
            <div className="ingest-stage"><div className="easel"><CropCorners /></div></div>
          </section>
        )}

        {config && onIngestScreen && (
          <section className="ingest">
            <div className="ingest-intro">
              <h1>{personal ? <>Erase the private parts. <span>Keep the photo.</span></> : <>Inspect the pixels. <span>Decide the boundary.</span></>}</h1>
              <p className="lede">{personal
                ? "Faces, license plates, text, handwriting, QR codes and hidden file data are found and covered in solid black. You can fix anything it misses."
                : "ConsentGuard combines local privacy detectors with a reviewer-approved mask. Originals never become public web assets."}</p>
            </div>

            <div className="ingest-stage">
              <div
                className={`easel ${dragging ? "dragging" : ""} ${asset ? "has-photo" : ""}`}
                onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false); }}
                onDrop={(event) => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files[0]; if (file) void upload(file); }}
                onClick={(event) => { if (!asset && !uploading && !(event.target as HTMLElement).closest("button")) chooseFile(); }}
              >
                <input ref={fileInput} id="photo-input" className="visually-hidden" type="file" aria-label="Choose a photo" accept="image/jpeg,image/png,image/webp" tabIndex={-1} onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); event.target.value = ""; }} />
                <CropCorners />
                {asset ? (
                  <img className="easel-photo" src={asset.normalized_url} alt={`Your photo, ${fileName}`} decoding="async" />
                ) : (
                  <div className="easel-empty">
                    {uploading ? <CircleNotch className="spin" aria-hidden="true" /> : <UploadSimple aria-hidden="true" />}
                    <h2>{uploading ? "Opening your photo" : "Drop a photo here"}</h2>
                    <p>or paste one with Ctrl+V</p>
                    <button className="secondary-button" onClick={chooseFile} disabled={uploading}>Choose photo</button>
                    <small>JPEG, PNG, or WebP, up to {Math.round(config.upload_max_bytes / 1024 / 1024)} MB</small>
                  </div>
                )}
              </div>

              {asset && (
                <div className="photo-meta">
                  <div>
                    <strong title={fileName}>{fileName}</strong>
                    <span>
                      <span className="data">{asset.width} × {asset.height}</span> {asset.source_format}
                      {personal
                        ? (asset.metadata_categories.length ? ` · hidden file data found (${asset.metadata_categories.join(", ")}), will be removed` : " · no hidden file data")
                        : <> · SHA-256 <code>{asset.pixel_sha256.slice(0, 12)}</code></>}
                    </span>
                  </div>
                  <button className="quiet-button" onClick={chooseFile} disabled={uploading}>{uploading ? "Opening…" : "Replace"}</button>
                </div>
              )}

              <div className="action-row">
                {personal ? (
                  <>
                    <button className="primary-button" onClick={eraseAndSave} disabled={!canStart}>Erase and save <ArrowRight aria-hidden="true" /></button>
                    <button className="secondary-button" onClick={analyze} disabled={!canStart}>Check it myself first</button>
                  </>
                ) : (
                  <button className="primary-button" onClick={analyze} disabled={!canStart}>Run local analysis <ArrowRight aria-hidden="true" /></button>
                )}
              </div>
              {!privacyGroups.length && <p className="hint">Pick at least one thing to look for.</p>}
              <p className="reminder">
                <Warning weight="fill" aria-hidden="true" />
                {personal
                  ? <span><strong>It will miss things.</strong> You'll see the result before you download it. Nothing leaves this computer.</span>
                  : <span><strong>Research configuration.</strong> Every result requires human review. Download stays blocked unless consent, release profile, and independent assurance checks all pass.</span>}
              </p>
            </div>

            <div className="ingest-options">
              <fieldset className="looks-for">
                <legend>{personal ? "It looks for" : "Privacy regions"}</legend>
                <div className="chip-row">
                  {config.privacy_groups.map((group) => {
                    const weak = (reliability[group] ?? "reliable") !== "reliable";
                    return (
                      <label key={group} className={`chip ${weak ? "weak" : ""}`}>
                        <input type="checkbox" checked={privacyGroups.includes(group)} onChange={() => toggle(group, privacyGroups, setPrivacyGroups)} />
                        <span>{group}{weak && <><Warning weight="fill" aria-hidden="true" /><span className="visually-hidden"> (hit and miss, check yourself)</span></>}</span>
                      </label>
                    );
                  })}
                </div>
                {unreliableSelected.length > 0 && (
                  <p className="chip-note"><Warning weight="fill" aria-hidden="true" /> Marked items are hit and miss. Check those yourself.</p>
                )}
              </fieldset>

              <button className="disclosure" onClick={() => setAdvanced((value) => !value)} aria-expanded={advanced} aria-controls="detector-settings">
                <SlidersHorizontal aria-hidden="true" /> Detectors <span className="disclosure-count">{providerKeys.length} of {config.providers.length} on</span>
              </button>
              {advanced && (
                <fieldset className="detector-list" id="detector-settings">
                  <legend className="visually-hidden">Detectors</legend>
                  {config.providers.map((provider) => (
                    <label key={provider.key} className={!provider.available ? "unavailable" : ""}>
                      <input type="checkbox" checked={providerKeys.includes(provider.key)} disabled={!provider.available} onChange={() => toggle(provider.key, providerKeys, setProviderKeys)} />
                      <span>{provider.label}</span>
                      <small>{provider.available ? "Ready" : "Not installed"}</small>
                    </label>
                  ))}
                </fieldset>
              )}
            </div>
          </section>
        )}

        {phase === "analyzing" && asset && (
          <section className="scanning">
            <div className="scan-frame">
              <img src={asset.normalized_url} alt="" decoding="async" />
              <span className="scan-line" aria-hidden="true" />
            </div>
            <div className="scan-copy">
              <h1>{personal ? "Looking for private details" : "Mapping privacy evidence"}</h1>
              <p>Checking the whole photo, then a closer second pass over small areas. Big photos take longer. The photo stays on this computer.</p>
              <ul className="chip-row static" aria-label="Looking for">
                {privacyGroups.map((group) => <li key={group} className="chip"><span>{group}</span></li>)}
              </ul>
            </div>
          </section>
        )}

        {(phase === "review" || phase === "verifying" || phase === "result") && analysis && session && (
          <section className="review">
            <div className="review-head">
              <h1 ref={pageHeading} tabIndex={-1}>
                {phase === "result" ? (personal ? "Your cleaned-up photo" : "Verification result") : (personal ? "Check what gets covered" : "Correct the redaction boundary")}
              </h1>
              <p>{phase === "result"
                ? "This is the new file, exactly as it will download."
                : "Hatched areas turn solid black in the saved photo. Paint over anything it missed; erase anything that doesn't need hiding."}</p>
              {!personal && <span className={`profile-status ${analysis.threshold_profile_release_ready ? "pass" : "warn"}`}>{analysis.threshold_profile_release_ready ? "Release profile" : "Research profile"}</span>}
            </div>

            <div className="review-main">
              {phase === "result" && result ? (
                <figure className="print-table">
                  <img className="print" src={`${result.rendered_url}?v=${result.decision.decision_digest}`} alt="Your photo with the private areas covered in black" decoding="async" />
                  <figcaption>
                    New file <span className="data">{analysis.width} × {analysis.height}</span>, written fresh with location and camera data removed
                    {!personal && <> · <code>{String(result.export_report.output_sha256 || "").slice(0, 16)}</code></>}
                  </figcaption>
                </figure>
              ) : (
                <Suspense fallback={<div className="editor editor-loading"><span className="skeleton-bar" />Loading the editing tools</div>}>
                  <MaskEditor ref={editorRef} sourceUrl={analysis.normalized_url} maskUrl={analysis.initial_mask_url} maskOverlayUrl={analysis.mask_overlay_url} overlayUrl={analysis.overlay_url} imageWidth={analysis.width} imageHeight={analysis.height} />
                </Suspense>
              )}
            </div>

            <aside className="inspector" aria-label={phase === "result" ? "Result" : "Review"}>
              {phase !== "result" ? (
                <>
                  <section className="panel">
                    <h2 className="panel-title">
                      {analysis.candidates.length ? `Found ${analysis.candidates.length} area${analysis.candidates.length === 1 ? "" : "s"}` : "Nothing found"}
                    </h2>
                    {!personal && (
                      <dl className="metrics">
                        <div><dt>Raw evidence</dt><dd className="data">{analysis.raw_evidence_count}</dd></div>
                        <div><dt>Providers run</dt><dd className="data">{analysis.selected_provider_keys.length}</dd></div>
                        <div><dt>Unavailable</dt><dd className={`data ${unavailable.length ? "warn-text" : ""}`}>{unavailable.length}</dd></div>
                      </dl>
                    )}
                    {analysis.candidates.length > 0 ? (
                      <ul className="found-list">
                        {analysis.candidates.slice(0, 8).map((candidate) => (
                          <li key={candidate.candidate_id}>
                            <span className="cover-swatch" aria-hidden="true" />
                            <div>
                              <strong>{[...new Set(candidate.privacy_classes.map(classLabel))].join(", ")}</strong>
                              <small>{personal ? percentOf(candidate.mask_pixels, analysis.width, analysis.height) : candidate.providers.join(" + ")}</small>
                            </div>
                          </li>
                        ))}
                        {analysis.candidates.length > 8 && <li className="more">and {analysis.candidates.length - 8} more</li>}
                      </ul>
                    ) : (
                      <p className="panel-note">Nothing matched what you asked it to look for. Look over the whole photo yourself before you share it.</p>
                    )}
                    {(analysis.unreliable_groups?.length ?? 0) > 0 && (
                      <p className="inline-warn"><Warning weight="fill" aria-hidden="true" /> Hit and miss here: {analysis.unreliable_groups?.join(", ")}. Look for these yourself.</p>
                    )}
                  </section>

                  {personal ? (
                    <section className="panel">
                      <h2 className="panel-title">Fixing the cover</h2>
                      <ul className="how-to">
                        <li><kbd>B</kbd><span><strong>Brush</strong> paints over anything it missed</span></li>
                        <li><kbd>E</kbd><span><strong>Eraser</strong> uncovers what doesn't need hiding</span></li>
                        <li><kbd>L</kbd><span><strong>Loupe</strong> magnifies the edges up close</span></li>
                        <li><kbd>Ctrl Z</kbd><span>undoes the last stroke</span></li>
                      </ul>
                    </section>
                  ) : (
                    <section className="panel consent-form">
                      <h2 className="panel-title">Consent assertion <LockKey aria-hidden="true" /></h2>
                      <label>Consent state<select ref={consentInput} value={consentState} onChange={(event) => { setConsentState(event.target.value as ConsentState); setReviewError(""); }}><option value="UNKNOWN">Unknown</option><option value="PENDING">Pending</option><option value="GRANTED">Granted</option><option value="DENIED">Denied</option><option value="REVOKED">Revoked</option><option value="EXPIRED">Expired</option></select></label>
                      {consentState !== "GRANTED" && <p className="inline-warn"><Warning weight="fill" aria-hidden="true" /> Only an explicit, current grant can proceed to verification.</p>}
                      <label>Subject reference<input ref={subjectInput} value={subjectRef} onChange={(event) => { setSubjectRef(event.target.value); setReviewError(""); }} placeholder="Non-PII alias, e.g. subject-01" /></label>
                      <div className="field-pair">
                        <label>Operation<select value={operation} onChange={(event) => setOperation(event.target.value)}><option value="share">Share</option><option value="publish">Publish</option><option value="archive">Archive</option></select></label>
                        <label>Audience<input ref={audienceInput} value={audience} onChange={(event) => { setAudience(event.target.value); setReviewError(""); }} placeholder="Project team" /></label>
                      </div>
                      <label><span className="field-label">Purpose <small>Required</small></span><textarea ref={purposeInput} value={purpose} onChange={(event) => { setPurpose(event.target.value); setReviewError(""); }} rows={2} placeholder="Why this image needs to be released" aria-invalid={Boolean(reviewError && !purpose.trim())} aria-describedby={reviewError ? "review-error" : undefined} /></label>
                      <label className="review-check"><input ref={reviewInput} type="checkbox" checked={reviewCompleted} onChange={(event) => { setReviewCompleted(event.target.checked); setReviewError(""); }} /><span><strong>I inspected the full image</strong><small>I approve the visible redaction boundary for this share context.</small></span></label>
                      {reviewError && <p className="form-error" id="review-error" role="alert"><Warning weight="fill" aria-hidden="true" /> {reviewError}</p>}
                    </section>
                  )}
                </>
              ) : result && (
                <>
                  <section className={`panel verdict ${result.export_available ? "pass" : "blocked"}`}>
                    <span className="verdict-icon">{result.export_available ? <CheckCircle weight="fill" aria-hidden="true" /> : <LockKey weight="fill" aria-hidden="true" />}</span>
                    <div>
                      <h2>{decisionTitle(result.decision.action)}</h2>
                      <p>{result.export_available
                        ? (personal ? "Written fresh, re-opened, and checked. The checks are listed below." : "Required checks passed for this prototype configuration. Residual limitations remain.")
                        : "The preview exists, but the download stays locked. The checks below say why."}</p>
                    </div>
                  </section>

                  <div className="download-group">
                    <p className="reminder">
                      <Warning weight="fill" aria-hidden="true" />
                      <span><strong>Look over the whole photo before you share it.</strong> Detection can miss things, and you're the last check.</span>
                    </p>
                    {result.export_available
                      ? <a className="primary-button wide" href={`/v1/sessions/${session.session_id}/export`} download={result.export_filename ?? "consentguard-redacted.png"}><DownloadSimple aria-hidden="true" /> {personal ? "Download the clean image" : "Download sanitized export"}</a>
                      : <button className="primary-button wide" disabled><LockKey aria-hidden="true" /> Download blocked</button>}
                    <button className="secondary-button wide" onClick={() => setPhase("review")}>{personal ? "Fix the cover myself" : "Return to mask review"}</button>
                  </div>

                  {(result.warnings?.length ?? 0) > 0 && (
                    <section className="panel">
                      <h2 className="panel-title">Worth knowing</h2>
                      {result.warnings?.map((code) => <p key={code} className="inline-warn"><Warning weight="fill" aria-hidden="true" /> {warningMessage(code)}</p>)}
                    </section>
                  )}

                  <section className="panel">
                    <h2 className="panel-title">Checks on the saved file</h2>
                    <ul className="check-list">
                      {result.assurance_checks.map((check) => {
                        const copy = checkCopy(check);
                        return (
                          <li key={check.name} className={check.status.toLowerCase()}>
                            {statusIcon(check.status)}
                            <div>
                              <strong>{copy.title}</strong>
                              <small>{copy.detail}</small>
                              {!personal && <code>{check.reason_code}</code>}
                            </div>
                            <span className="status-word">{STATUS_WORD[check.status] ?? readable(check.status)}</span>
                          </li>
                        );
                      })}
                    </ul>
                  </section>

                  {!personal && result.decision.reason_codes.length > 0 && (
                    <section className="panel reasons">
                      <h2 className="panel-title">Decision reasons</h2>
                      {result.decision.reason_codes.map((reason) => <code key={reason}>{reason}</code>)}
                    </section>
                  )}
                  {personal && <button className="quiet-button delete-photo" onClick={reset}><Trash aria-hidden="true" /> Delete this photo from the app</button>}
                </>
              )}
            </aside>
            {phase !== "result" && (
              <div className="save-bar">
                {reviewError && personal && <p className="form-error" role="alert"><Warning weight="fill" aria-hidden="true" /> {reviewError}</p>}
                <button className="primary-button wide" onClick={render} disabled={phase === "verifying"}>
                  {phase === "verifying"
                    ? <><CircleNotch className="spin" aria-hidden="true" /> {personal ? "Saving and checking the file" : "Running assurance checks"}</>
                    : <>{personal ? "Save my version" : "Render and verify"} <ArrowRight aria-hidden="true" /></>}
                </button>
              </div>
            )}
          </section>
        )}
      </main>
      <footer>
        <span>ConsentGuard{config && !personal ? " research prototype" : ""}</span>
        <span>Runs on this computer only · no tracking · your photo is deleted when you close this tab</span>
      </footer>
    </div>
  );
}
