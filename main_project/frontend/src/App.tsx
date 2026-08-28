import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Check, CheckCircle, CircleNotch, DownloadSimple, FileImage, Fingerprint, Info, LockKey, Pulse, SlidersHorizontal, Trash, UploadSimple, Warning, XCircle } from "@phosphor-icons/react";
import { api } from "./api";
import type { MaskEditorHandle } from "./MaskEditor";
import type { Analysis, AppConfig, Asset, ConsentState, RenderResult, Session } from "./types";

type Phase = "upload" | "ready" | "analyzing" | "review" | "verifying" | "result";

type ReviewFields = {
  consentState: ConsentState;
  subjectRef: string;
  audience: string;
  purpose: string;
  reviewCompleted: boolean;
};

const steps = ["Ingest", "Detect", "Review", "Verify"];
const MaskEditor = lazy(() => import("./MaskEditor"));

function readable(value: string): string {
  return value.toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function statusIcon(status: string) {
  if (status === "PASS") return <CheckCircle weight="fill" />;
  if (status === "FAIL") return <XCircle weight="fill" />;
  return <Warning weight="fill" />;
}

function activeStep(phase: Phase): number {
  if (phase === "upload" || phase === "ready") return 0;
  if (phase === "analyzing") return 1;
  if (phase === "review") return 2;
  return 3;
}

export function reviewValidationMessage(fields: ReviewFields): string {
  if (fields.consentState !== "GRANTED") return "Set Consent state to Granted before verification.";
  if (!fields.subjectRef.trim()) return "Add a non-PII subject reference before verification.";
  if (!fields.audience.trim()) return "Add the intended audience before verification.";
  if (!fields.purpose.trim()) return "Add a release purpose before verification.";
  if (!fields.reviewCompleted) return "Confirm that you inspected the full image before verification.";
  return "";
}

export default function App() {
  const editorRef = useRef<MaskEditorHandle>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const consentInput = useRef<HTMLSelectElement>(null);
  const subjectInput = useRef<HTMLInputElement>(null);
  const audienceInput = useRef<HTMLInputElement>(null);
  const purposeInput = useRef<HTMLTextAreaElement>(null);
  const reviewInput = useRef<HTMLInputElement>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [asset, setAsset] = useState<Asset | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [result, setResult] = useState<RenderResult | null>(null);
  const [fileName, setFileName] = useState("");
  const [phase, setPhase] = useState<Phase>("upload");
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

  useEffect(() => {
    api.config()
      .then((next) => {
        setConfig(next);
        setProviderKeys(next.providers.filter((provider) => provider.available).map((provider) => provider.key));
        setPrivacyGroups(next.privacy_groups);
      })
      .catch((reason: Error) => setError(`Could not connect to the local reviewer API. ${reason.message}`));
  }, []);

  useEffect(() => {
    if (phase === "result") resultHeading.current?.focus();
  }, [phase]);

  const progress = activeStep(phase);
  const selectedProviders = useMemo(() => config?.providers.filter((provider) => providerKeys.includes(provider.key)) ?? [], [config, providerKeys]);

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
      setError("Choose a JPEG, PNG, or WebP still image.");
      return;
    }
    setError("");
    setPhase("ready");
    try {
      if (session) await api.deleteSession(session.session_id).catch(() => undefined);
      const nextSession = await api.createSession();
      const nextAsset = await api.upload(nextSession.session_id, file);
      setSession(nextSession);
      setAsset(nextAsset);
      setFileName(file.name);
    } catch (reason) {
      setPhase("upload");
      setError(reason instanceof Error ? reason.message : "The image could not be ingested.");
    }
  };

  const analyze = async () => {
    if (!session || !providerKeys.length) return;
    setPhase("analyzing");
    setError("");
    try {
      const next = await api.analyze(session.session_id, providerKeys, privacyGroups);
      setAnalysis(next);
      setPhase("review");
    } catch (reason) {
      setPhase("ready");
      setError(reason instanceof Error ? reason.message : "Analysis failed.");
    }
  };

  const render = async () => {
    if (!session) return;
    if (!editorRef.current) {
      setReviewError("Review tools are still loading. Try again in a moment.");
      return;
    }
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
    setReviewError("");
    setPhase("verifying");
    setError("");
    try {
      const mask = await editorRef.current.exportMask();
      const next = await api.render(session.session_id, mask, {
        consentState,
        subjectRef,
        operation,
        audience,
        purpose,
        reviewCompleted,
      });
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

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div><strong>ConsentGuard</strong><small>review workspace</small></div>
        </div>
        <div className="runtime-badge"><Pulse weight="fill" /> local runtime <span>127.0.0.1</span></div>
        <button className="icon-text-button" onClick={reset} disabled={!session}><Trash /> Clear session</button>
      </header>

      <nav className="workflow-nav" aria-label="Review progress">
        {steps.map((step, index) => (
          <div key={step} className={`workflow-step ${index === progress ? "active" : ""} ${index < progress ? "complete" : ""}`}>
            <span>{index < progress ? <Check /> : String(index + 1).padStart(2, "0")}</span>
            <p>{step}</p>
          </div>
        ))}
      </nav>

      <main id="main-content" className="workspace">
        <div className="research-notice"><Info weight="fill" /><p><strong>Research configuration.</strong> Every result requires human review. Download stays blocked unless consent, release profile, and independent assurance checks all pass.</p></div>
        {error && <div className="error-banner" role="alert"><XCircle weight="fill" /><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}

        {(phase === "upload" || phase === "ready") && (
          <section className="ingest-layout">
            <div className="ingest-copy">
              <span className="eyebrow">human-controlled release</span>
              <h1>Inspect the pixels.<br /><em>Decide the boundary.</em></h1>
              <p>ConsentGuard combines local privacy detectors with a reviewer-approved mask. Originals never become public web assets.</p>
              <div className="constraint-row"><span><LockKey /> Local processing</span><span><Fingerprint /> Metadata removed</span></div>
            </div>
            <div className="ingest-panel">
              <div
                className={`dropzone ${dragging ? "dragging" : ""} ${asset ? "has-file" : ""}`}
                onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files[0]; if (file) void upload(file); }}
              >
                <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(file); }} />
                {asset ? (
                  <>
                    <div className="file-glyph"><FileImage weight="duotone" /></div>
                    <div className="file-info"><strong>{fileName}</strong><span>{asset.width} × {asset.height} · {asset.source_format}</span></div>
                    <button className="text-button" onClick={() => fileInput.current?.click()}>Replace</button>
                  </>
                ) : (
                  <>
                    <div className="upload-glyph"><UploadSimple /></div>
                    <h2>Drop a still image here</h2>
                    <p>JPEG, PNG, or WebP · up to {config ? Math.round(config.upload_max_bytes / 1024 / 1024) : 25} MB</p>
                    <button className="secondary-button" onClick={() => fileInput.current?.click()}>Choose image</button>
                  </>
                )}
              </div>

              <button className="advanced-toggle" onClick={() => setAdvanced((value) => !value)} aria-expanded={advanced}><SlidersHorizontal /> Analysis controls <span>{advanced ? "−" : "+"}</span></button>
              {advanced && config && (
                <div className="advanced-panel">
                  <fieldset><legend>Model providers</legend>{config.providers.map((provider) => <label key={provider.key} className={!provider.available ? "disabled" : ""}><input type="checkbox" checked={providerKeys.includes(provider.key)} disabled={!provider.available} onChange={() => toggle(provider.key, providerKeys, setProviderKeys)} /><span>{provider.label}</span><small>{provider.available ? "ready" : "unavailable"}</small></label>)}</fieldset>
                  <fieldset><legend>Privacy regions</legend><div className="chip-grid">{config.privacy_groups.map((group) => <label key={group} className="filter-chip"><input type="checkbox" checked={privacyGroups.includes(group)} onChange={() => toggle(group, privacyGroups, setPrivacyGroups)} /><span>{group}</span></label>)}</div></fieldset>
                </div>
              )}
              <button className="primary-button analyze-button" onClick={analyze} disabled={!asset || !providerKeys.length}>Run local analysis <ArrowRight /></button>
              {asset && <p className="asset-note">SHA-256 <code>{asset.pixel_sha256.slice(0, 18)}…</code>{asset.metadata_categories.length ? ` · ${asset.metadata_categories.length} metadata categor${asset.metadata_categories.length === 1 ? "y" : "ies"} detected` : " · no listed metadata"}</p>}
            </div>
          </section>
        )}

        {phase === "analyzing" && (
          <section className="analysis-state">
            <div className="radar"><span /><span /><span /><Fingerprint weight="duotone" /></div>
            <span className="eyebrow">sequential provider run</span>
            <h1>Mapping privacy evidence</h1>
            <p>GPU providers run one at a time. The source remains inside this session.</p>
            <div className="provider-queue">{selectedProviders.map((provider, index) => <div key={provider.key} style={{ animationDelay: `${index * 90}ms` }}><CircleNotch className="spin" /><span>{provider.label}</span><small>queued</small></div>)}</div>
          </section>
        )}

        {(phase === "review" || phase === "verifying" || phase === "result") && analysis && session && (
          <section className="review-layout">
            <div className="review-main">
              <div className="section-heading"><div><span className="eyebrow">native-resolution review</span><h1 ref={resultHeading} tabIndex={phase === "result" ? -1 : undefined}>{phase === "result" ? "Verification result" : "Correct the redaction boundary"}</h1></div><div className={`profile-status ${analysis.threshold_profile_release_ready ? "pass" : "warning"}`}><span />{analysis.threshold_profile_release_ready ? "Release profile" : "Research profile"}</div></div>
              {phase === "result" && result ? (
                <div className="rendered-frame"><img src={`${result.rendered_url}?v=${result.decision.decision_digest}`} alt="Newly encoded image with the approved redaction mask applied" /><div className="rendered-label"><span>newly encoded output</span><code>{String(result.export_report.output_sha256 || "").slice(0, 18)}…</code></div></div>
              ) : (
                <Suspense fallback={<div className="editor-shell editor-loading"><span />Loading review tools</div>}>
                  <MaskEditor ref={editorRef} sourceUrl={analysis.normalized_url} maskUrl={analysis.initial_mask_url} maskOverlayUrl={analysis.mask_overlay_url} overlayUrl={analysis.overlay_url} imageWidth={analysis.width} imageHeight={analysis.height} />
                </Suspense>
              )}
            </div>

            <aside className="inspector">
              {phase !== "result" ? (
                <>
                  <section className="inspector-section evidence-summary">
                    <div className="inspector-title"><span>Evidence summary</span><strong>{analysis.candidates.length.toString().padStart(2, "0")}</strong></div>
                    <div className="metric-row"><span>Raw evidence</span><code>{analysis.raw_evidence_count}</code></div>
                    <div className="metric-row"><span>Providers run</span><code>{analysis.selected_provider_keys.length}</code></div>
                    <div className="metric-row"><span>Unavailable</span><code className={analysis.unavailable_providers.length ? "warning-text" : ""}>{analysis.unavailable_providers.length}</code></div>
                    <div className="candidate-list">{analysis.candidates.slice(0, 6).map((candidate) => <article key={candidate.candidate_id}><span className="candidate-mark" /><div><strong>{candidate.privacy_classes.map(readable).join(", ")}</strong><small>{candidate.providers.join(" + ")}</small></div><code>{candidate.mask_pixels.toLocaleString()}</code></article>)}{!analysis.candidates.length && <p className="empty-evidence">No regions met the selected thresholds. The reviewer must still inspect the full image.</p>}</div>
                  </section>

                  <section className="inspector-section consent-section">
                    <div className="inspector-title"><span>Consent assertion</span><LockKey /></div>
                    <label>Consent state<select ref={consentInput} value={consentState} onChange={(event) => { setConsentState(event.target.value as ConsentState); setReviewError(""); }}><option value="UNKNOWN">Unknown</option><option value="PENDING">Pending</option><option value="GRANTED">Granted</option><option value="DENIED">Denied</option><option value="REVOKED">Revoked</option><option value="EXPIRED">Expired</option></select></label>
                    {consentState !== "GRANTED" && <p className="inline-warning"><Warning weight="fill" /> Only an explicit, current grant can proceed to verification.</p>}
                    <label>Subject reference<input ref={subjectInput} value={subjectRef} onChange={(event) => { setSubjectRef(event.target.value); setReviewError(""); }} placeholder="Non-PII alias, e.g. subject-01" /></label>
                    <div className="field-pair"><label>Operation<select value={operation} onChange={(event) => setOperation(event.target.value)}><option value="share">Share</option><option value="publish">Publish</option><option value="archive">Archive</option></select></label><label>Audience<input ref={audienceInput} value={audience} onChange={(event) => { setAudience(event.target.value); setReviewError(""); }} placeholder="Project team" /></label></div>
                    <label><span className="field-label">Purpose <small>Required</small></span><textarea ref={purposeInput} value={purpose} onChange={(event) => { setPurpose(event.target.value); setReviewError(""); }} rows={2} placeholder="Why this image needs to be released" aria-invalid={Boolean(reviewError && !purpose.trim())} aria-describedby={reviewError ? "review-error" : undefined} /></label>
                    <label className="review-check"><input ref={reviewInput} type="checkbox" checked={reviewCompleted} onChange={(event) => { setReviewCompleted(event.target.checked); setReviewError(""); }} /><span><strong>I inspected the full image</strong><small>I approve the visible redaction boundary for this share context.</small></span></label>
                    {reviewError && <p className="review-error" id="review-error" role="alert"><Warning weight="fill" /> {reviewError}</p>}
                  </section>
                  <button className="primary-button verify-button" onClick={render} disabled={phase === "verifying"}>{phase === "verifying" ? <><CircleNotch className="spin" /> Running assurance checks</> : <>Render and verify <ArrowRight /></>}</button>
                </>
              ) : result && (
                <>
                  <section className={`decision-panel ${result.export_available ? "pass" : "blocked"}`}>
                    <span className="decision-icon">{result.export_available ? <CheckCircle weight="fill" /> : <LockKey weight="fill" />}</span>
                    <span className="eyebrow">release decision</span>
                    <h2>{readable(result.decision.action)}</h2>
                    <p>{result.export_available ? "Required checks passed for this prototype configuration. Residual limitations remain." : "The reviewed preview exists, but no export capability was issued."}</p>
                  </section>
                  <section className="inspector-section assurance-list">
                    <div className="inspector-title"><span>Assurance checks</span><code>{result.assurance_status}</code></div>
                    {result.assurance_checks.map((check) => <article key={check.name} className={check.status.toLowerCase()}>{statusIcon(check.status)}<div><strong>{readable(check.name)}</strong><small>{readable(check.reason_code)}</small></div><code>{check.status}</code></article>)}
                  </section>
                  {result.decision.reason_codes.length > 0 && <section className="reason-box"><strong>Decision reasons</strong>{result.decision.reason_codes.map((reason) => <code key={reason}>{reason}</code>)}</section>}
                  {result.export_available ? <a className="primary-button download-button" href={`/v1/sessions/${session.session_id}/export`} download><DownloadSimple /> Download sanitized export</a> : <button className="primary-button download-button" disabled><LockKey /> Download blocked</button>}
                  <button className="secondary-button full-width" onClick={() => setPhase("review")}>Return to mask review</button>
                </>
              )}
            </aside>
          </section>
        )}
      </main>
      <footer><span>ConsentGuard research prototype</span><span>Images stay session-local · no telemetry · automatic expiry</span></footer>
    </div>
  );
}
