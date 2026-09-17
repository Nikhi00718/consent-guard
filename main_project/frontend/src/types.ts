export type ProviderOption = {
  key: string;
  label: string;
  available: boolean;
};

export type AppConfig = {
  providers: ProviderOption[];
  privacy_groups: string[];
  upload_max_bytes: number;
  upload_max_pixels: number;
  session_ttl_seconds: number;
};

export type Session = {
  session_id: string;
  created_at: number;
  expires_at: number;
};

export type Asset = {
  width: number;
  height: number;
  source_format: string;
  source_sha256: string;
  pixel_sha256: string;
  metadata_categories: string[];
  orientation_applied: boolean;
  normalized_url: string;
};

export type Candidate = {
  candidate_id: string;
  privacy_classes: string[];
  providers: string[];
  uncertainty_flags: string[];
  mandatory_review: boolean;
  mask_pixels: number;
};

export type Analysis = {
  width: number;
  height: number;
  raw_evidence_count: number;
  candidates: Candidate[];
  selected_provider_keys: string[];
  selected_privacy_groups: string[];
  unavailable_providers: string[];
  provider_errors: Record<string, string>;
  threshold_profile_id: string;
  threshold_profile_release_ready: boolean;
  normalized_url: string;
  initial_mask_url: string;
  mask_overlay_url: string;
  overlay_url: string;
};

export type AssuranceCheck = {
  name: string;
  status: "PASS" | "FAIL" | "UNCERTAIN" | "NOT_RUN";
  reason_code: string;
  details: Record<string, unknown>;
};

export type ReleaseDecision = {
  action: string;
  reason_codes: string[];
  policy_version: string;
  review_required: boolean;
  export_allowed: boolean;
  decision_digest: string;
};

export type RenderResult = {
  assurance_status: AssuranceCheck["status"];
  assurance_checks: AssuranceCheck[];
  decision: ReleaseDecision;
  export_report: Record<string, unknown>;
  rendered_url: string;
  export_available: boolean;
};

export type ConsentState = "UNKNOWN" | "PENDING" | "GRANTED" | "DENIED" | "REVOKED" | "EXPIRED";
