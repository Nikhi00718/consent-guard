import type { Analysis, AppConfig, Asset, ConsentState, RenderResult, Session } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail || detail;
    } catch {
      // Keep the status-based fallback for non-JSON failures.
    }
    throw new Error(detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  config: () => request<AppConfig>("/v1/config"),
  createSession: () => request<Session>("/v1/sessions", { method: "POST" }),
  upload: (sessionId: string, file: File) => {
    const form = new FormData();
    form.append("asset", file);
    return request<Asset>(`/v1/sessions/${sessionId}/assets`, { method: "POST", body: form });
  },
  analyze: (sessionId: string, providerKeys: string[], privacyGroups: string[]) =>
    request<Analysis>(`/v1/sessions/${sessionId}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider_keys: providerKeys, privacy_groups: privacyGroups }),
    }),
  render: (
    sessionId: string,
    mask: Blob,
    review: {
      consentState: ConsentState;
      subjectRef: string;
      operation: string;
      audience: string;
      purpose: string;
      reviewCompleted: boolean;
    },
  ) => {
    const form = new FormData();
    form.append("mask", mask, "approved-mask.png");
    form.append("consent_state", review.consentState);
    form.append("subject_ref", review.subjectRef);
    form.append("operation", review.operation);
    form.append("audience", review.audience);
    form.append("purpose", review.purpose);
    form.append("review_completed", String(review.reviewCompleted));
    return request<RenderResult>(`/v1/sessions/${sessionId}/render`, { method: "POST", body: form });
  },
  deleteSession: (sessionId: string) => request<void>(`/v1/sessions/${sessionId}`, { method: "DELETE" }),
};
