"""FastAPI application for the ConsentGuard reviewer frontend."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from consentguard.stage_04_fusion_calibration.domain import ConsentState
from consentguard.stage_05_review_export.api.models import AnalyzeRequest, HealthResponse
from consentguard.stage_05_review_export.api.sessions import ReviewSessionManager


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail="Session not found")
    if isinstance(error, TimeoutError):
        return HTTPException(status_code=410, detail="Session expired")
    if isinstance(error, PermissionError):
        return HTTPException(status_code=403, detail=str(error))
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail="Asset not found")
    return HTTPException(status_code=422, detail=str(error))


def _private_file(path: Path, *, filename: str | None = None) -> FileResponse:
    return FileResponse(
        path,
        filename=filename,
        headers={"Cache-Control": "no-store, max-age=0", "X-Content-Type-Options": "nosniff"},
    )


def create_app(
    providers: dict[str, object],
    thresholds,
    *,
    provider_labels: dict[str, str] | None = None,
    privacy_groups: dict[str, set[str]] | None = None,
    session_root: str | Path | None = None,
    frontend_dist: str | Path | None = None,
    ttl_seconds: int = 3600,
) -> FastAPI:
    root = Path(session_root) if session_root is not None else Path(tempfile.mkdtemp(prefix="consentguard-web-"))
    manager = ReviewSessionManager(
        root,
        providers,
        thresholds,
        provider_labels=provider_labels,
        privacy_groups=privacy_groups,
        ttl_seconds=ttl_seconds,
    )
    app = FastAPI(
        title="ConsentGuard reviewer API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.review_sessions = manager

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", configured_providers=len(providers))

    @app.get("/v1/config")
    def config():
        return manager.config()

    @app.post("/v1/sessions", status_code=201)
    def create_session():
        return manager.create()

    @app.post("/v1/sessions/{session_id}/assets", status_code=201)
    def upload_asset(session_id: str, asset: UploadFile = File(...)):
        try:
            payload = asset.file.read(manager.ingest_limits.max_bytes + 1)
            suffix = Path(asset.filename or "upload.bin").suffix or ".bin"
            return manager.stage_asset(session_id, payload, suffix)
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/v1/sessions/{session_id}/analyze")
    def analyze(session_id: str, request: AnalyzeRequest):
        try:
            return manager.analyze(session_id, request.provider_keys, request.privacy_groups)
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/v1/sessions/{session_id}/assets/normalized")
    def normalized_asset(session_id: str):
        try:
            return _private_file(manager.asset_path(session_id, "normalized"))
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/v1/sessions/{session_id}/assets/overlay")
    def overlay_asset(session_id: str):
        try:
            return _private_file(manager.asset_path(session_id, "overlay"))
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/v1/sessions/{session_id}/assets/mask-overlay")
    def mask_overlay_asset(session_id: str):
        try:
            return _private_file(manager.asset_path(session_id, "mask-overlay"))
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/v1/sessions/{session_id}/masks/initial")
    def initial_mask(session_id: str):
        try:
            return _private_file(manager.asset_path(session_id, "initial-mask"))
        except Exception as error:
            raise _http_error(error) from error

    @app.post("/v1/sessions/{session_id}/render")
    def render(
        session_id: str,
        mask: UploadFile = File(...),
        consent_state: ConsentState = Form(...),
        subject_ref: str = Form(...),
        operation: str = Form(...),
        audience: str = Form(...),
        purpose: str = Form(...),
        review_completed: bool = Form(...),
    ):
        try:
            payload = mask.file.read(manager.ingest_limits.max_bytes + 1)
            return manager.render(
                session_id,
                payload,
                consent_state=consent_state,
                subject_ref=subject_ref,
                operation=operation,
                audience=audience,
                purpose=purpose,
                review_completed=review_completed,
            )
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/v1/sessions/{session_id}/assets/rendered")
    def rendered_asset(session_id: str):
        try:
            return _private_file(manager.asset_path(session_id, "rendered"))
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/v1/sessions/{session_id}/export")
    def download_export(session_id: str):
        try:
            return _private_file(manager.export_path(session_id), filename="consentguard-redacted.png")
        except Exception as error:
            raise _http_error(error) from error

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        try:
            manager.delete(session_id)
            return Response(status_code=204)
        except Exception as error:
            raise _http_error(error) from error

    if frontend_dist is not None:
        dist = Path(frontend_dist)
        if dist.is_dir():
            app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
