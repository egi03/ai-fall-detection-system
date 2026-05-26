"""FastAPI app: upload a video, get back annotated MP4 + alarm timeline.

Run via ``python scripts/webapp.py`` (which calls :func:`create_app` and hands
it to uvicorn). Synchronous processing — POST /analyze blocks until the
pipeline finishes, then returns a JSON payload the page renders.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

import yaml
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.utils.logger import get_logger
from src.webapp.live import LiveSession
from src.webapp.pipeline import process_video

logger = get_logger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_JOBS_DIR = _PROJECT_ROOT / "logs" / "webapp_jobs"

# Allow generous uploads; cap at 200 MB to avoid filling disk on a typo.
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_ACCEPTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _load_config() -> dict:
    """Load config/config.yaml, falling back to demo defaults if missing."""
    cfg_path = _PROJECT_ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "video": {"target_fps": 15, "resolution": [640, 480]},
        "pose": {
            "backend": "mediapipe",
            "model_complexity": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "features": {"window_size": 30},
        "model": {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
    }


def _load_dotenv() -> None:
    """Tiny .env loader so GOOGLE_API_KEY is available without python-dotenv."""
    import os

    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def create_app() -> FastAPI:
    _load_dotenv()
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Fall Detection — Web Demo", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.mount("/jobs", StaticFiles(directory=str(_JOBS_DIR)), name="jobs")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html")

    @app.post("/analyze")
    async def analyze(
        video: UploadFile = File(...),
        use_vlm: Optional[str] = Form(None),
    ) -> JSONResponse:
        if not video.filename:
            raise HTTPException(status_code=400, detail="No filename")
        ext = Path(video.filename).suffix.lower()
        if ext not in _ACCEPTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported extension {ext!r}. Use one of {sorted(_ACCEPTED_EXTENSIONS)}.",
            )

        job_id = uuid.uuid4().hex[:12]
        job_dir = _JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        upload_path = job_dir / f"input{ext}"

        bytes_written = 0
        with open(upload_path, "wb") as fh:
            while chunk := await video.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > _MAX_UPLOAD_BYTES:
                    fh.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB cap.",
                    )
                fh.write(chunk)

        logger.info(f"Job {job_id}: received {bytes_written} bytes -> {upload_path}")

        annotated = job_dir / "annotated.mp4"
        try:
            result = process_video(
                video_path=upload_path,
                output_video_path=annotated,
                config=_load_config(),
                use_vlm=bool(use_vlm),
            )
        except Exception as exc:  # noqa: BLE001 - surface failure to the browser
            logger.exception(f"Job {job_id} failed")
            raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

        payload = result.to_json()
        payload["job_id"] = job_id
        payload["annotated_url"] = f"/jobs/{job_id}/annotated.mp4"
        payload["timeline_url"] = f"/jobs/{job_id}/timeline.json"

        with open(job_dir / "timeline.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        return JSONResponse(payload)

    @app.websocket("/ws/live")
    async def live_ws(websocket: WebSocket) -> None:
        """Streaming pipeline: browser sends JPEG frames, server returns state.

        Lifecycle:
          1. Accept the socket and build a :class:`LiveSession` (heavy: loads
             the BiLSTM ensemble + the pose backend).
          2. Send a one-shot ``config`` message with skeleton topology, FPS,
             and threshold so the client can render its overlay.
          3. Per inbound binary frame, run the pipeline on a thread (pose
             estimation and torch inference are blocking) and reply with the
             per-frame payload.
          4. On disconnect, release the pose handle.
        """
        await websocket.accept()
        session: Optional[LiveSession] = None
        try:
            try:
                session = await asyncio.to_thread(LiveSession, _load_config())
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to start LiveSession")
                await websocket.send_json(
                    {"type": "error", "detail": f"Could not start live pipeline: {exc}"}
                )
                await websocket.close()
                return

            await websocket.send_json(session.config_payload())

            while True:
                message = await websocket.receive()
                # Starlette gives us {"type": ..., "bytes": ...} or {"type": ..., "text": ...}
                if message.get("type") == "websocket.disconnect":
                    break

                jpeg_bytes = message.get("bytes")
                if jpeg_bytes is None:
                    # Allow text "ping" / control messages without crashing.
                    text = message.get("text")
                    if text == "ping":
                        await websocket.send_json({"type": "pong"})
                    continue

                payload = await asyncio.to_thread(session.process_frame, jpeg_bytes)
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            logger.info("Live WS client disconnected")
        except Exception:  # noqa: BLE001
            logger.exception("Live WS error")
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass
        finally:
            if session is not None:
                await asyncio.to_thread(session.release)

    return app


app = create_app()
