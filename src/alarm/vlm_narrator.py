"""Vision-language narration of confirmed fall events (Gemini backend).

When the alarm FSM transitions to ``FALL_CONFIRMED`` the demo can hand the
trigger frame and the last ~2 seconds of pose-feature trajectories to a
multimodal Gemini model and receive back a one-paragraph natural-language
description of what happened: location in scene, fall direction, peak
vertical velocity, post-fall stillness, etc.

Design constraints
------------------
* **Non-blocking.** API calls run on a background worker thread; the demo
  loop never stalls on network I/O.
* **Optional.** If ``GOOGLE_API_KEY`` is missing or the ``google-genai``
  SDK is not installed, the narrator constructor raises a clear error and
  the demo falls back to the standard alarm-only behaviour.
* **One alarm at a time.** The queue is depth-1 — only the most recent
  pending request is kept, so a rapidly-firing FSM cannot pile up jobs.

Notes
-----
Gemini's vision API accepts inline image bytes through ``types.Part``
parts. We JPEG-compress the frame at quality 70 to keep request payloads
small (~30-60 KB per snapshot), which makes round-trip latency
acceptable for a live demo (~1-2 s on Flash-Lite).
"""

import base64  # noqa: F401  (kept available for callers that need raw b64)
import os
import queue
import threading
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Default Gemini model. The user explicitly opted for Flash-Lite.
_DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Maximum number of feature rows used to build the trajectory summary.
_MAX_TRAJECTORY_ROWS = 30

# Soft wall-clock budget for the worker before logging a slow-request warning.
_SLOW_THRESHOLD_SEC = 5.0


def _encode_frame_jpeg(frame_bgr: np.ndarray, quality: int = 70) -> bytes:
    """JPEG-compress a BGR frame and return the raw byte string."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed when serializing fall snapshot")
    return buf.tobytes()


def _summarize_trajectory(features: np.ndarray) -> str:
    """Build a compact textual summary of recent feature trajectories.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(T, 15)`` — recent 15-D feature rows. Order must match
        :data:`src.explainability.joint_mapping.FEATURE_NAMES`.

    Returns
    -------
    str
        Short multi-line summary the model can read alongside the image.
    """
    if features.ndim != 2 or features.shape[0] == 0:
        return "(no feature trajectory available)"
    features = features[-_MAX_TRAJECTORY_ROWS:]

    def stat(col: int) -> tuple:
        x = features[:, col]
        x = x[~np.isnan(x)] if np.issubdtype(x.dtype, np.floating) else x
        if x.size == 0:
            return (float("nan"),) * 4
        return float(x[0]), float(x[-1]), float(np.max(x)), float(np.min(x))

    torso_first, torso_last, torso_max, _ = stat(0)
    vel_first, vel_last, vel_max, vel_min = stat(3)
    accel_first, accel_last, accel_max, accel_min = stat(4)
    ar_first, ar_last, ar_max, _ = stat(2)

    return (
        "Pose-feature trajectory over the last "
        f"{features.shape[0]} frames (approx 2 s at 15 FPS):\n"
        f"- Torso inclination from vertical: {torso_first:.0f}deg "
        f"-> {torso_last:.0f}deg (peak {torso_max:.0f}deg).\n"
        f"- Vertical CoM velocity (px/s, +ve = downward): "
        f"{vel_first:.0f} -> {vel_last:.0f} (range {vel_min:.0f}..{vel_max:.0f}).\n"
        f"- Vertical CoM acceleration (px/s^2): "
        f"{accel_first:.0f} -> {accel_last:.0f} (range {accel_min:.0f}..{accel_max:.0f}).\n"
        f"- Bounding-box aspect ratio (width/height, >1 = horizontal): "
        f"{ar_first:.2f} -> {ar_last:.2f} (peak {ar_max:.2f})."
    )


_SYSTEM_PROMPT = (
    "You are a concise medical-monitoring assistant. The user will give you "
    "a still frame captured the instant a fall-detection system confirmed a "
    "fall, plus a numerical summary of the subject's pose dynamics in the "
    "two seconds leading up to that frame. Respond in 1-2 sentences (40-60 "
    "words) describing where in the scene the subject is, the most likely "
    "fall direction, and whether the subject appears to be moving. Do not "
    "speculate about identity or medical condition. Output prose only — no "
    "bullet points, no headings, no preamble."
)


class VLMNarrator:
    """Threaded Gemini vision client for fall-event narration.

    Parameters
    ----------
    api_key : str, optional
        Google AI Studio API key. If ``None`` reads ``GOOGLE_API_KEY``.
    model : str, optional
        Gemini model id. Defaults to a Flash-Lite class vision-capable model.
    max_tokens : int
        Output token cap for the narration response.

    Raises
    ------
    RuntimeError
        If the ``google-genai`` package is missing or no API key is available.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 160,
    ) -> None:
        try:
            from google import genai  # noqa: F401  (presence check)
            from google.genai import types  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' package is not installed. "
                "Install it with: pip install google-genai"
            ) from exc

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Add it to your .env file or export "
                "it before enabling --vlm, or pass api_key explicitly."
            )

        from google import genai
        from google.genai import types

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=resolved_key)
        self._model = model or _DEFAULT_MODEL
        self._max_tokens = max_tokens

        # Depth-1 request queue: dropping older pending jobs is intentional
        self._request_q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        # Unbounded result queue (cleared lazily by poll())
        self._result_q: "queue.Queue[str]" = queue.Queue()
        self._stop_evt = threading.Event()

        self._worker = threading.Thread(
            target=self._worker_loop, name="vlm-narrator", daemon=True,
        )
        self._worker.start()

    @property
    def model_name(self) -> str:
        """Resolved Gemini model id used for narrations."""
        return self._model

    def request_narration(
        self,
        rgb_frame: np.ndarray,
        feature_history: np.ndarray,
        confidence: float,
    ) -> None:
        """Enqueue a non-blocking narration request.

        If a previous request is still queued, the older one is dropped
        — the demo only cares about the latest alarm.

        Parameters
        ----------
        rgb_frame : np.ndarray
            BGR frame captured at FALL_CONFIRMED. Will be JPEG-compressed.
        feature_history : np.ndarray
            Recent feature rows; see :func:`_summarize_trajectory`.
        confidence : float
            Smoothed alarm confidence at trigger time. Embedded into the
            prompt context so the model can mention "high-confidence" etc.
        """
        try:
            self._request_q.get_nowait()  # drop stale
        except queue.Empty:
            pass
        self._request_q.put_nowait((rgb_frame.copy(), feature_history.copy(), float(confidence)))

    def poll(self) -> Optional[str]:
        """Return the most recent finished narration, or ``None``."""
        latest: Optional[str] = None
        try:
            while True:
                latest = self._result_q.get_nowait()
        except queue.Empty:
            pass
        return latest

    def shutdown(self) -> None:
        """Signal the worker thread to exit and wait for it briefly."""
        self._stop_evt.set()
        try:
            self._request_q.put_nowait((None, None, None))
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        import time

        while not self._stop_evt.is_set():
            try:
                frame, features, confidence = self._request_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break

            t0 = time.monotonic()
            try:
                narration = self._call_api(frame, features, confidence)
                elapsed = time.monotonic() - t0
                if elapsed > _SLOW_THRESHOLD_SEC:
                    logger.warning(f"VLM narration slow: {elapsed:.1f}s")
                else:
                    logger.info(f"VLM narration ready in {elapsed:.1f}s")
                self._result_q.put(narration)
            except Exception as exc:  # noqa: BLE001 - log + continue, never crash demo
                logger.error(f"VLM narration failed: {exc}")

    def _call_api(
        self,
        frame: np.ndarray,
        features: np.ndarray,
        confidence: float,
    ) -> str:
        """Send a single multimodal request and return the model's text."""
        jpeg_bytes = _encode_frame_jpeg(frame)
        summary = _summarize_trajectory(features)
        user_text = (
            f"Alarm confidence at trigger: {confidence:.2f}.\n\n"
            f"{summary}\n\n"
            "Describe what just happened in the image."
        )

        contents = [
            self._types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            self._types.Part.from_text(text=user_text),
        ]

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=self._max_tokens,
                temperature=0.4,
            ),
        )

        text = (getattr(response, "text", None) or "").strip()
        return text or "(empty narration)"
