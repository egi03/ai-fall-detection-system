"""
Real-time display manager for the fall detection application.

Manages the OpenCV display window, status bar, confidence bar, and
alarm indicators. Includes display-side smoothing to prevent visual
flicker from frame-to-frame noise in model outputs.

DECISION: Using cv2.imshow per research/8.1. It must run in the main
thread. cv2.waitKey(1) is required to pump the OS message queue.

Reference: research/8.1 - cv2.imshow must run in main thread.
cv2.waitKey(1) required to pump the OS message queue.
Reference: research/8.4 - Non-blocking multithreaded architecture.
Reference: research/9.1 - Status bar and alarm indicator design.
"""

import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from src.visualization.skeleton import (
    COLOR_NORMAL,
    COLOR_WARNING,
    COLOR_ALARM,
    COLOR_TEXT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Map alarm state strings to colors
_STATE_COLORS = {
    "NORMAL": COLOR_NORMAL,
    "IMPACT_DETECTED": COLOR_WARNING,
    "FALL_CONFIRMED": COLOR_ALARM,
    "RECOVERED": (0, 200, 200),       # Yellow-ish — recovered
    "NO PERSON": (128, 128, 128),     # Gray — no person detected
}

# Prefix match for INIT states (e.g. "INIT (15/30)")
_INIT_COLOR = (255, 200, 0)  # Cyan-ish — initializing

# Confidence bar colors (BGR): green → yellow → orange → red
_BAR_COLORS = [
    (0, 180, 0),      # Low confidence: green
    (0, 200, 200),    # Medium: yellow
    (0, 140, 255),    # High: orange
    (0, 0, 255),      # Very high: red
]


def _confidence_bar_color(confidence: float) -> Tuple[int, int, int]:
    """Return interpolated BGR color for a confidence value [0, 1]."""
    if confidence < 0.25:
        return _BAR_COLORS[0]
    elif confidence < 0.50:
        return _BAR_COLORS[1]
    elif confidence < 0.75:
        return _BAR_COLORS[2]
    else:
        return _BAR_COLORS[3]


def _draw_text_with_shadow(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw text with a dark shadow for readability on any background."""
    import cv2

    # Shadow offset
    sx, sy = org[0] + 1, org[1] + 1
    cv2.putText(img, text, (sx, sy), font, font_scale, shadow_color, thickness + 1)
    cv2.putText(img, text, org, font, font_scale, color, thickness)


class DisplayManager:
    """
    Manages the real-time video display with status overlays.

    Includes display-side smoothing to prevent visual flicker:
    - Confidence value is EMA-smoothed for display
    - FPS is updated at a fixed interval (not every frame)
    - "NO PERSON" is debounced to avoid 1-frame flashes
    - Status bar has a fixed minimum width

    Parameters
    ----------
    window_name : str
        Name of the OpenCV display window.
    resolution : tuple of int
        Display resolution (width, height).
    fps_update_interval : float
        Seconds between FPS display updates.
    no_person_grace_frames : int
        Number of consecutive no-person frames before showing "NO PERSON".
    """

    def __init__(
        self,
        window_name: str = "Fall Detection System",
        resolution: Tuple[int, int] = (640, 480),
        fps_update_interval: float = 0.5,
        no_person_grace_frames: int = 5,
        sparkline_length: int = 150,
    ) -> None:
        self._window_name = window_name
        self._resolution = resolution
        self._is_open = False

        # Display smoothing state
        self._display_conf = 0.0           # EMA-smoothed confidence for display
        self._display_conf_alpha = 0.3     # Display EMA factor
        self._displayed_fps = 0.0          # Last displayed FPS value
        self._fps_update_interval = fps_update_interval
        self._last_fps_update = 0.0

        # "NO PERSON" debouncing
        self._no_person_grace = no_person_grace_frames
        self._no_person_count = 0
        self._last_valid_state = "NORMAL"
        self._last_valid_conf = 0.0

        # Probability history for sparkline
        self._prob_history: deque = deque(maxlen=sparkline_length)

        # Alarm border pulsing
        self._frame_counter = 0

        # Fixed minimum width for status bar (computed once)
        self._min_status_width = 0

    def _ensure_window(self) -> None:
        """Create the display window if not already open."""
        if not self._is_open:
            import cv2

            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self._window_name,
                self._resolution[0],
                self._resolution[1],
            )
            self._is_open = True

    def show(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        alarm_state: Optional[str] = None,
        confidence: float = 0.0,
        fall_count: int = 0,
        top_features: Optional[List[Tuple[str, float]]] = None,
        narration: Optional[str] = None,
    ) -> bool:
        """
        Display a frame with status overlays.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to display.
        fps : float
            Current processing FPS to show in status bar.
        alarm_state : str, optional
            Current alarm state string (e.g., 'NORMAL', 'FALL_CONFIRMED').
        confidence : float
            Current model confidence score [0.0, 1.0].

        Returns
        -------
        bool
            False if the user requested to close the window (pressed 'q').
        """
        import cv2

        self._ensure_window()
        self._frame_counter += 1

        output = frame.copy()
        h, w = output.shape[:2]

        # --- Smooth confidence for display ---
        self._display_conf = (
            self._display_conf_alpha * confidence
            + (1.0 - self._display_conf_alpha) * self._display_conf
        )
        self._prob_history.append(self._display_conf)

        # --- Debounce "NO PERSON" ---
        if alarm_state == "NO PERSON":
            self._no_person_count += 1
            if self._no_person_count < self._no_person_grace:
                # Hold last valid state during grace period
                alarm_state = self._last_valid_state
                confidence = self._last_valid_conf
        else:
            self._no_person_count = 0
            self._last_valid_state = alarm_state or "NORMAL"
            self._last_valid_conf = confidence

        # --- Update displayed FPS at fixed intervals ---
        now = time.monotonic()
        if fps > 0 and (now - self._last_fps_update >= self._fps_update_interval):
            self._displayed_fps = fps
            self._last_fps_update = now

        # --- Flash tint on FALL_CONFIRMED (drawn before UI so overlays stay on top) ---
        if alarm_state == "FALL_CONFIRMED":
            pulse_t = abs((self._frame_counter % 20) - 10) / 10.0
            alpha = 0.12 + 0.07 * pulse_t
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 160), -1)
            cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)

        # --- Draw FPS counter (bottom-left) with shadow ---
        if self._displayed_fps > 0:
            fps_text = f"FPS: {self._displayed_fps:.1f}"
            _draw_text_with_shadow(
                output,
                fps_text,
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1,
            )

        # --- Draw alarm state (top-left) ---
        if alarm_state:
            if alarm_state.startswith("INIT"):
                color = _INIT_COLOR
            else:
                color = _STATE_COLORS.get(alarm_state, COLOR_NORMAL)

            # Build display text — use smoothed confidence
            display_text = alarm_state.replace("_", " ")
            show_conf = self._display_conf if confidence > 0 else 0.0
            if show_conf > 0.005:
                display_text = f"{display_text}  {show_conf:.1%}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(
                display_text, font, font_scale, thickness
            )

            # Fixed minimum width to prevent rectangle resizing flicker
            if self._min_status_width == 0:
                # Compute once: width of longest expected text
                ref_text = "FALL CONFIRMED  100.0%"
                (ref_w, _), _ = cv2.getTextSize(ref_text, font, font_scale, thickness)
                self._min_status_width = ref_w + 20
            box_w = max(tw + 10, self._min_status_width)

            pad = 8
            box_top = pad
            box_bottom = pad + th + baseline + 10

            # Status background
            cv2.rectangle(
                output,
                (pad, box_top),
                (pad + box_w, box_bottom),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                output,
                display_text,
                (pad + 5, pad + th + 5),
                font,
                font_scale,
                color,
                thickness,
            )

            # --- Confidence bar (below status text) ---
            if show_conf > 0.005:
                bar_y = box_bottom + 4
                bar_h = 8
                bar_max_w = box_w - 4
                bar_fill_w = int(bar_max_w * min(self._display_conf, 1.0))

                # Bar background
                cv2.rectangle(
                    output,
                    (pad, bar_y),
                    (pad + box_w, bar_y + bar_h),
                    (0, 0, 0),
                    -1,
                )
                # Bar outline
                cv2.rectangle(
                    output,
                    (pad + 2, bar_y + 1),
                    (pad + 2 + bar_max_w, bar_y + bar_h - 1),
                    (60, 60, 60),
                    1,
                )
                # Bar fill
                if bar_fill_w > 0:
                    bar_color = _confidence_bar_color(self._display_conf)
                    cv2.rectangle(
                        output,
                        (pad + 2, bar_y + 1),
                        (pad + 2 + bar_fill_w, bar_y + bar_h - 1),
                        bar_color,
                        -1,
                    )

            # --- Pulsing red border on FALL_CONFIRMED ---
            if alarm_state == "FALL_CONFIRMED":
                pulse = abs((self._frame_counter % 20) - 10)
                border_thickness = 3 + int(pulse * 0.3)
                cv2.rectangle(
                    output,
                    (0, 0),
                    (w - 1, h - 1),
                    COLOR_ALARM,
                    border_thickness,
                )

        # --- Fall count (top-right) ---
        if fall_count > 0:
            count_text = f"Falls: {fall_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (cw, ch), _ = cv2.getTextSize(count_text, font, 0.65, 2)
            _draw_text_with_shadow(
                output, count_text, (w - cw - 12, ch + 10), font, 0.65, (0, 220, 220), 2,
            )

        # --- Centered "FALL DETECTED" text ---
        if alarm_state == "FALL_CONFIRMED":
            label = "FALL DETECTED"
            fscale = min(w / 380.0, 1.8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (lw, lh), _ = cv2.getTextSize(label, font, fscale, 3)
            lx, ly = (w - lw) // 2, h // 2 + lh // 2
            cv2.putText(output, label, (lx + 2, ly + 2), font, fscale, (0, 0, 0), 5)
            cv2.putText(output, label, (lx, ly), font, fscale, COLOR_ALARM, 3)

        # --- P(fall) sparkline (bottom-right) ---
        if len(self._prob_history) > 2:
            gw, gh = 150, 45
            gx = w - gw - 10
            gy = h - gh - 28
            cv2.rectangle(output, (gx - 3, gy - 14), (gx + gw + 3, gy + gh + 3), (0, 0, 0), -1)
            cv2.rectangle(output, (gx - 3, gy - 14), (gx + gw + 3, gy + gh + 3), (50, 50, 50), 1)
            ty = gy + gh - int(0.5 * gh)
            cv2.line(output, (gx, ty), (gx + gw, ty), (0, 180, 180), 1)
            probs = list(self._prob_history)
            maxlen = self._prob_history.maxlen or len(probs)
            pts = [
                (
                    gx + int(i * gw / max(maxlen - 1, 1)),
                    max(gy, min(gy + gh, gy + gh - int(max(0.0, min(1.0, p)) * gh))),
                )
                for i, p in enumerate(probs)
            ]
            for i in range(1, len(pts)):
                col = COLOR_ALARM if probs[i] >= 0.5 else COLOR_NORMAL
                cv2.line(output, pts[i - 1], pts[i], col, 1)
            cv2.putText(
                output, "P(fall)", (gx, gy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1,
            )

        # --- Top features attribution panel (right side, mid-height) ---
        if top_features:
            self._draw_top_features_panel(output, top_features)

        # --- VLM narration banner (bottom strip) ---
        if narration:
            self._draw_narration_banner(output, narration)

        cv2.imshow(self._window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            return False

        return True

    def _draw_top_features_panel(
        self,
        output: np.ndarray,
        top_features: List[Tuple[str, float]],
    ) -> None:
        """Draw a compact "Why?" panel listing top attribution features.

        Parameters
        ----------
        output : np.ndarray
            Frame to draw on (modified in place).
        top_features : list of (str, float)
            Sequence of ``(feature_name, signed_intensity)`` tuples where
            intensity is in ``[-1, 1]``. The list is rendered in order.
        """
        import cv2

        h, w = output.shape[:2]
        panel_w = 180
        row_h = 22
        rows = len(top_features)
        if rows == 0:
            return
        pad = 8
        panel_h = pad + 22 + rows * row_h + pad  # header + rows
        x0 = w - panel_w - 10
        y0 = max(120, h // 2 - panel_h // 2)

        cv2.rectangle(output, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(output, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)
        cv2.putText(
            output, "Why?  (top features)",
            (x0 + pad, y0 + pad + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
        )

        bar_left = x0 + pad
        bar_right = x0 + panel_w - pad
        mid_x = (bar_left + bar_right) // 2
        bar_max_half = (bar_right - bar_left) // 2 - 2

        for i, (name, intensity) in enumerate(top_features):
            row_top = y0 + pad + 22 + i * row_h
            text_y = row_top + 8
            display_name = name if len(name) <= 18 else name[:17] + "…"
            cv2.putText(
                output, display_name,
                (bar_left, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1,
            )

            # Symmetric divergent bar — center anchored at midpoint
            bar_y = row_top + 13
            bar_h = 5
            cv2.line(
                output, (mid_x, bar_y - 1), (mid_x, bar_y + bar_h + 1),
                (140, 140, 140), 1,
            )
            mag = max(-1.0, min(1.0, float(intensity)))
            length = int(abs(mag) * bar_max_half)
            if length > 0:
                if mag >= 0:
                    color = (40, 40, 255)  # red — pushes toward fall
                    cv2.rectangle(
                        output, (mid_x, bar_y), (mid_x + length, bar_y + bar_h),
                        color, -1,
                    )
                else:
                    color = (255, 140, 40)  # blue — pushes away from fall
                    cv2.rectangle(
                        output, (mid_x - length, bar_y), (mid_x, bar_y + bar_h),
                        color, -1,
                    )

    def _draw_narration_banner(self, output: np.ndarray, narration: str) -> None:
        """Render a wrapped multi-line VLM narration along the bottom strip."""
        import cv2

        h, w = output.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        max_chars = max(20, w // 9)

        # Word-wrap into lines
        words = narration.split()
        lines: List[str] = []
        current = ""
        for word in words:
            tentative = (current + " " + word).strip()
            if len(tentative) > max_chars and current:
                lines.append(current)
                current = word
            else:
                current = tentative
        if current:
            lines.append(current)
        lines = lines[:4]  # cap to four lines to keep banner compact

        line_h = 18
        pad = 8
        banner_h = pad * 2 + line_h * len(lines)
        banner_top = h - banner_h - 38  # leave room for the FPS line

        overlay = output.copy()
        cv2.rectangle(overlay, (0, banner_top), (w, banner_top + banner_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        cv2.rectangle(output, (0, banner_top), (w, banner_top + banner_h),
                      (100, 100, 100), 1)

        for i, line in enumerate(lines):
            ty = banner_top + pad + (i + 1) * line_h - 4
            cv2.putText(output, line, (12, ty), font, scale,
                        (240, 240, 240), thick, cv2.LINE_AA)

    def destroy(self) -> None:
        """Destroy the display window and release resources."""
        if self._is_open:
            import cv2

            cv2.destroyWindow(self._window_name)
            self._is_open = False
            logger.info("Display window destroyed")
