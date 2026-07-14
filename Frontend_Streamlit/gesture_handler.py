"""
gesture_handler.py — Gesture recognition helpers for DhanUI.

Handles:
  - MediaPipe model initialisation
  - Processing a camera frame for gesture detection
  - Mapping a gesture name to a robot action text
"""
from __future__ import annotations
import time
import os
import sys

import cv2
import numpy as np

# Ensure project root is on the path when imported from the Frontend_Streamlit directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MediaPipe_Model.media_pipe_video import GestureRecognitionApp

# ---------------------------------------------------------------------------
# Gesture → Action mapping
# ---------------------------------------------------------------------------

GESTURE_MAP: dict[str, str] = {
    # ── Original three ──────────────────────────────────────────────────────
    "thumb_up":      "Bring the catheter",
    "open_palm":     "Go back to neutral position",
    "closed_fist":   "Stop",
    # ── Full MediaPipe gesture set ───────────────────────────────────────────
    "victory":       "Dispose waste",
    "pointing_up":   "Pick up object",
    "thumb_down":    "Emergency stop",
    "iloveyou":      "Wake up robot",
}


def gesture_to_action(gesture_name: str) -> tuple[str, bool]:
    """
    Map a detected gesture name to a robot action string.

    Returns:
        (action_text, is_valid) — action_text is empty string if not recognised.
    """
    action = GESTURE_MAP.get(gesture_name.lower(), "")
    return action, bool(action)


# ---------------------------------------------------------------------------
# MediaPipe initialisation
# ---------------------------------------------------------------------------

def build_recognizer(log_fn=print) -> GestureRecognitionApp | None:
    """
    Initialise the MediaPipe gesture recogniser.

    Args:
        log_fn: Callable(message, level) used for status reporting.

    Returns:
        Initialised GestureRecognitionApp, or None on failure.
    """
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "MediaPipe_Model", "gesture_recognizer.task"
    )
    if not os.path.exists(model_path):
        log_fn(f"✗ Model not found at: {model_path}", "error")
        return None

    app = GestureRecognitionApp(model_path=model_path, camera_index=0)
    if not app.initialize_recognizer():
        log_fn("✗ Failed to initialize gesture recognizer", "error")
        return None

    app.start_time = time.time()
    # Camera is optional — headless HF Spaces won't have one
    camera_ok = app.initialize_camera()
    msg = (
        "✓ Gesture recognizer + local camera ready"
        if camera_ok
        else "✓ Gesture recognizer ready (web-mode, no local camera)"
    )
    log_fn(msg, "success")
    return app


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------

def process_frame(app: GestureRecognitionApp, frame: np.ndarray, log_fn=print) -> str:
    """
    Run gesture recognition on a single OpenCV frame.

    Args:
        app:    Initialised GestureRecognitionApp.
        frame:  BGR frame from st.camera_input / cv2.
        log_fn: Callable(message, level) for status reporting.

    Returns:
        Detected gesture name, or empty string.
    """
    try:
        app.process_frame(frame)
        time.sleep(0.5)  # allow async callback to fire
        gestures = app.get_detected_gestures()
        if not gestures:
            log_fn("⚠️ No gesture detected in frame", "warning")
            return ""

        latest = gestures[-1]
        log_fn(
            f"✓ Gesture detected: {latest['gesture']} (confidence: {latest['confidence']:.2f})",
            "success",
        )
        app.clear_gestures()
        return latest["gesture"]
    except Exception as exc:
        log_fn(f"✗ Gesture recognition error: {exc}", "error")
        return ""
