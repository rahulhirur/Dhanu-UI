"""
state.py — Session state initialisation for DhanUI.

Call `init_session_state()` once at the top of your Streamlit app before
rendering any widgets.
"""
from __future__ import annotations
import streamlit as st


_DEFAULTS: dict[str, object] = {
    "camera_enabled": False,
    "microphone_enabled": False,
    "camera_frame": None,
    "gesture_text": "",
    "sub_response": "",
    "speech_text": "",
    "combined_text": "",
    "llm_output": "",
    "status_messages": [],
    "gesture_recognizer": None,
    "camera_thread": None,
    "final_recognized_gesture_app": None,
    "last_audio_hash": None,
}


def init_session_state() -> None:
    """Idempotently initialise every session-state key we use."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
