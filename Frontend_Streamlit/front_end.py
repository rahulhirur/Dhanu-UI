"""
front_end.py — DhanUI Streamlit application entry point.

Run with:
    uv run streamlit run Frontend_Streamlit/front_end.py
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import sys

import cv2
import numpy as np
import speech_recognition as sr
import streamlit as st
from gtts import gTTS

# ---------------------------------------------------------------------------
# Path setup — make project root importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Frontend_Streamlit.state import init_session_state
from Frontend_Streamlit.ui_components import STATUS_CSS, add_status, render_status_log, render_task_summary
from Frontend_Streamlit.gesture_handler import build_recognizer, gesture_to_action, process_frame
from S2LLM_Cerb.llm import query_cerebras

# Unity runner is only relevant on Windows
if sys.platform == "win32":
    from un_run.uni_runner import unity_runner

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DhanUI - Robot Control Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(STATUS_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
init_session_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tts(text: str) -> None:
    """Synthesise speech via gTTS and play it in the browser."""
    try:
        fp = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(fp)
        fp.seek(0)
        st.audio(fp, format="audio/mp3", autoplay=True)
        add_status(f"📢 Speaking: {text}", "info")
    except Exception as exc:
        add_status(f"✗ TTS error: {exc}", "error")


def _save_plan(output_json: dict) -> None:
    """Persist the LLM plan JSON, falling back to a local plans/ dir."""
    unity_path = os.path.join(
        r"C:\Users\kmu61\Downloads\Faps_unity-main\Faps_unity-main\Assets\Plans",
        "tool_delivery_plan.json",
    )
    try:
        os.makedirs(os.path.dirname(unity_path), exist_ok=True)
        with open(unity_path, "w") as f:
            json.dump(output_json, f, indent=4)
        add_status("✓ Plan saved to Unity directory", "success")
    except Exception:
        fallback = os.path.join(os.path.dirname(__file__), "..", "plans", "tool_delivery_plan.json")
        os.makedirs(os.path.dirname(fallback), exist_ok=True)
        with open(fallback, "w") as f:
            json.dump(output_json, f, indent=4)
        add_status(f"✓ Plan saved to fallback: {os.path.normpath(fallback)}", "success")


def _transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw WAV bytes via Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🤖 DhanUI — Robot Control Interface")
st.caption("Multi-modal input system: camera · microphone · gestures · LLM")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    # ── Camera ──────────────────────────────────────────────────────────────
    st.subheader("📹 Camera")
    st.session_state.camera_enabled = st.checkbox("Enable camera", value=st.session_state.camera_enabled)

    if st.session_state.camera_enabled:
        img_file = st.camera_input("Snap a frame")
        if img_file:
            raw = np.frombuffer(img_file.getvalue(), np.uint8)
            st.session_state.camera_frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            add_status("✓ Frame captured", "success")

    st.divider()

    # ── Microphone ──────────────────────────────────────────────────────────
    st.subheader("🎙️ Microphone")
    st.session_state.microphone_enabled = st.checkbox("Enable microphone", value=st.session_state.microphone_enabled)

    if st.session_state.microphone_enabled:
        audio_file = st.audio_input("Record speech")
        if audio_file:
            audio_bytes = audio_file.read()
            h = hashlib.md5(audio_bytes).hexdigest()
            if st.session_state.last_audio_hash != h:
                st.session_state.last_audio_hash = h
                add_status("🎙️ Transcribing…", "info")
                try:
                    text = _transcribe(audio_bytes)
                    st.session_state.speech_text = text
                    add_status(f"✓ Recognised: {text}", "success")
                except sr.UnknownValueError:
                    st.session_state.speech_text = ""
                    add_status("✗ Could not understand audio", "warning")
                except Exception as exc:
                    st.session_state.speech_text = ""
                    add_status(f"✗ Transcription error: {exc}", "error")

    st.divider()

    # ── Gesture recognizer ──────────────────────────────────────────────────
    st.subheader("👆 Gesture")
    if st.button("🔧 Init gesture recognizer"):
        st.session_state.gesture_recognizer = build_recognizer(log_fn=add_status)

    if st.button("▶️ Run on captured frame"):
        if st.session_state.camera_frame is None:
            add_status("⚠️ No frame captured yet", "warning")
        elif st.session_state.gesture_recognizer is None:
            add_status("⚠️ Initialise gesture recognizer first", "warning")
        else:
            name = process_frame(st.session_state.gesture_recognizer, st.session_state.camera_frame, add_status)
            st.session_state.gesture_text = name
            if name:
                action, _ = gesture_to_action(name)
                st.session_state.sub_response = action

    st.divider()

    # ── LLM Orchestrator ────────────────────────────────────────────────────
    if st.button("⚡ Process through LLM", use_container_width=True, type="primary"):
        if st.session_state.sub_response:
            _tts(f"Gesture: {st.session_state.sub_response}. Executing.")
        combined = f"{st.session_state.sub_response} {st.session_state.speech_text}".strip()
        if combined:
            st.session_state.combined_text = combined
            add_status(f"🤖 Sending to LLM: {combined}", "info")
            try:
                response = query_cerebras(combined)
                st.session_state.llm_output = response
                add_status("✓ LLM response received", "success")
            except Exception as exc:
                add_status(f"✗ LLM error: {exc}", "error")
        else:
            add_status("⚠️ No gesture or speech input to send", "warning")

    # ── Unity runner ─────────────────────────────────────────
    st.divider()
    st.button(
        "🎮 Launch Unity Sim",
        disabled=True,
        help="Unity simulation is not enabled.",
        use_container_width=True
    )

# ---------------------------------------------------------------------------
# Main content — two columns
# ---------------------------------------------------------------------------
col_inputs, col_outputs = st.columns(2)

with col_inputs:
    st.header("📥 Inputs")

    if st.session_state.camera_enabled and st.session_state.camera_frame is not None:
        frame_rgb = cv2.cvtColor(st.session_state.camera_frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Captured frame", use_container_width=True)

    st.markdown("**👆 Gesture**")
    if st.session_state.gesture_text:
        st.success(f"{st.session_state.gesture_text}  →  {st.session_state.sub_response}")
    else:
        st.caption("No gesture detected yet")

    st.markdown("**🗣️ Speech**")
    if st.session_state.speech_text:
        st.success(st.session_state.speech_text)
    else:
        st.caption("No speech transcribed yet")

    st.markdown("**🔗 Combined input to LLM**")
    if st.session_state.combined_text:
        st.info(st.session_state.combined_text)
    else:
        st.caption("Combined input will appear here")

with col_outputs:
    st.header("📊 Results")
    render_status_log()
    st.divider()

    st.subheader("🤖 LLM Output")
    if st.session_state.llm_output:
        try:
            plan = json.loads(st.session_state.llm_output)
            st.json(plan)
            _save_plan(plan)
            render_task_summary(plan)
        except json.JSONDecodeError:
            st.text_area("Raw output", value=st.session_state.llm_output, height=300, disabled=True)
    else:
        st.caption("LLM output will appear here after processing")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
cam_status = "✓ On" if st.session_state.camera_enabled else "✗ Off"
mic_status = "✓ On" if st.session_state.microphone_enabled else "✗ Off"
gest_status = "✓ Ready" if st.session_state.gesture_recognizer else "✗ Not init"
st.caption(f"Camera: {cam_status}  ·  Microphone: {mic_status}  ·  Gesture recognizer: {gest_status}")