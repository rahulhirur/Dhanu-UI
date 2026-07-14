"""
ui_components.py — Reusable display components for DhanUI.
"""
from __future__ import annotations
from datetime import datetime
import streamlit as st

_MAX_MESSAGES = 20

# ---------------------------------------------------------------------------
# Status log helpers
# ---------------------------------------------------------------------------

def add_status(message: str, level: str = "info") -> None:
    """Append a timestamped status entry (info | success | warning | error)."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.status_messages.append({"message": message, "type": level, "time": ts})
    if len(st.session_state.status_messages) > _MAX_MESSAGES:
        st.session_state.status_messages = st.session_state.status_messages[-_MAX_MESSAGES:]


def render_status_log() -> None:
    """Render the status message log in the current Streamlit container."""
    st.subheader("📋 Status Log")
    if not st.session_state.status_messages:
        st.caption("No events yet…")
        return
    for msg in reversed(st.session_state.status_messages):
        css = f"status-box status-{msg['type']}"
        st.markdown(
            f"<div class='{css}'>[{msg['time']}] {msg['message']}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Task summary renderer
# ---------------------------------------------------------------------------

def render_task_summary(output_json: dict) -> None:
    """Render a concise task summary from a parsed LLM JSON plan."""
    st.subheader("📋 Task Summary")
    cols = st.columns(2)
    cols[0].metric("Intent", output_json.get("intent", "—"))
    cols[1].metric("Task Type", output_json.get("task_type", "—"))

    steps = output_json.get("steps", [])
    if steps:
        st.write("**Steps:**")
        for i, step in enumerate(steps, 1):
            action = step.get("action", "N/A")
            frm = step.get("from_area") or "—"
            to = step.get("to_area") or "—"
            obj = step.get("object") or "—"
            st.write(f"  {i}. **{action}** | from `{frm}` → to `{to}` | object: `{obj}`")


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

STATUS_CSS = """
<style>
.status-box {
    padding: 10px 15px;
    border-radius: 8px;
    margin: 4px 0;
    font-size: 0.85rem;
    font-weight: 500;
}
.status-success { background:#d4edda; color:#155724; border:1px solid #c3e6cb; }
.status-error   { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; }
.status-info    { background:#d1ecf1; color:#0c5460; border:1px solid #bee5eb; }
.status-warning { background:#fff3cd; color:#856404; border:1px solid #ffeeba; }
</style>
"""
