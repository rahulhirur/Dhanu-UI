"""
tests/test_gesture.py — Unit tests for gesture_handler logic.

No hardware required — tests only the pure-logic parts.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Frontend_Streamlit.gesture_handler import gesture_to_action, GESTURE_MAP


class TestGestureToAction:
    # ── Original three ───────────────────────────────────────────────────────
    def test_thumb_up(self):
        action, valid = gesture_to_action("thumb_up")
        assert valid is True
        assert action == GESTURE_MAP["thumb_up"]

    def test_open_palm(self):
        action, valid = gesture_to_action("open_palm")
        assert valid is True
        assert action == GESTURE_MAP["open_palm"]

    def test_closed_fist(self):
        action, valid = gesture_to_action("closed_fist")
        assert valid is True
        assert action == GESTURE_MAP["closed_fist"]

    # ── New gestures ─────────────────────────────────────────────────────────
    def test_victory(self):
        action, valid = gesture_to_action("Victory")
        assert valid is True
        assert action == GESTURE_MAP["victory"]

    def test_pointing_up(self):
        action, valid = gesture_to_action("Pointing_Up")
        assert valid is True
        assert action == GESTURE_MAP["pointing_up"]

    def test_thumb_down(self):
        action, valid = gesture_to_action("Thumb_Down")
        assert valid is True
        assert action == GESTURE_MAP["thumb_down"]

    def test_iloveyou(self):
        action, valid = gesture_to_action("ILoveYou")
        assert valid is True
        assert action == GESTURE_MAP["iloveyou"]

    # ── Edge cases ───────────────────────────────────────────────────────────
    def test_case_insensitive(self):
        action, valid = gesture_to_action("THUMB_UP")
        assert valid is True
        assert action == GESTURE_MAP["thumb_up"]

    def test_unknown_gesture_returns_invalid(self):
        action, valid = gesture_to_action("wave")
        assert valid is False
        assert action == ""

    def test_empty_string_returns_invalid(self):
        action, valid = gesture_to_action("")
        assert valid is False
        assert action == ""

    def test_all_mapped_gestures_are_valid(self):
        """Every entry in GESTURE_MAP must return valid=True."""
        for name, expected_action in GESTURE_MAP.items():
            action, valid = gesture_to_action(name)
            assert valid is True, f"Expected valid=True for '{name}'"
            assert action == expected_action
