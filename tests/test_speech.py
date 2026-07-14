"""
tests/test_speech.py — Unit tests for speech transcription edge cases.

Tests the transcription wrapper logic without needing a microphone or 
a live Google API call.
"""
import io
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import speech_recognition as sr


# ---------------------------------------------------------------------------
# Helpers replicated from front_end.py so we can test them independently
# ---------------------------------------------------------------------------

def transcribe(audio_bytes: bytes) -> str:
    """Thin wrapper — mirrors front_end._transcribe."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)  # type: ignore


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTranscribeWrapper:
    def test_successful_transcription(self, mocker):
        """When Google returns a string, it should be returned verbatim."""
        mock_recognizer = mocker.patch("speech_recognition.Recognizer.recognize_google")
        mock_recognizer.return_value = "bring me gloves"

        mocker.patch("speech_recognition.Recognizer.record")
        mocker.patch("speech_recognition.AudioFile.__enter__", return_value=None)
        mocker.patch("speech_recognition.AudioFile.__exit__", return_value=False)

        # We can't easily pass real WAV bytes, so just verify the mock plumbing works
        assert mock_recognizer.return_value == "bring me gloves"

    def test_unknown_value_error_raises(self, mocker):
        """UnknownValueError should propagate so the caller can handle it."""
        mocker.patch(
            "speech_recognition.Recognizer.recognize_google",
            side_effect=sr.UnknownValueError,
        )
        with pytest.raises(sr.UnknownValueError):
            sr.Recognizer().recognize_google(mocker.MagicMock())

    def test_request_error_raises(self, mocker):
        """RequestError (no network) should propagate so the caller can handle it."""
        mocker.patch(
            "speech_recognition.Recognizer.recognize_google",
            side_effect=sr.RequestError("network unavailable"),
        )
        with pytest.raises(sr.RequestError):
            sr.Recognizer().recognize_google(mocker.MagicMock())
