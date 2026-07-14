"""
tests/test_llm.py — Unit tests for the LLM query wrapper.

Mocks _get_client() so no API key is needed during CI.
"""
import json
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VALID_PLAN = {
    "intent": "fetch",
    "task_type": "tool_delivery",
    "steps": [
        {"action": "move", "from_area": None, "to_area": "Tool_area", "object": None},
        {"action": "pick", "from_area": "Tool_area", "to_area": None, "object": "gloves"},
        {"action": "move", "from_area": "Tool_area", "to_area": "Handover_area", "object": None},
        {"action": "handover", "from_area": "Handover_area", "to_area": None, "object": "gloves"},
        {"action": "move", "from_area": "Handover_area", "to_area": "Neutral_area", "object": None},
    ],
}


def _make_mock_client(mocker):
    """Return a mock Cerebras client whose completions.create returns VALID_PLAN."""
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = json.dumps(VALID_PLAN)

    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    # Patch the lazy getter so it returns our mock without ever hitting the API
    mocker.patch("S2LLM_Cerb.llm._get_client", return_value=mock_client)
    return mock_client


class TestQueryCerebras:
    def test_returns_valid_json(self, mocker):
        """query_cerebras should return JSON-parseable text."""
        _make_mock_client(mocker)
        from S2LLM_Cerb.llm import query_cerebras
        result = query_cerebras("bring me gloves")
        parsed = json.loads(result)
        assert parsed["intent"] == "fetch"
        assert "steps" in parsed
        assert len(parsed["steps"]) > 0

    def test_output_has_required_keys(self, mocker):
        _make_mock_client(mocker)
        from S2LLM_Cerb.llm import query_cerebras
        result = json.loads(query_cerebras("bring me gloves"))
        assert set(result.keys()) >= {"intent", "task_type", "steps"}

    def test_step_schema(self, mocker):
        """Every step in the plan must have action, from_area, to_area, object."""
        _make_mock_client(mocker)
        from S2LLM_Cerb.llm import query_cerebras
        result = json.loads(query_cerebras("bring me gloves"))
        required_keys = {"action", "from_area", "to_area", "object"}
        for step in result["steps"]:
            assert required_keys <= set(step.keys()), f"Missing keys in step: {step}"
