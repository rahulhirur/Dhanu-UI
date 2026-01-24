import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import json

load_dotenv()

client = Cerebras(
    api_key=os.getenv("CEREBRAS_API_KEY")
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["wake_up", "fetch", "dispose", "idle", "unknown"]
        },
        "task_type": {
            "type": "string"
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "from_area": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"}
                        ]
                    },
                    "to_area": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"}
                        ]
                    },
                    "object": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"}
                        ]
                    }
                },
                "required": ["action", "from_area", "to_area", "object"],
                "additionalProperties": False
            }
        }
    },
    "required": ["intent", "task_type", "steps"],
    "additionalProperties": False
}

SYSTEM_PROMPT = """
You are a robot task-planning assistant.

The robot operates ONLY in the following areas:
- Tool_area
- Handover_area
- Trash_area
- Neutral_area

Your job is to convert user commands into a structured JSON task plan.

Rules:
- Respond ONLY in valid JSON matching the schema.
- Use area names exactly as defined.
- Tools are always picked from Tool_area.
- Items handed to humans go to Handover_area.
- Items to be discarded go to Trash_area.
- After every task, return to Neutral_area.
- Greetings or start commands wake the robot and move it to Neutral_area.
- If a command is unclear, infer the closest logical task.
- Never add explanations, comments, or extra text.
"""

FEW_SHOT_EXAMPLES = [
    # Example 1
    {
        "role": "user",
        "content": "hello robot, we are starting"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "wake_up",
            "task_type": "initialization",
            "steps": [
                {"action": "wake_up", "from_area": None, "to_area": None, "object": None},
                {"action": "move", "from_area": None, "to_area": "Neutral_area", "object": None},
                {"action": "wait", "from_area": "Neutral_area", "to_area": None, "object": None}
            ]
        })
    },

    # Example 2
    {
        "role": "user",
        "content": "bring me gloves"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "fetch",
            "task_type": "tool_delivery",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Tool_area", "object": None},
                {"action": "pick", "from_area": "Tool_area", "to_area": None, "object": "gloves"},
                {"action": "move", "from_area": "Tool_area", "to_area": "Handover_area", "object": None},
                {"action": "handover", "from_area": "Handover_area", "to_area": None, "object": "gloves"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 4
    {
        "role": "user",
        "content": "throw this trash"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "dispose",
            "task_type": "waste_disposal",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Handover_area", "object": None},
                {"action": "pick", "from_area": "Handover_area", "to_area": None, "object": "trash"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Trash_area", "object": None},
                {"action": "dispose", "from_area": "Trash_area", "to_area": None, "object": "trash"},
                {"action": "move", "from_area": "Trash_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    }
]


def query_cerebras(user_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + str(FEW_SHOT_EXAMPLES)
                
                
            },
            
            {
                "role": "user",
                "content": user_text
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "robot_task_plan",
                "strict": True,
                "schema": JSON_SCHEMA
            }
        },
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content

