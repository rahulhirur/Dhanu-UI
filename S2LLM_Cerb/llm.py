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
    },

    # Example 5
    {
        "role": "user",
        "content": "get the forceps"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "fetch",
            "task_type": "tool_fetch",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Tool_area", "object": None},
                {"action": "pick", "from_area": "Tool_area", "to_area": None, "object": "forceps"},
                {"action": "move", "from_area": "Tool_area", "to_area": "Handover_area", "object": None},
                {"action": "handover", "from_area": "Handover_area", "to_area": None, "object": "forceps"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 6
    {
        "role": "user",
        "content": "bring gloves"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "fetch",
            "task_type": "tool_fetch",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Tool_area", "object": None},
                {"action": "pick", "from_area": "Tool_area", "to_area": None, "object": "gloves"},
                {"action": "move", "from_area": "Tool_area", "to_area": "Handover_area", "object": None},
                {"action": "handover", "from_area": "Handover_area", "to_area": None, "object": "gloves"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 7
    {
        "role": "user",
        "content": "fetch a syringe"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "fetch",
            "task_type": "tool_fetch",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Tool_area", "object": None},
                {"action": "pick", "from_area": "Tool_area", "to_area": None, "object": "syringe"},
                {"action": "move", "from_area": "Tool_area", "to_area": "Handover_area", "object": None},
                {"action": "handover", "from_area": "Handover_area", "to_area": None, "object": "syringe"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 8
    {
        "role": "user",
        "content": "hand over the scissors"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "handover",
            "task_type": "tool_handover",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Tool_area", "object": None},
                {"action": "pick", "from_area": "Tool_area", "to_area": None, "object": "scissors"},
                {"action": "move", "from_area": "Tool_area", "to_area": "Handover_area", "object": None},
                {"action": "handover", "from_area": "Handover_area", "to_area": None, "object": "scissors"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 9
    {
        "role": "user",
        "content": "pass the clamp"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "handover",
            "task_type": "tool_handover",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Tool_area", "object": None},
                {"action": "pick", "from_area": "Tool_area", "to_area": None, "object": "clamp"},
                {"action": "move", "from_area": "Tool_area", "to_area": "Handover_area", "object": None},
                {"action": "handover", "from_area": "Handover_area", "to_area": None, "object": "clamp"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 10
    {
        "role": "user",
        "content": "dispose the gloves"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "dispose",
            "task_type": "waste_disposal",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Handover_area", "object": None},
                {"action": "pick", "from_area": "Handover_area", "to_area": None, "object": "gloves"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Trash_area", "object": None},
                {"action": "dispose", "from_area": "Trash_area", "to_area": None, "object": "gloves"},
                {"action": "move", "from_area": "Trash_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 11
    {
        "role": "user",
        "content": "throw away the syringe"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "dispose",
            "task_type": "waste_disposal",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Handover_area", "object": None},
                {"action": "pick", "from_area": "Handover_area", "to_area": None, "object": "syringe"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Trash_area", "object": None},
                {"action": "dispose", "from_area": "Trash_area", "to_area": None, "object": "syringe"},
                {"action": "move", "from_area": "Trash_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 12
    {
        "role": "user",
        "content": "discard the used gauze"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "dispose",
            "task_type": "waste_disposal",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Handover_area", "object": None},
                {"action": "pick", "from_area": "Handover_area", "to_area": None, "object": "gauze"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Trash_area", "object": None},
                {"action": "dispose", "from_area": "Trash_area", "to_area": None, "object": "gauze"},
                {"action": "move", "from_area": "Trash_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 13
    {
        "role": "user",
        "content": "return the unused gloves"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "return",
            "task_type": "tool_return",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Handover_area", "object": None},
                {"action": "pick", "from_area": "Handover_area", "to_area": None, "object": "gloves"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Tool_area", "object": None},
                {"action": "place", "from_area": "Tool_area", "to_area": None, "object": "gloves"},
                {"action": "move", "from_area": "Tool_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 14
    {
        "role": "user",
        "content": "put this tool back"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "return",
            "task_type": "tool_return",
            "steps": [
                {"action": "move", "from_area": None, "to_area": "Handover_area", "object": None},
                {"action": "pick", "from_area": "Handover_area", "to_area": None, "object": "tool"},
                {"action": "move", "from_area": "Handover_area", "to_area": "Tool_area", "object": None},
                {"action": "place", "from_area": "Tool_area", "to_area": None, "object": "tool"},
                {"action": "move", "from_area": "Tool_area", "to_area": "Neutral_area", "object": None}
            ]
        })
    },

    # Example 15
    {
        "role": "user",
        "content": "wait"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "standby",
            "task_type": "idle",
            "steps": [
                {"action": "move", "from_area": None, "to_area": None, "object": None}
            ]
        })
    },

    # Example 16
    {
        "role": "user",
        "content": "stand by"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "standby",
            "task_type": "idle",
            "steps": [
                {"action": "move", "from_area": None, "to_area": None, "object": None}
            ]
        })
    },

    # Example 17
    {
        "role": "user",
        "content": "go back to home"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "reset",
            "task_type": "reset_position",
            "steps": [
                {"action": "move", "from_area": None, "to_area": None, "object": None}
            ]
        })
    },

    # Example 18 
    {
        "role": "user",
        "content": "reset position"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "reset",
            "task_type": "reset_position",
            "steps": [
                {"action": "move", "from_area": None, "to_area": None, "object": None}
            ]
        })
    },

    # Example 19
    {
        "role": "user",
        "content": "reset position"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "intent": "reset",
            "task_type": "reset_position",
            "steps": [
                {"action": "move", "from_area": None, "to_area": None, "object": None}
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


