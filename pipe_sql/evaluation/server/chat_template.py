"""Replicate the exact training chat template from finetuning/train.py.

This module formats messages using direct string formatting (not Jinja2)
to match the training template exactly. The model was trained with this
specific format, so inference MUST use the same template.
"""

from __future__ import annotations

import json
import re


def format_messages(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Build a prompt string matching the training chat template.

    Format matches finetuning/train.py CHAT_TEMPLATE (lines 29-50):
    - system: <|im_start|>system\n{content}<|im_end|>\n
    - user: <|im_start|>user\n{content}<|im_end|>\n
    - assistant: <|im_start|>assistant\n{content}[tool_calls]<|im_end|>\n
    - tool: <|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n
    """
    parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            parts.append(f"<|im_start|>system\n{msg['content']}<|im_end|>\n")

        elif role == "user":
            parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>\n")

        elif role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            tc_str = ""
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "{}")
                # Arguments may be a dict (from OpenAI format) or string
                if isinstance(args, dict):
                    args = json.dumps(args)
                tc_str += f"\n<tool_call>\n{name}({args})\n</tool_call>"

            parts.append(f"<|im_start|>assistant\n{content}{tc_str}<|im_end|>\n")

        elif role == "tool":
            content = msg.get("content", "")
            parts.append(
                f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            )

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "".join(parts)


# Known tool names from training data
_KNOWN_TOOLS = {"list_tables", "describe_table", "sample_data", "execute_pipe_sql", "validate_pipe_sql"}

# Regex for training format: <tool_call>name({json_args})</tool_call>
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\w+)\((\{.*?\})\)\s*</tool_call>",
    re.DOTALL,
)

# Fallback regex for Qwen native format: {"name": ..., "arguments": ...}
_QWEN_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}\s*</tool_call>',
    re.DOTALL,
)

# Bare function call pattern (no tags) — handles models that generate garbled
# special tokens instead of proper <tool_call> tags.
# Matches: tool_name({json_args}) at word boundary
_BARE_CALL_RE = re.compile(
    r"\b(" + "|".join(_KNOWN_TOOLS) + r")\((\{.*?\})\)",
    re.DOTALL,
)


def parse_assistant_response(raw_text: str) -> dict:
    """Parse model output into content and optional tool calls.

    Handles multiple formats:
    1. Training format: <tool_call>name(args)</tool_call>
    2. Qwen native format: <tool_call>{"name":..., "arguments":...}</tool_call>
    3. Bare function calls: name(args) without tags (for models that
       generate garbled special tokens instead of proper <tool_call> tags)

    Returns:
        {"content": str, "tool_calls": list[dict] | None}
        Each tool call: {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
    """
    # Strip trailing <|im_end|> and whitespace
    text = raw_text.split("<|im_end|>")[0].strip()

    # Try training format first
    matches = _TOOL_CALL_RE.findall(text)
    tag_based = True

    if not matches:
        # Try Qwen native format
        matches = _QWEN_TOOL_CALL_RE.findall(text)

    if not matches:
        # Fallback: bare function call pattern (no tags)
        matches = _BARE_CALL_RE.findall(text)
        tag_based = False

    if not matches:
        return {"content": text, "tool_calls": None}

    # Extract content before the first tool call
    if tag_based:
        content_end = text.find("<tool_call>")
        content = text[:content_end].strip() if content_end >= 0 else text
    else:
        # For bare calls, find the first match position and take content before it
        first_match = _BARE_CALL_RE.search(text)
        if first_match:
            # Take text before the function name, strip non-alphanumeric trailing chars
            content = text[:first_match.start()].rstrip()
            # Remove any garbled trailing characters (non-ASCII)
            while content and ord(content[-1]) > 127:
                content = content[:-1]
            content = content.rstrip()
        else:
            content = text

    tool_calls = []
    for i, (name, args_str) in enumerate(matches):
        # Validate JSON args
        try:
            json.loads(args_str)
        except json.JSONDecodeError:
            continue

        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args_str,
            },
        })

    if not tool_calls:
        return {"content": text, "tool_calls": None}

    return {"content": content, "tool_calls": tool_calls}
