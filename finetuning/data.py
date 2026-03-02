from __future__ import annotations

import json
import typing as t

from datasets import Dataset


def load_chat_dataset(path: str, limit: t.Optional[int] = None) -> Dataset:
    """Load JSONL with 'messages' and optional 'tools' fields, return HF Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            record: dict[str, t.Any] = {"messages": row["messages"]}
            if "tools" in row:
                record["tools"] = row["tools"]
            records.append(record)
            if limit and len(records) >= limit:
                break
    return Dataset.from_list(records)


def formatting_func(examples: dict[str, list]) -> list[str]:
    """Format messages into text strings for SFTTrainer.

    This is used as the formatting_func argument to SFTTrainer. The actual
    chat template application is handled by SFTTrainer when we pass the
    messages directly.
    """
    return examples["messages"]
