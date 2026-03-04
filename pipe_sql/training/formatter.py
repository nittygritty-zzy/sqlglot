"""Format trajectory steps into OpenAI chat-format message dicts."""

from __future__ import annotations

from dataclasses import dataclass, field

from pipe_sql.training.trajectory import TrajectoryStep

SYSTEM_MESSAGE = (
    "You are a SQL assistant that builds pipe SQL queries incrementally. "
    "Given a natural language question and a database schema, you construct "
    "the query one pipe operator at a time. Pipe SQL uses |> to chain operators: "
    "FROM, WHERE, SELECT, AGGREGATE, JOIN, ORDER BY, LIMIT, EXTEND. "
    "Respond with ONLY the next pipe operator to append."
)


@dataclass
class ChatSample:
    messages: list[dict[str, str]]
    metadata: dict[str, object] = field(default_factory=dict)
    tools: list[dict] | None = None


def _build_user_content(
    schema_str: str,
    question: str,
    query_so_far: str,
    cte_prefix: str,
) -> str:
    """Build the user message content for a trajectory step."""
    parts = [f"Schema:\n{schema_str}"]

    if cte_prefix:
        parts.append(f"\nGiven CTEs:\n{cte_prefix}")

    parts.append(f"\nQuestion: {question}")

    if query_so_far:
        parts.append(f"\nQuery so far:\n{query_so_far}")

    return "\n".join(parts)


def format_step(
    step: TrajectoryStep,
    schema_str: str,
    question: str,
    question_id: str,
    db_id: str,
    difficulty: str,
) -> ChatSample:
    """Format a single trajectory step into a ChatSample."""
    user_content = _build_user_content(
        schema_str=schema_str,
        question=question,
        query_so_far=step.query_so_far,
        cte_prefix=step.cte_prefix,
    )

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": step.operator_text},
    ]

    metadata = {
        "question_id": question_id,
        "db_id": db_id,
        "difficulty": difficulty,
        "step_index": step.step_index,
        "total_steps": step.total_steps,
        "op_type": step.op_type,
    }

    return ChatSample(messages=messages, metadata=metadata)


def format_trajectory(
    steps: list[TrajectoryStep],
    schema_str: str,
    question: str,
    question_id: str,
    db_id: str,
    difficulty: str,
) -> list[ChatSample]:
    """Format all trajectory steps for one golden pair."""
    return [
        format_step(step, schema_str, question, question_id, db_id, difficulty)
        for step in steps
    ]
