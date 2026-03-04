"""Build tool-calling conversations for training data."""

from __future__ import annotations

import json
import random

import sqlglot
from sqlglot import exp

from pipe_sql.training.formatter import ChatSample
from pipe_sql.training.schema_extractor import TableSchema
from pipe_sql.training.tool_executor import (
    describe_table_result,
    execute_pipe_sql_result,
    list_tables_result,
    sample_data_result,
    validate_pipe_sql_result,
)

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all table names in a database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_id": {"type": "string", "description": "Database identifier"},
                },
                "required": ["db_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Get columns, types, primary keys, and foreign keys for a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_id": {"type": "string", "description": "Database identifier"},
                    "table_name": {"type": "string", "description": "Table name"},
                },
                "required": ["db_id", "table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_data",
            "description": "Return sample rows from a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_id": {"type": "string", "description": "Database identifier"},
                    "table_name": {"type": "string", "description": "Table name"},
                    "limit": {"type": "integer", "description": "Max rows to return", "default": 5},
                },
                "required": ["db_id", "table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_pipe_sql",
            "description": "Execute a pipe SQL query and return results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_id": {"type": "string", "description": "Database identifier"},
                    "pipe_sql": {"type": "string", "description": "Pipe SQL query"},
                },
                "required": ["db_id", "pipe_sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_pipe_sql",
            "description": "Check pipe SQL syntax validity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipe_sql": {"type": "string", "description": "Pipe SQL query to validate"},
                },
                "required": ["pipe_sql"],
            },
        },
    },
]

TOOL_SYSTEM_MESSAGE = (
    "You are a SQL assistant that builds pipe SQL queries. "
    "You have access to tools for exploring database schemas and executing queries. "
    "Pipe SQL uses |> to chain operators: FROM, WHERE, SELECT, AGGREGATE, JOIN, ORDER BY, LIMIT, EXTEND. "
    "First explore the schema, then write the final pipe SQL query."
)

_LIST_TABLES_THOUGHTS = [
    "Let me start by exploring the database schema.",
    "I'll first check what tables are available in this database.",
    "Let me see what tables exist in the database.",
    "First, I need to understand the database structure.",
]

_DESCRIBE_TABLE_THOUGHTS = [
    "Let me look at the columns in the {table} table.",
    "I'll examine the schema of the {table} table.",
    "Let me check the structure of {table}.",
    "I need to see what columns {table} has.",
]

_SAMPLE_DATA_THOUGHTS = [
    "Let me look at some sample data from {table} to understand the values.",
    "I'll check some example rows from {table}.",
    "Let me see what data looks like in {table}.",
]

_EXECUTE_THOUGHTS = [
    "Let me test a partial query first.",
    "I'll verify the intermediate result.",
    "Let me check if this partial query works.",
]

_FINAL_THOUGHTS = [
    "Now I can write the final pipe SQL query.",
    "Based on the schema, here is the pipe SQL query.",
    "Here's the final pipe SQL query.",
    "Now I have enough information to write the query.",
]

_VERIFY_THOUGHTS = [
    "Let me verify this query works correctly.",
    "I'll run the query to check the results.",
    "Let me execute the query to make sure it's correct.",
    "I should verify the query produces the expected output.",
]

_VALIDATE_THOUGHTS = [
    "Let me validate the syntax of this query.",
    "I'll check that the query syntax is correct.",
    "Let me verify the pipe SQL syntax.",
]

_CONFIRM_MESSAGES = [
    "The query executed successfully and returns the expected results.",
    "The results look correct. The query works as expected.",
    "Verified — the query returns the right output.",
    "The query runs successfully and produces the correct results.",
]

_VALIDATE_CONFIRM_MESSAGES = [
    "The syntax is valid. The query is ready to use.",
    "Syntax check passed. The query is correct.",
    "The pipe SQL syntax is valid.",
]


def extract_referenced_tables(gold_sql: str) -> list[str]:
    """Parse gold SQL AST and extract table names from FROM/JOIN clauses."""
    try:
        ast = sqlglot.parse_one(gold_sql, dialect="sqlite")
        tables = []
        seen = set()
        for table in ast.find_all(exp.Table):
            name = table.name
            if name and name.lower() not in seen:
                seen.add(name.lower())
                tables.append(name)
        return tables
    except Exception:
        return []


def _build_partial_pipe_sql(pipe_sql: str, num_ops: int) -> str | None:
    """Extract the first N pipe operators from a pipe SQL string."""
    parts = pipe_sql.split(" |> ")
    if num_ops >= len(parts) or num_ops < 1:
        return None
    return " |> ".join(parts[:num_ops])


def _make_tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _make_tool_response(call_id: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }


def format_tool_calling_sample(
    db_id: str,
    db_path: str | None,
    question: str,
    question_id: str,
    difficulty: str,
    gold_sql: str,
    pipe_sql: str,
    tables: list[TableSchema],
    rng: random.Random,
) -> ChatSample | None:
    """Build a tool-calling conversation sample.

    Picks pattern A/B/C based on difficulty, scripts tool calls with
    deterministic IDs, simulates responses via tool_executor functions.
    """
    ref_tables = extract_referenced_tables(gold_sql)
    if not ref_tables:
        return None

    # Pick pattern based on difficulty
    pipe_ops = pipe_sql.split(" |> ")
    num_ops = len(pipe_ops)

    if difficulty in ("easy",) or num_ops <= 2:
        pattern = "A"
    elif difficulty in ("medium",) or num_ops <= 4:
        pattern = "B"
    else:
        pattern = "C"

    messages: list[dict] = [{"role": "system", "content": TOOL_SYSTEM_MESSAGE}]
    messages.append({"role": "user", "content": f"Database: {db_id}\nQuestion: {question}"})

    call_counter = 0

    def next_call_id() -> str:
        nonlocal call_counter
        call_counter += 1
        return f"call_{call_counter}"

    # Step 1: list_tables (all patterns)
    cid = next_call_id()
    messages.append({
        "role": "assistant",
        "content": rng.choice(_LIST_TABLES_THOUGHTS),
        "tool_calls": [_make_tool_call(cid, "list_tables", {"db_id": db_id})],
    })
    messages.append(_make_tool_response(cid, list_tables_result(tables)))

    # Step 2: describe_table for referenced tables
    # Pattern A: 1 table, Pattern B: 1-2 tables, Pattern C: 2-3 tables
    if pattern == "A":
        describe_tables = ref_tables[:1]
    elif pattern == "B":
        describe_tables = ref_tables[:2]
    else:
        describe_tables = ref_tables[:3]

    for tname in describe_tables:
        cid = next_call_id()
        thought = rng.choice(_DESCRIBE_TABLE_THOUGHTS).format(table=tname)
        messages.append({
            "role": "assistant",
            "content": thought,
            "tool_calls": [_make_tool_call(cid, "describe_table", {"db_id": db_id, "table_name": tname})],
        })
        messages.append(_make_tool_response(cid, describe_table_result(tables, tname)))

    # Step 3: sample_data (patterns B and C, optional for B)
    if pattern == "B" and rng.random() < 0.5 and db_path:
        sample_table = rng.choice(ref_tables[:2]) if len(ref_tables) > 1 else ref_tables[0]
        cid = next_call_id()
        thought = rng.choice(_SAMPLE_DATA_THOUGHTS).format(table=sample_table)
        messages.append({
            "role": "assistant",
            "content": thought,
            "tool_calls": [_make_tool_call(cid, "sample_data", {"db_id": db_id, "table_name": sample_table, "limit": 5})],
        })
        messages.append(_make_tool_response(cid, sample_data_result(db_path, sample_table, 5)))
    elif pattern == "C" and db_path:
        sample_table = ref_tables[0]
        cid = next_call_id()
        thought = rng.choice(_SAMPLE_DATA_THOUGHTS).format(table=sample_table)
        messages.append({
            "role": "assistant",
            "content": thought,
            "tool_calls": [_make_tool_call(cid, "sample_data", {"db_id": db_id, "table_name": sample_table, "limit": 5})],
        })
        messages.append(_make_tool_response(cid, sample_data_result(db_path, sample_table, 5)))

    # Step 4: execute_pipe_sql partial (pattern C only)
    if pattern == "C" and db_path:
        partial_ops = max(1, num_ops // 2)
        partial_sql = _build_partial_pipe_sql(pipe_sql, partial_ops)
        if partial_sql:
            cid = next_call_id()
            messages.append({
                "role": "assistant",
                "content": rng.choice(_EXECUTE_THOUGHTS),
                "tool_calls": [_make_tool_call(cid, "execute_pipe_sql", {"db_id": db_id, "pipe_sql": partial_sql})],
            })
            messages.append(_make_tool_response(cid, execute_pipe_sql_result(db_path, partial_sql)))

    # Final step: produce the pipe SQL and verify
    if db_path:
        # Produce query + execute to verify
        cid = next_call_id()
        messages.append({
            "role": "assistant",
            "content": f"{rng.choice(_FINAL_THOUGHTS)}\n\n```sql\n{pipe_sql}\n```\n\n{rng.choice(_VERIFY_THOUGHTS)}",
            "tool_calls": [_make_tool_call(cid, "execute_pipe_sql", {"db_id": db_id, "pipe_sql": pipe_sql})],
        })
        messages.append(_make_tool_response(cid, execute_pipe_sql_result(db_path, pipe_sql)))
        messages.append({
            "role": "assistant",
            "content": rng.choice(_CONFIRM_MESSAGES),
        })
    else:
        # No db_path — fall back to syntax validation
        cid = next_call_id()
        messages.append({
            "role": "assistant",
            "content": f"{rng.choice(_FINAL_THOUGHTS)}\n\n```sql\n{pipe_sql}\n```\n\n{rng.choice(_VALIDATE_THOUGHTS)}",
            "tool_calls": [_make_tool_call(cid, "validate_pipe_sql", {"pipe_sql": pipe_sql})],
        })
        messages.append(_make_tool_response(cid, validate_pipe_sql_result(pipe_sql)))
        messages.append({
            "role": "assistant",
            "content": rng.choice(_VALIDATE_CONFIRM_MESSAGES),
        })

    metadata = {
        "question_id": question_id,
        "db_id": db_id,
        "difficulty": difficulty,
        "sample_type": "tool_calling",
        "pattern": pattern,
    }

    return ChatSample(messages=messages, metadata=metadata, tools=TOOL_DEFINITIONS)
