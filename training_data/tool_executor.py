"""Simulate tool execution against SQLite databases for tool-calling training data."""

from __future__ import annotations

import json
import sqlite3

import sqlglot

from training_data.schema_extractor import TableSchema, format_schema_compact


def list_tables_result(tables: list[TableSchema]) -> str:
    """Return JSON list of table names."""
    return json.dumps([t.name for t in tables])


def describe_table_result(tables: list[TableSchema], table_name: str) -> str:
    """Return compact schema for one table."""
    matching = [t for t in tables if t.name.lower() == table_name.lower()]
    if not matching:
        return f"Error: table '{table_name}' not found."
    return format_schema_compact(matching)


def sample_data_result(db_path: str, table_name: str, limit: int = 5) -> str:
    """Query SQLite and return formatted sample rows."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {limit}")
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        conn.close()
    except Exception as e:
        return f"Error: {e}"

    if not rows:
        return f"{' | '.join(cols)}\n(no rows)"

    lines = [" | ".join(cols)]
    for row in rows:
        lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
    return "\n".join(lines)


def execute_pipe_sql_result(db_path: str, pipe_sql: str) -> str:
    """Transpile pipe SQL to SQLite via sqlglot, execute, and return results."""
    try:
        transpiled = sqlglot.transpile(pipe_sql, read="sqlite", write="sqlite")[0]
    except Exception as e:
        return f"Transpile error: {e}"

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(transpiled)
        rows = cursor.fetchmany(10)
        cols = [desc[0] for desc in cursor.description]
        conn.close()
    except Exception as e:
        return f"Execution error: {e}"

    lines = [" | ".join(cols)]
    for row in rows:
        lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
    if len(rows) == 10:
        lines.append("... (truncated)")
    return "\n".join(lines)


def validate_pipe_sql_result(pipe_sql: str) -> str:
    """Transpile pipe SQL to check syntax validity."""
    try:
        sqlglot.transpile(pipe_sql, read="sqlite", write="sqlite")
        return "Valid syntax."
    except Exception as e:
        return f"Syntax error: {e}"
