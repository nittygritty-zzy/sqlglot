"""Decompose pipe SQL queries into per-operator trajectory steps."""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot

from pipe_sql.decompiler.emitter import emit_pipe_query
from pipe_sql.decompiler.result import PipeOpType, PipeQuery
from pipe_sql.decompiler.serializer import serialize


@dataclass
class TrajectoryStep:
    query_so_far: str
    operator_text: str
    op_type: str
    step_index: int
    total_steps: int
    cte_prefix: str


def _serialize_partial(pipe_query: PipeQuery, up_to: int) -> str:
    """Serialize operators [0..up_to) of a PipeQuery."""
    parts = []
    for i in range(up_to):
        op = pipe_query.operators[i]
        if i == 0:
            parts.append(op.sql_fragment)
        else:
            parts.append(f"|> {op.sql_fragment}")
    return " ".join(parts)


def _build_cte_prefix(pipe_query: PipeQuery) -> str:
    """Build the CTE prefix string if CTEs exist."""
    if not pipe_query.ctes:
        return ""
    cte_parts = []
    for name, cte_query in pipe_query.ctes:
        cte_body = serialize(cte_query)
        cte_parts.append(f"{name} AS ({cte_body})")
    return "WITH " + ", ".join(cte_parts)


def _decompose_from_pipe_query(pipe_query: PipeQuery) -> list[TrajectoryStep]:
    """Decompose a PipeQuery object into trajectory steps."""
    cte_prefix = _build_cte_prefix(pipe_query)
    total = len(pipe_query.operators)
    steps = []

    for i, op in enumerate(pipe_query.operators):
        query_so_far = _serialize_partial(pipe_query, i)
        if cte_prefix and query_so_far:
            query_so_far = f"{cte_prefix} {query_so_far}"
        elif cte_prefix:
            query_so_far = ""

        if i == 0:
            operator_text = op.sql_fragment
        else:
            operator_text = f"|> {op.sql_fragment}"

        steps.append(TrajectoryStep(
            query_so_far=query_so_far,
            operator_text=operator_text,
            op_type=op.op_type.name,
            step_index=i,
            total_steps=total,
            cte_prefix=cte_prefix,
        ))

    return steps


def _decompose_fallback(pipe_sql: str) -> list[TrajectoryStep]:
    """Fallback: split pipe_sql string on ' |> ' delimiter."""
    parts = pipe_sql.split(" |> ")
    total = len(parts)
    steps = []

    for i, part in enumerate(parts):
        if i == 0:
            query_so_far = ""
            operator_text = part
        else:
            query_so_far = " |> ".join(parts[:i])
            operator_text = f"|> {part}"

        # Guess op_type from the fragment text
        first_word = part.strip().split()[0].upper() if part.strip() else "UNKNOWN"
        op_type = first_word if first_word in {t.name for t in PipeOpType} else "UNKNOWN"

        steps.append(TrajectoryStep(
            query_so_far=query_so_far,
            operator_text=operator_text,
            op_type=op_type,
            step_index=i,
            total_steps=total,
            cte_prefix="",
        ))

    return steps


def decompose_trajectory(
    gold_sql: str,
    pipe_sql: str,
    dialect: str = "sqlite",
) -> list[TrajectoryStep]:
    """Decompose a pipe query into per-operator trajectory steps.

    Tries to re-emit from gold_sql AST for structured decomposition.
    Falls back to string splitting if that fails.
    """
    try:
        ast = sqlglot.parse_one(gold_sql, dialect=dialect)
        pipe_query = emit_pipe_query(ast, dialect=dialect)
        steps = _decompose_from_pipe_query(pipe_query)
        if steps:
            return steps
    except Exception:
        pass

    return _decompose_fallback(pipe_sql)
