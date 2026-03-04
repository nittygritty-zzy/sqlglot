"""Terminal rule: emit ORDER BY and LIMIT pipe operators."""

from __future__ import annotations

import re

from sqlglot import exp

from ..result import PipeOperator, PipeOpType


def _strip_table_qualifiers(sql: str) -> str:
    """Strip table qualifiers from column references (T2.col -> col)."""
    return re.sub(r"\b(\w+)\.(\w+)\b", r"\2", sql)


def emit(
    ast: exp.Select,
    dialect: str = "sqlite",
    strip_qualifiers: bool = True,
    select_alias_map: dict[str, str] | None = None,
) -> list[PipeOperator]:
    """Emit terminal operators: ORDER BY, LIMIT, OFFSET.

    strip_qualifiers: when True, strip table qualifiers from ORDER BY
    (needed when ORDER BY follows a SELECT that wraps in a CTE).
    select_alias_map: mapping from original column name (upper) to alias,
    for substituting aliased column refs in ORDER BY after CTE wrapping.
    """
    operators = []
    operators.extend(
        emit_order_only(
            ast,
            dialect=dialect,
            strip_qualifiers=strip_qualifiers,
            select_alias_map=select_alias_map,
        )
    )
    operators.extend(emit_limit_only(ast, dialect=dialect))
    return operators


def _apply_alias_map(
    order_exprs: list[exp.Expression], alias_map: dict[str, str], dialect: str = "sqlite"
) -> list[exp.Expression]:
    """Replace expressions in ORDER BY using the alias map. Returns possibly modified copies.

    Handles both column-level replacements (I_BRAND_ID → brand_id) and
    expression-level replacements (SUBSTRING(W, 1, 20) → _col_0).
    """
    result = []
    for order_expr in order_exprs:
        inner = order_expr.this if isinstance(order_expr, exp.Ordered) else order_expr
        inner_sql = inner.sql(dialect=dialect).upper()
        # Also try stripped version
        stripped = inner.copy()
        for col in stripped.find_all(exp.Column):
            col.set("table", None)
        stripped_sql = stripped.sql(dialect=dialect).upper()

        if inner_sql in alias_map:
            # Replace entire expression with alias reference
            new_inner = exp.column(alias_map[inner_sql])
            if isinstance(order_expr, exp.Ordered):
                new_order = order_expr.copy()
                new_order.set("this", new_inner)
                result.append(new_order)
            else:
                result.append(new_inner)
        elif stripped_sql in alias_map:
            new_inner = exp.column(alias_map[stripped_sql])
            if isinstance(order_expr, exp.Ordered):
                new_order = order_expr.copy()
                new_order.set("this", new_inner)
                result.append(new_order)
            else:
                result.append(new_inner)
        else:
            # Fall back to column-level replacement
            new_expr = order_expr.copy()
            for col in new_expr.find_all(exp.Column):
                col_name = col.name.upper()
                if col_name in alias_map and not col.table:
                    col.this.args["this"] = alias_map[col_name]
            result.append(new_expr)
    return result


def emit_order_only(
    ast: exp.Select,
    dialect: str = "sqlite",
    strip_qualifiers: bool = False,
    select_alias_map: dict[str, str] | None = None,
) -> list[PipeOperator]:
    """Emit only ORDER BY operator."""
    order = ast.args.get("order")
    if not order:
        return []

    if select_alias_map:
        order_copies = _apply_alias_map(
            [e.copy() for e in order.expressions], select_alias_map, dialect=dialect
        )
        order_parts = [e.sql(dialect=dialect) for e in order_copies]
    else:
        order_parts = [e.sql(dialect=dialect) for e in order.expressions]
    order_str = "ORDER BY " + ", ".join(order_parts)

    if strip_qualifiers:
        order_str = _strip_table_qualifiers(order_str)

    return [PipeOperator(op_type=PipeOpType.ORDER_BY, sql_fragment=order_str)]


def emit_limit_only(ast: exp.Select, dialect: str = "sqlite") -> list[PipeOperator]:
    """Emit only LIMIT/OFFSET operators."""
    limit = ast.args.get("limit")
    offset = ast.args.get("offset")

    if not limit and not offset:
        return []

    if limit:
        # exp.Fetch (Oracle/TSQL FETCH FIRST N ROWS ONLY) stores count in "count",
        # while exp.Limit stores it in "expression"
        if isinstance(limit, exp.Fetch):
            count = limit.args.get("count")
        else:
            count = limit.expression

        limit_str = f"LIMIT {count.sql(dialect=dialect)}"
        if offset:
            limit_str += f" OFFSET {offset.expression.sql(dialect=dialect)}"
        return [PipeOperator(op_type=PipeOpType.LIMIT, sql_fragment=limit_str)]

    return [
        PipeOperator(
            op_type=PipeOpType.LIMIT,
            sql_fragment=f"LIMIT ALL OFFSET {offset.expression.sql(dialect=dialect)}",
        )
    ]
