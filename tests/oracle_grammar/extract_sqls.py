"""Extract Oracle DML statements from test_oracle.py.

Parses the test file to find SQL strings from validate_identity() and validate_all()
calls, runs each through sqlglot parse→generate to get the actual Oracle output,
filters to DML only, deduplicates, and writes to oracle_dml_statements.txt.

Usage:
    python -m tests.oracle_grammar.extract_sqls
"""

from __future__ import annotations

import ast
import os
import sys

import sqlglot
from sqlglot import exp

DML_PREFIXES = ("SELECT", "INSERT", "UPDATE", "DELETE", "MERGE", "WITH")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ORACLE_PATH = os.path.join(SCRIPT_DIR, "..", "dialects", "test_oracle.py")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "oracle_dml_statements.txt")


def _extract_string_arg(node: ast.expr) -> str | None:
    """Extract a string literal from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return None  # f-strings
    return None


def _extract_sqls_from_source(source: str) -> list[str]:
    """Parse test_oracle.py and extract SQL strings from validate_identity/validate_all calls."""
    tree = ast.parse(source)
    sqls: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        method_name = None
        if isinstance(func, ast.Attribute):
            method_name = func.attr

        if method_name == "validate_identity":
            if node.args:
                sql = _extract_string_arg(node.args[0])
                if sql:
                    sqls.append(sql)
                    # Also extract write_sql if provided
                    if len(node.args) > 1:
                        write_sql = _extract_string_arg(node.args[1])
                        if write_sql:
                            sqls.append(write_sql)
                    for kw in node.keywords:
                        if kw.arg == "write_sql":
                            write_sql = _extract_string_arg(kw.value)
                            if write_sql:
                                sqls.append(write_sql)

        elif method_name == "validate_all":
            if node.args:
                sql = _extract_string_arg(node.args[0])
                if sql:
                    sqls.append(sql)
            for kw in node.keywords:
                if kw.arg == "write" and isinstance(kw.value, ast.Dict):
                    for key, val in zip(kw.value.keys, kw.value.values):
                        key_str = _extract_string_arg(key) if key else None
                        val_str = _extract_string_arg(val) if val else None
                        if key_str == "oracle" and val_str:
                            sqls.append(val_str)

    return sqls


def _to_oracle_output(sql: str) -> str | None:
    """Parse SQL with Oracle dialect and regenerate as Oracle output."""
    try:
        parsed = sqlglot.parse_one(sql, read="oracle")
        return parsed.sql(dialect="oracle")
    except Exception:
        return None


def _is_dml(sql: str) -> bool:
    """Check if a SQL string is a DML statement."""
    upper = sql.strip().upper()
    return any(upper.startswith(p) for p in DML_PREFIXES)


def extract_and_write() -> list[str]:
    """Main extraction: read test file, extract, transform, filter, deduplicate, write."""
    with open(TEST_ORACLE_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    raw_sqls = _extract_sqls_from_source(source)
    print(f"Extracted {len(raw_sqls)} raw SQL strings from test_oracle.py")

    oracle_outputs: list[str] = []
    seen: set[str] = set()

    for sql in raw_sqls:
        output = _to_oracle_output(sql)
        if output and _is_dml(output) and output not in seen:
            seen.add(output)
            oracle_outputs.append(output)

    print(f"Filtered to {len(oracle_outputs)} unique DML statements")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sql in oracle_outputs:
            f.write(sql + "\n")

    print(f"Written to {OUTPUT_PATH}")
    return oracle_outputs


if __name__ == "__main__":
    extract_and_write()
