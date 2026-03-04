"""Oracle round-trip validation: Oracle SQL → pipe SQL → Oracle SQL.

Validates semantic preservation of the decompiler for Oracle dialect.
The round-trip is:
    1. Parse Oracle SQL into AST
    2. Decompile AST into pipe SQL  (pipe_decompiler)
    3. Transpile pipe SQL back to Oracle SQL  (sqlglot.transpile)
    4. Compare original vs round-tripped Oracle SQL

Since we don't have an Oracle DB for execution, we compare normalized ASTs.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from enum import Enum, auto

import sqlglot
from sqlglot import exp

from pipe_sql.decompiler import decompile


class Status(Enum):
    MATCH = auto()
    MISMATCH = auto()
    DECOMPILE_ERROR = auto()
    ROUNDTRIP_PARSE_ERROR = auto()


class MismatchCategory(Enum):
    FOR_UPDATE_LOST = "for_update_lost"
    CONNECT_BY_LOST = "connect_by_lost"
    BULK_COLLECT_LOST = "bulk_collect_lost"
    INTO_LOST = "into_lost"
    HINTS_LOST = "hints_lost"
    NULLS_ORDER_CHANGED = "nulls_order_changed"
    DISTINCT_TO_GROUPBY = "distinct_to_groupby"
    CTE_WRAPPING = "cte_wrapping"
    MATCH_RECOGNIZE_LOST = "match_recognize_lost"
    FETCH_FIRST_LOST = "fetch_first_lost"
    SAMPLE_LOST = "sample_lost"
    OTHER = "other"


@dataclass
class ValidationRecord:
    original_sql: str
    pipe_sql: str = ""
    round_tripped_sql: str = ""
    status: Status = Status.MATCH
    error: str = ""
    mismatch_category: MismatchCategory | None = None

    @property
    def ok(self) -> bool:
        return self.status == Status.MATCH


@dataclass
class ValidationSummary:
    total: int = 0
    match: int = 0
    mismatch: int = 0
    decompile_error: int = 0
    roundtrip_error: int = 0
    mismatch_breakdown: dict[str, int] = field(default_factory=dict)
    records: list[ValidationRecord] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.match / self.total if self.total else 0.0

    def __str__(self) -> str:
        lines = [
            f"Oracle Decompiler Validation: {self.match}/{self.total} "
            f"({self.match_rate:.1%}) round-trip match",
            f"  Mismatch:        {self.mismatch}",
            f"  Decompile error: {self.decompile_error}",
            f"  Roundtrip error: {self.roundtrip_error}",
        ]
        if self.mismatch_breakdown:
            lines.append("  Mismatch breakdown:")
            for cat, count in sorted(self.mismatch_breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat}: {count}")
        return "\n".join(lines)


def _normalize_sql(sql: str) -> str:
    """Normalize Oracle SQL for comparison by parsing and regenerating."""
    ast = sqlglot.parse_one(sql, dialect="oracle")
    return ast.sql(dialect="oracle")


def _classify_mismatch(sql: str, pipe_sql: str, rt_sql: str) -> MismatchCategory:
    """Classify a round-trip mismatch by its root cause."""
    upper = sql.upper()

    if "FOR UPDATE" in upper:
        return MismatchCategory.FOR_UPDATE_LOST
    if "START WITH" in upper or "CONNECT BY" in upper:
        return MismatchCategory.CONNECT_BY_LOST
    if "BULK COLLECT" in upper:
        return MismatchCategory.BULK_COLLECT_LOST
    if upper.startswith("SELECT") and " INTO " in upper and "INSERT" not in upper:
        return MismatchCategory.INTO_LOST
    if "/*+" in sql:
        return MismatchCategory.HINTS_LOST
    if "NULLS FIRST" in upper or "NULLS LAST" in upper:
        return MismatchCategory.NULLS_ORDER_CHANGED
    if "DISTINCT" in upper and "GROUP BY" in rt_sql.upper():
        return MismatchCategory.DISTINCT_TO_GROUPBY
    if "MATCH_RECOGNIZE" in upper:
        return MismatchCategory.MATCH_RECOGNIZE_LOST
    if "FETCH" in upper and ("FIRST" in upper or "NEXT" in upper):
        return MismatchCategory.FETCH_FIRST_LOST
    if "SAMPLE" in upper:
        return MismatchCategory.SAMPLE_LOST
    if "__tmp" in rt_sql:
        return MismatchCategory.CTE_WRAPPING
    return MismatchCategory.OTHER


def validate_one(sql: str) -> ValidationRecord:
    """Validate a single Oracle SQL statement through the decompiler round-trip."""
    record = ValidationRecord(original_sql=sql)

    # Step 1: Decompile Oracle SQL → pipe SQL
    try:
        record.pipe_sql = decompile(sql, dialect="oracle")
    except Exception as ex:
        record.status = Status.DECOMPILE_ERROR
        record.error = f"{type(ex).__name__}: {ex}"
        return record

    # Step 2: Transpile pipe SQL → Oracle SQL
    # Use read="oracle" so Oracle-specific syntax in pipe fragments is parsed correctly
    try:
        record.round_tripped_sql = sqlglot.transpile(
            record.pipe_sql, read="oracle", write="oracle"
        )[0]
    except Exception as ex:
        record.status = Status.ROUNDTRIP_PARSE_ERROR
        record.error = f"{type(ex).__name__}: {ex}"
        return record

    # Step 3: Compare normalized SQL
    try:
        orig_norm = _normalize_sql(sql)
        rt_norm = _normalize_sql(record.round_tripped_sql)
    except Exception as ex:
        record.status = Status.ROUNDTRIP_PARSE_ERROR
        record.error = f"Normalization error: {ex}"
        return record

    if orig_norm == rt_norm:
        record.status = Status.MATCH
    else:
        record.status = Status.MISMATCH
        record.mismatch_category = _classify_mismatch(sql, record.pipe_sql, record.round_tripped_sql)

    return record


def validate_all(statements: list[str]) -> ValidationSummary:
    """Validate all Oracle SQL statements and return a summary."""
    summary = ValidationSummary(total=len(statements))

    for sql in statements:
        record = validate_one(sql)
        summary.records.append(record)

        if record.status == Status.MATCH:
            summary.match += 1
        elif record.status == Status.MISMATCH:
            summary.mismatch += 1
            cat = record.mismatch_category.value if record.mismatch_category else "other"
            summary.mismatch_breakdown[cat] = summary.mismatch_breakdown.get(cat, 0) + 1
        elif record.status == Status.DECOMPILE_ERROR:
            summary.decompile_error += 1
        elif record.status == Status.ROUNDTRIP_PARSE_ERROR:
            summary.roundtrip_error += 1

    return summary


def load_fixture(path: str | None = None) -> list[str]:
    """Load Oracle DML statements from the fixture file."""
    import os

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "oracle_dml_statements.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
