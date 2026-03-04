"""Known discrepancies between SQLGlot Oracle output and the ANTLR PL/SQL grammar.

Each entry maps a SQL string (or substring pattern) to a category and reason.
Categories:
  - ANTLR_TOO_STRICT: Grammar rejects valid Oracle SQL
  - SQLGLOT_DEVIATION: SQLGlot generates non-standard but functional Oracle SQL
  - NOT_DML: Statement type not in scope for DML validation
"""

from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass


@dataclass
class KnownGap:
    pattern: str  # regex pattern matched against the SQL
    category: str
    reason: str

    def matches(self, sql: str) -> bool:
        return bool(re.search(self.pattern, sql, re.IGNORECASE))


KNOWN_GAPS: list[KnownGap] = [
    # ANTLR grammar limitations — valid Oracle SQL rejected by the community grammar
    KnownGap(
        pattern=r"KEEP\s*\(\s*DENSE_RANK",
        category="ANTLR_TOO_STRICT",
        reason="ANTLR grammar lacks aggregate KEEP (DENSE_RANK FIRST/LAST) clause",
    ),
    KnownGap(
        pattern=r"UNPIVOT\b.*\|\|",
        category="ANTLR_TOO_STRICT",
        reason="ANTLR grammar does not allow concatenation expressions in UNPIVOT alias",
    ),
    KnownGap(
        pattern=r"\bJSON_OBJECTAGG\b",
        category="ANTLR_TOO_STRICT",
        reason="ANTLR grammar does not include JSON_OBJECTAGG (Oracle 12.2+)",
    ),
    KnownGap(
        pattern=r"\bMATCH_RECOGNIZE\b",
        category="ANTLR_TOO_STRICT",
        reason="ANTLR grammar does not support MATCH_RECOGNIZE (Oracle 12c)",
    ),
    KnownGap(
        pattern=r"(?:SUM|AVG|MIN|MAX|COUNT)\s*\(.*\)\s+OVER\s*\(",
        category="ANTLR_TOO_STRICT",
        reason="ANTLR grammar rejects some window function syntaxes with aggregate functions",
    ),
    KnownGap(
        pattern=r",\s*(?:MIN|MAX)\s*\([^)]+\)(?:\s*,|\s+FROM)",
        category="ANTLR_TOO_STRICT",
        reason="ANTLR grammar rejects MIN/MAX as subsequent select expressions",
    ),
    KnownGap(
        pattern=r"BETWEEN\s*\(\s*SELECT\b",
        category="ANTLR_TOO_STRICT",
        reason="ANTLR grammar rejects subqueries in BETWEEN clause",
    ),
]


def is_known_gap(sql: str) -> t.Optional[KnownGap]:
    """Return the matching KnownGap if the SQL is a known discrepancy, else None."""
    for gap in KNOWN_GAPS:
        if gap.matches(sql):
            return gap
    return None
