"""Core ANTLR-based Oracle SQL grammar validator."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

try:
    from antlr4 import CommonTokenStream, InputStream
    from antlr4.error.ErrorListener import ErrorListener

    from tests.oracle_grammar.generated.PlSqlLexer import PlSqlLexer
    from tests.oracle_grammar.generated.PlSqlParser import PlSqlParser

    HAS_ANTLR = True
except ImportError:
    HAS_ANTLR = False


@dataclass
class GrammarValidationResult:
    sql: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_valid:
            return f"VALID: {self.sql[:80]}"
        return f"INVALID: {self.sql[:80]}\n  " + "\n  ".join(self.errors)


class _CollectingErrorListener(ErrorListener):
    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(
        self,
        recognizer: t.Any,
        offendingSymbol: t.Any,
        line: int,
        column: int,
        msg: str,
        e: t.Any,
    ) -> None:
        self.errors.append(f"line {line}:{column} {msg}")


def validate_oracle_sql(sql: str) -> GrammarValidationResult:
    """Parse a single SQL statement through the ANTLR Oracle PL/SQL grammar.

    Appends a semicolon if missing (the grammar expects statement terminators).
    Uses `sql_script` as the entry rule.
    """
    if not HAS_ANTLR:
        raise RuntimeError("antlr4-python3-runtime is not installed")

    normalized = sql.strip()
    if not normalized.endswith(";"):
        normalized += ";"

    input_stream = InputStream(normalized)
    lexer = PlSqlLexer(input_stream)
    lexer.removeErrorListeners()
    lex_errors = _CollectingErrorListener()
    lexer.addErrorListener(lex_errors)

    token_stream = CommonTokenStream(lexer)
    parser = PlSqlParser(token_stream)
    parser.removeErrorListeners()
    parse_errors = _CollectingErrorListener()
    parser.addErrorListener(parse_errors)

    parser.sql_script()

    all_errors = lex_errors.errors + parse_errors.errors
    return GrammarValidationResult(
        sql=sql,
        is_valid=len(all_errors) == 0,
        errors=all_errors,
    )
