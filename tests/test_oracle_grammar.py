"""Oracle grammar validation and decompiler round-trip tests.

Two test suites:
1. ANTLR grammar conformance — validates Oracle DML against formal PL/SQL grammar
2. Decompiler round-trip — validates Oracle SQL → pipe SQL → Oracle SQL preservation
"""

import os
import unittest

try:
    from tests.oracle_grammar.oracle_validator import validate_oracle_sql, HAS_ANTLR
except ImportError:
    HAS_ANTLR = False

from tests.oracle_grammar.known_gaps import is_known_gap, KNOWN_GAPS
from tests.oracle_grammar.oracle_validation import (
    MismatchCategory,
    Status,
    validate_all,
    validate_one,
    load_fixture,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "oracle_grammar", "oracle_dml_statements.txt"
)


def _load_statements() -> list[str]:
    if not os.path.exists(FIXTURE_PATH):
        return []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Suite 1: ANTLR Grammar Conformance
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_ANTLR, "antlr4-python3-runtime not installed")
class TestOracleGrammarConformance(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.statements = _load_statements()

    def test_dml_grammar_conformance(self):
        """Validate all extracted Oracle DML against the ANTLR PL/SQL grammar."""
        failures = []

        for sql in self.statements:
            result = validate_oracle_sql(sql)
            if not result.is_valid and not is_known_gap(sql):
                failures.append(result)

        total = len(self.statements)
        known = sum(1 for s in self.statements if is_known_gap(s))
        passed = total - len(failures) - known

        msg_parts = [
            f"\nGrammar conformance: {passed}/{total} passed"
            f" ({known} known gaps, {len(failures)} failures)"
        ]
        for f in failures:
            msg_parts.append(f"\n--- FAILURE ---\nSQL: {f.sql}\nErrors:")
            for e in f.errors:
                msg_parts.append(f"  {e}")

        self.assertEqual(len(failures), 0, "\n".join(msg_parts))

    def test_known_gaps_still_fail(self):
        """Ensure known gap entries haven't become stale (i.e., they still fail)."""
        stale = []

        for sql in self.statements:
            gap = is_known_gap(sql)
            if gap:
                result = validate_oracle_sql(sql)
                if result.is_valid:
                    stale.append(f"SQL now passes: {sql[:80]} (gap: {gap.reason})")

        if stale:
            self.fail(
                f"{len(stale)} known gap(s) are stale (SQL now passes grammar):\n"
                + "\n".join(stale)
            )

    def test_fixture_not_empty(self):
        """Ensure the fixture file exists and has statements."""
        self.assertGreater(
            len(self.statements), 0, "oracle_dml_statements.txt is empty or missing"
        )


# ---------------------------------------------------------------------------
# Suite 2: Decompiler Round-Trip Validation
# ---------------------------------------------------------------------------

# Categories that are expected failures due to inherent limitations of pipe SQL
# (these Oracle constructs have no pipe SQL equivalent, or the CTE-based
# round-trip introduces structural changes that are semantically equivalent
# but textually different)
EXPECTED_LOSS_CATEGORIES = {
    MismatchCategory.FOR_UPDATE_LOST,
    MismatchCategory.CONNECT_BY_LOST,
    MismatchCategory.BULK_COLLECT_LOST,
    MismatchCategory.INTO_LOST,
    MismatchCategory.HINTS_LOST,
    MismatchCategory.MATCH_RECOGNIZE_LOST,
    MismatchCategory.FETCH_FIRST_LOST,
    MismatchCategory.SAMPLE_LOST,
    MismatchCategory.CTE_WRAPPING,          # Pipe SQL round-trip adds WITH __tmpN wrappers
    MismatchCategory.DISTINCT_TO_GROUPBY,   # DISTINCT emitted as AGGREGATE GROUP BY
    MismatchCategory.NULLS_ORDER_CHANGED,   # NULLS FIRST/LAST normalization differences
}


class TestOracleDecompilerRoundTrip(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.statements = _load_statements()
        cls.summary = validate_all(cls.statements)

    def test_no_decompile_errors(self):
        """No Oracle DML statement should crash the decompiler."""
        errors = [r for r in self.summary.records if r.status == Status.DECOMPILE_ERROR]
        self.assertEqual(
            len(errors),
            0,
            f"{len(errors)} decompile error(s):\n"
            + "\n".join(f"  {r.original_sql[:80]}\n    {r.error}" for r in errors),
        )

    def test_roundtrip_parse_errors_within_budget(self):
        """Round-trip parse errors should not exceed the known count.

        Current: 1 known (FETCH NEXT with NVL bind-variable expression).
        """
        errors = [r for r in self.summary.records if r.status == Status.ROUNDTRIP_PARSE_ERROR]
        self.assertLessEqual(
            len(errors),
            1,
            f"{len(errors)} round-trip parse error(s) (budget: 1):\n"
            + "\n".join(
                f"  SQL:  {r.original_sql[:70]}\n  PIPE: {r.pipe_sql[:70]}\n    {r.error}"
                for r in errors
            ),
        )

    def test_no_unexpected_mismatches(self):
        """Mismatches should only be in expected-loss categories (no regressions)."""
        unexpected = [
            r
            for r in self.summary.records
            if r.status == Status.MISMATCH
            and r.mismatch_category not in EXPECTED_LOSS_CATEGORIES
        ]
        if unexpected:
            msgs = []
            for r in unexpected:
                cat = r.mismatch_category.value if r.mismatch_category else "?"
                msgs.append(
                    f"  [{cat}] {r.original_sql[:70]}\n"
                    f"    PIPE: {r.pipe_sql[:70]}\n"
                    f"    RT:   {r.round_tripped_sql[:70]}"
                )
            self.fail(
                f"{len(unexpected)} unexpected mismatch(es) "
                f"(not in expected-loss categories):\n" + "\n".join(msgs)
            )

    def test_match_rate_above_threshold(self):
        """Exact round-trip match rate should stay above 38%."""
        self.assertGreaterEqual(
            self.summary.match_rate,
            0.38,
            f"Match rate {self.summary.match_rate:.1%} is below 38% threshold.\n"
            f"{self.summary}",
        )

    def test_summary_report(self):
        """Print the full validation summary (always passes, for visibility)."""
        print(f"\n{self.summary}")


if __name__ == "__main__":
    unittest.main()
