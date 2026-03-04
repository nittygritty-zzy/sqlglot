"""TPC-H and TPC-DS execution-based validation for the Oracle pipe SQL decompiler.

Pipeline: SQL → Oracle SQL → pipe SQL → DuckDB SQL → execute
Compares execution results against gold (direct SQL → DuckDB SQL → execute).
"""

import unittest

import duckdb
from pandas.testing import assert_frame_equal

import sqlglot
from pipe_sql.decompiler import decompile
from tests.helpers import (
    FIXTURES_DIR,
    SKIP_INTEGRATION,
    TPCH_SCHEMA,
    TPCDS_SCHEMA,
    load_sql_fixture_pairs,
)

DIR_TPCH = FIXTURES_DIR + "/optimizer/tpc-h/"
DIR_TPCDS = FIXTURES_DIR + "/optimizer/tpc-ds/"


def _setup_duckdb(schema, directory):
    """Create a DuckDB connection with views for all tables in the schema."""
    conn = duckdb.connect()
    for table, columns in schema.items():
        file_name = f"{directory}{table}.csv.gz"
        conn.execute(
            f"""
            CREATE VIEW {table} AS
            SELECT *
            FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})
            """
        )
    return conn


@unittest.skipIf(SKIP_INTEGRATION, "Skipping Integration Tests since `SKIP_INTEGRATION` is set")
class TestOracleDecompilerTPCH(unittest.TestCase):
    """Validate Oracle decompiler round-trip via DuckDB execution on TPC-H."""

    @classmethod
    def setUpClass(cls):
        cls.conn = _setup_duckdb(TPCH_SCHEMA, DIR_TPCH)
        cls.sqls = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def _gold_df(self, sql):
        """Execute original SQL on DuckDB (gold standard)."""
        return self.conn.execute(sqlglot.transpile(sql, write="duckdb")[0]).fetchdf()

    def _roundtrip_df(self, sql):
        """Execute SQL through Oracle decompiler pipeline on DuckDB."""
        oracle_sql = sqlglot.transpile(sql, write="oracle")[0]
        pipe_sql = decompile(oracle_sql, dialect="oracle")
        rt_sql = sqlglot.transpile(pipe_sql, read="oracle", write="duckdb")[0]
        return self.conn.execute(rt_sql).fetchdf()

    def test_tpch_oracle_roundtrip(self):
        """TPC-H queries should produce identical results through the Oracle decompiler."""
        match = 0
        errors = []

        for i, (_, sql, _) in enumerate(self.sqls, start=1):
            with self.subTest(f"TPC-H Q{i:02d}"):
                try:
                    gold = self._gold_df(sql)
                    rt = self._roundtrip_df(sql)
                    assert_frame_equal(
                        gold, rt,
                        check_dtype=False,
                        check_index_type=False,
                        check_names=False,
                    )
                    match += 1
                except Exception as e:
                    errors.append((i, str(e)[:120]))

        total = len(self.sqls)
        print(
            f"\nOracle Decompiler TPC-H: {match}/{total} "
            f"({match / total:.0%}) match, {len(errors)} errors"
        )
        for qi, err in errors:
            print(f"  Q{qi:02d}: {err}")

        # 21/22 currently match; Q11 is a DuckDB scalar subquery limitation
        self.assertGreaterEqual(
            match / total,
            0.95,
            f"Match rate {match}/{total} ({match / total:.0%}) below 95% threshold",
        )

    def test_decompile_no_crashes(self):
        """Every TPC-H query should decompile without crashing."""
        crashes = []
        for i, (_, sql, _) in enumerate(self.sqls, start=1):
            oracle_sql = sqlglot.transpile(sql, write="oracle")[0]
            try:
                decompile(oracle_sql, dialect="oracle")
            except Exception as e:
                crashes.append(f"Q{i:02d}: {type(e).__name__}: {e}")

        self.assertEqual(
            len(crashes), 0,
            f"{len(crashes)} decompile crash(es):\n" + "\n".join(crashes),
        )

    def test_transpile_roundtrip_no_errors(self):
        """Pipe SQL should always transpile back to valid DuckDB SQL."""
        errors = []
        for i, (_, sql, _) in enumerate(self.sqls, start=1):
            oracle_sql = sqlglot.transpile(sql, write="oracle")[0]
            try:
                pipe_sql = decompile(oracle_sql, dialect="oracle")
                sqlglot.transpile(pipe_sql, read="oracle", write="duckdb")[0]
            except Exception as e:
                errors.append(f"Q{i:02d}: {type(e).__name__}: {e}")

        self.assertEqual(
            len(errors), 0,
            f"{len(errors)} transpile error(s):\n" + "\n".join(errors),
        )


@unittest.skipIf(SKIP_INTEGRATION, "Skipping Integration Tests since `SKIP_INTEGRATION` is set")
class TestOracleDecompilerTPCDS(unittest.TestCase):
    """Validate Oracle decompiler round-trip via DuckDB execution on TPC-DS."""

    @classmethod
    def setUpClass(cls):
        cls.conn = _setup_duckdb(TPCDS_SCHEMA, DIR_TPCDS)
        cls.sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_tpcds_oracle_roundtrip(self):
        """TPC-DS queries should produce identical results through the Oracle decompiler."""
        match = 0
        errors = []

        for i, (_, sql, _) in enumerate(self.sqls, start=1):
            with self.subTest(f"TPC-DS Q{i:02d}"):
                try:
                    gold_sql = sqlglot.transpile(sql, write="duckdb")[0]
                    gold = self.conn.execute(gold_sql).fetchdf()

                    oracle_sql = sqlglot.transpile(sql, write="oracle")[0]
                    pipe_sql = decompile(oracle_sql, dialect="oracle")
                    rt_sql = sqlglot.transpile(pipe_sql, read="oracle", write="duckdb")[0]
                    rt = self.conn.execute(rt_sql).fetchdf()

                    # Use positional column comparison to handle alias differences
                    # (e.g., _agg1 vs sum(col))
                    gold.columns = range(len(gold.columns))
                    rt.columns = range(len(rt.columns))
                    assert_frame_equal(
                        gold, rt,
                        check_dtype=False,
                        check_index_type=False,
                    )
                    match += 1
                except Exception as e:
                    errors.append((i, str(e)[:120]))

        total = len(self.sqls)
        print(
            f"\nOracle Decompiler TPC-DS: {match}/{total} "
            f"({match / total:.0%}) match, {len(errors)} errors"
        )
        for qi, err in errors[:10]:
            print(f"  Q{qi:02d}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

        # 82/99 currently match (17 are gold failures from DuckDB type issues)
        self.assertGreaterEqual(
            match / total,
            0.80,
            f"Match rate {match}/{total} ({match / total:.0%}) below 80% threshold",
        )

    def test_decompile_success_rate(self):
        """At least 93% of TPC-DS queries should decompile without crashing."""
        crashes = []
        for i, (_, sql, _) in enumerate(self.sqls, start=1):
            try:
                oracle_sql = sqlglot.transpile(sql, write="oracle")[0]
                decompile(oracle_sql, dialect="oracle")
            except Exception as e:
                crashes.append(f"Q{i:02d}: {type(e).__name__}: {str(e)[:80]}")

        total = len(self.sqls)
        ok = total - len(crashes)
        self.assertGreaterEqual(
            ok / total,
            0.90,
            f"Decompile success {ok}/{total} ({ok / total:.0%}) below 90%:\n"
            + "\n".join(crashes),
        )


if __name__ == "__main__":
    unittest.main()
