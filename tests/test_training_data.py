"""Tests for the training data generation pipeline."""

from __future__ import annotations

import json
import os
import random
import sqlite3
import tempfile
import unittest

from training_data.formatter import SYSTEM_MESSAGE, ChatSample, format_step, format_trajectory
from training_data.schema_extractor import (
    TableSchema,
    ColumnInfo,
    ForeignKey,
    extract_db_schema,
    format_schema_compact,
    build_schema_cache,
    build_tables_cache,
    build_db_path_cache,
)
from training_data.tool_executor import (
    list_tables_result,
    describe_table_result,
    sample_data_result,
    execute_pipe_sql_result,
    validate_pipe_sql_result,
)
from training_data.tool_formatter import (
    TOOL_DEFINITIONS,
    TOOL_SYSTEM_MESSAGE,
    extract_referenced_tables,
    format_tool_calling_sample,
)
from training_data.trajectory import (
    TrajectoryStep,
    decompose_trajectory,
    _decompose_fallback,
)
from training_data.writer import write_output


class TestSchemaExtractor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a test SQLite database
        db_dir = os.path.join(self.tmpdir, "test_db")
        os.makedirs(db_dir)
        self.db_path = os.path.join(db_dir, "test_db.sqlite")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE singer ("
            "singer_id INTEGER PRIMARY KEY, "
            "name TEXT NOT NULL, "
            "country VARCHAR(100), "
            "age INTEGER)"
        )
        conn.execute(
            "CREATE TABLE concert ("
            "concert_id INTEGER PRIMARY KEY, "
            "concert_name TEXT, "
            "stadium_id INTEGER, "
            "FOREIGN KEY (stadium_id) REFERENCES singer(singer_id))"
        )
        conn.commit()
        conn.close()

    def test_extract_db_schema(self):
        tables = extract_db_schema(self.db_path)
        self.assertEqual(len(tables), 2)

        singer = tables[1]  # sorted by name
        self.assertEqual(singer.name, "singer")
        self.assertEqual(len(singer.columns), 4)
        self.assertTrue(singer.columns[0].is_pk)
        self.assertEqual(singer.columns[0].name, "singer_id")

    def test_type_normalization(self):
        tables = extract_db_schema(self.db_path)
        singer = next(t for t in tables if t.name == "singer")
        country_col = next(c for c in singer.columns if c.name == "country")
        # VARCHAR(100) should be normalized to TEXT
        self.assertEqual(country_col.col_type, "TEXT")

    def test_format_schema_compact(self):
        tables = extract_db_schema(self.db_path)
        schema_str = format_schema_compact(tables)
        self.assertIn("singer(singer_id INTEGER PK", schema_str)
        self.assertIn("concert(concert_id INTEGER PK", schema_str)
        self.assertIn("FK->singer.singer_id", schema_str)

    def test_build_schema_cache(self):
        cache = build_schema_cache([self.tmpdir])
        self.assertIn("test_db", cache)
        self.assertIn("singer", cache["test_db"])

    def test_build_schema_cache_missing_dir(self):
        cache = build_schema_cache(["/nonexistent/path"])
        self.assertEqual(cache, {})


class TestTrajectory(unittest.TestCase):
    def test_decompose_simple_query(self):
        gold_sql = "SELECT count(*) FROM singer"
        pipe_sql = "FROM singer |> AGGREGATE COUNT(*)"
        steps = decompose_trajectory(gold_sql, pipe_sql)
        self.assertGreaterEqual(len(steps), 1)
        # First step should be FROM
        self.assertEqual(steps[0].step_index, 0)
        self.assertIn("FROM", steps[0].operator_text.upper())

    def test_decompose_multi_operator(self):
        gold_sql = "SELECT name, country FROM singer WHERE age > 20 ORDER BY age"
        pipe_sql = "FROM singer |> WHERE age > 20 |> ORDER BY age |> SELECT name, country"
        steps = decompose_trajectory(gold_sql, pipe_sql)
        self.assertGreaterEqual(len(steps), 2)
        # Steps after first should start with |>
        for step in steps[1:]:
            self.assertTrue(step.operator_text.startswith("|>"))

    def test_fallback_decomposition(self):
        pipe_sql = "FROM t1 |> WHERE x > 1 |> SELECT a, b"
        steps = _decompose_fallback(pipe_sql)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].operator_text, "FROM t1")
        self.assertEqual(steps[1].operator_text, "|> WHERE x > 1")
        self.assertEqual(steps[2].operator_text, "|> SELECT a, b")
        self.assertEqual(steps[0].query_so_far, "")
        self.assertEqual(steps[1].query_so_far, "FROM t1")
        self.assertEqual(steps[2].query_so_far, "FROM t1 |> WHERE x > 1")

    def test_passthrough_single_step(self):
        # Set operations yield 1-step trajectories
        gold_sql = "SELECT a FROM t1 UNION SELECT a FROM t2"
        pipe_sql = "SELECT a FROM t1 UNION SELECT a FROM t2"
        steps = decompose_trajectory(gold_sql, pipe_sql)
        self.assertGreaterEqual(len(steps), 1)

    def test_step_total_steps_consistent(self):
        gold_sql = "SELECT name FROM singer WHERE age > 20"
        pipe_sql = "FROM singer |> WHERE age > 20 |> SELECT name"
        steps = decompose_trajectory(gold_sql, pipe_sql)
        for step in steps:
            self.assertEqual(step.total_steps, len(steps))


class TestFormatter(unittest.TestCase):
    def test_format_step_first(self):
        step = TrajectoryStep(
            query_so_far="",
            operator_text="FROM singer",
            op_type="FROM",
            step_index=0,
            total_steps=2,
            cte_prefix="",
        )
        sample = format_step(
            step, "singer(id INTEGER PK, name TEXT)", "Count singers",
            "q1", "concert_singer", "easy",
        )
        self.assertEqual(len(sample.messages), 3)
        self.assertEqual(sample.messages[0]["role"], "system")
        self.assertEqual(sample.messages[0]["content"], SYSTEM_MESSAGE)
        self.assertEqual(sample.messages[1]["role"], "user")
        self.assertIn("Schema:", sample.messages[1]["content"])
        self.assertIn("Question: Count singers", sample.messages[1]["content"])
        self.assertNotIn("Query so far:", sample.messages[1]["content"])
        self.assertEqual(sample.messages[2]["role"], "assistant")
        self.assertEqual(sample.messages[2]["content"], "FROM singer")
        self.assertEqual(sample.metadata["question_id"], "q1")
        self.assertEqual(sample.metadata["op_type"], "FROM")

    def test_format_step_subsequent(self):
        step = TrajectoryStep(
            query_so_far="FROM singer",
            operator_text="|> AGGREGATE COUNT(*)",
            op_type="AGGREGATE",
            step_index=1,
            total_steps=2,
            cte_prefix="",
        )
        sample = format_step(
            step, "singer(id INTEGER PK)", "Count singers",
            "q1", "concert_singer", "easy",
        )
        self.assertIn("Query so far:", sample.messages[1]["content"])
        self.assertIn("FROM singer", sample.messages[1]["content"])
        self.assertEqual(sample.messages[2]["content"], "|> AGGREGATE COUNT(*)")

    def test_format_step_with_cte(self):
        step = TrajectoryStep(
            query_so_far="",
            operator_text="FROM cte1",
            op_type="FROM",
            step_index=0,
            total_steps=1,
            cte_prefix="WITH cte1 AS (FROM t |> SELECT a)",
        )
        sample = format_step(
            step, "t(a INTEGER)", "Query with CTE",
            "q1", "db1", "hard",
        )
        self.assertIn("Given CTEs:", sample.messages[1]["content"])
        self.assertIn("WITH cte1", sample.messages[1]["content"])

    def test_format_trajectory(self):
        steps = [
            TrajectoryStep("", "FROM t", "FROM", 0, 2, ""),
            TrajectoryStep("FROM t", "|> SELECT a", "SELECT", 1, 2, ""),
        ]
        samples = format_trajectory(steps, "t(a INT)", "Get a", "q1", "db1", "easy")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].metadata["step_index"], 0)
        self.assertEqual(samples[1].metadata["step_index"], 1)


class TestWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_samples(self, n_queries: int = 20, steps_per: int = 3) -> list[ChatSample]:
        samples = []
        for q in range(n_queries):
            for s in range(steps_per):
                samples.append(ChatSample(
                    messages=[
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": f"user q{q} s{s}"},
                        {"role": "assistant", "content": f"asst q{q} s{s}"},
                    ],
                    metadata={
                        "question_id": f"q_{q}",
                        "db_id": f"db_{q % 5}",
                        "difficulty": "easy",
                        "step_index": s,
                        "total_steps": steps_per,
                        "op_type": "FROM" if s == 0 else "SELECT",
                    },
                ))
        return samples

    def test_write_output_creates_files(self):
        samples = self._make_samples()
        stats = write_output(samples, self.tmpdir, train_ratio=0.95, seed=42)

        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "train.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "dev.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "stats.json")))

        self.assertEqual(stats["total_samples"], 60)
        self.assertEqual(stats["train_samples"] + stats["dev_samples"], 60)
        self.assertGreater(stats["train_samples"], stats["dev_samples"])

    def test_split_keeps_query_steps_together(self):
        samples = self._make_samples(n_queries=100, steps_per=3)
        stats = write_output(samples, self.tmpdir, train_ratio=0.8, seed=42)

        # Read train and dev and check no question_id appears in both
        train_qids = set()
        dev_qids = set()

        with open(os.path.join(self.tmpdir, "train.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                train_qids.add(d["metadata"]["question_id"])

        with open(os.path.join(self.tmpdir, "dev.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                dev_qids.add(d["metadata"]["question_id"])

        self.assertEqual(len(train_qids & dev_qids), 0, "question_ids should not overlap")

    def test_jsonl_format(self):
        samples = self._make_samples(n_queries=2, steps_per=2)
        write_output(samples, self.tmpdir)

        with open(os.path.join(self.tmpdir, "train.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                self.assertIn("messages", d)
                self.assertIn("metadata", d)
                self.assertEqual(len(d["messages"]), 3)
                self.assertEqual(d["messages"][0]["role"], "system")


class TestToolExecutor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_dir = os.path.join(self.tmpdir, "test_db")
        os.makedirs(db_dir)
        self.db_path = os.path.join(db_dir, "test_db.sqlite")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE singer ("
            "singer_id INTEGER PRIMARY KEY, "
            "name TEXT NOT NULL, "
            "age INTEGER)"
        )
        conn.execute("INSERT INTO singer VALUES (1, 'Alice', 25)")
        conn.execute("INSERT INTO singer VALUES (2, 'Bob', 30)")
        conn.execute(
            "CREATE TABLE concert ("
            "concert_id INTEGER PRIMARY KEY, "
            "concert_name TEXT, "
            "singer_id INTEGER, "
            "FOREIGN KEY (singer_id) REFERENCES singer(singer_id))"
        )
        conn.execute("INSERT INTO concert VALUES (1, 'Rock Show', 1)")
        conn.commit()
        conn.close()
        self.tables = extract_db_schema(self.db_path)

    def test_list_tables_result(self):
        result = list_tables_result(self.tables)
        parsed = json.loads(result)
        self.assertIn("singer", parsed)
        self.assertIn("concert", parsed)

    def test_describe_table_result(self):
        result = describe_table_result(self.tables, "singer")
        self.assertIn("singer", result)
        self.assertIn("singer_id", result)
        self.assertIn("PK", result)

    def test_describe_table_not_found(self):
        result = describe_table_result(self.tables, "nonexistent")
        self.assertIn("Error", result)

    def test_sample_data_result(self):
        result = sample_data_result(self.db_path, "singer", limit=5)
        self.assertIn("singer_id", result)
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)

    def test_execute_pipe_sql_result(self):
        result = execute_pipe_sql_result(self.db_path, "FROM singer |> AGGREGATE COUNT(*)")
        self.assertIn("2", result)

    def test_validate_pipe_sql_valid(self):
        result = validate_pipe_sql_result("FROM singer |> SELECT name")
        self.assertEqual(result, "Valid syntax.")

    def test_validate_pipe_sql_invalid(self):
        result = validate_pipe_sql_result("SELECTT BAD SYNTAX !!!")
        # Should either say valid (if sqlglot is lenient) or return error
        # Just ensure it doesn't crash
        self.assertIsInstance(result, str)


class TestToolFormatter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_dir = os.path.join(self.tmpdir, "test_db")
        os.makedirs(db_dir)
        self.db_path = os.path.join(db_dir, "test_db.sqlite")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE singer ("
            "singer_id INTEGER PRIMARY KEY, "
            "name TEXT NOT NULL, "
            "age INTEGER)"
        )
        conn.execute("INSERT INTO singer VALUES (1, 'Alice', 25)")
        conn.execute("INSERT INTO singer VALUES (2, 'Bob', 30)")
        conn.execute(
            "CREATE TABLE concert ("
            "concert_id INTEGER PRIMARY KEY, "
            "concert_name TEXT, "
            "singer_id INTEGER, "
            "FOREIGN KEY (singer_id) REFERENCES singer(singer_id))"
        )
        conn.execute("INSERT INTO concert VALUES (1, 'Rock Show', 1)")
        conn.commit()
        conn.close()
        self.tables = extract_db_schema(self.db_path)
        self.rng = random.Random(42)

    def test_extract_referenced_tables(self):
        tables = extract_referenced_tables("SELECT name FROM singer WHERE age > 20")
        self.assertEqual(tables, ["singer"])

    def test_extract_referenced_tables_join(self):
        tables = extract_referenced_tables(
            "SELECT s.name, c.concert_name FROM singer s JOIN concert c ON s.singer_id = c.singer_id"
        )
        self.assertIn("singer", [t.lower() for t in tables])
        self.assertIn("concert", [t.lower() for t in tables])

    def test_extract_referenced_tables_invalid(self):
        tables = extract_referenced_tables("NOT VALID SQL !!!")
        self.assertEqual(tables, [])

    def test_format_pattern_a_easy(self):
        sample = format_tool_calling_sample(
            db_id="test_db",
            db_path=self.db_path,
            question="How many singers are there?",
            question_id="q1",
            difficulty="easy",
            gold_sql="SELECT count(*) FROM singer",
            pipe_sql="FROM singer |> AGGREGATE COUNT(*)",
            tables=self.tables,
            rng=self.rng,
        )
        self.assertIsNotNone(sample)
        self.assertEqual(sample.metadata["pattern"], "A")
        self.assertEqual(sample.metadata["sample_type"], "tool_calling")
        self.assertIsNotNone(sample.tools)
        self.assertEqual(len(sample.tools), 5)

        # Check message structure
        roles = [m["role"] for m in sample.messages]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[1], "user")
        # Should have at least: system, user, assistant(list_tables), tool, assistant(describe), tool,
        # assistant(query + verify), tool(result), assistant(confirm)
        self.assertGreaterEqual(len(sample.messages), 9)

        # Last message is the confirmation
        last_msg = sample.messages[-1]
        self.assertEqual(last_msg["role"], "assistant")
        self.assertNotIn("tool_calls", last_msg)

        # Second-to-last assistant message (before tool response) has the pipe SQL + verify tool call
        query_msg = sample.messages[-3]
        self.assertEqual(query_msg["role"], "assistant")
        self.assertIn("FROM singer", query_msg["content"])
        self.assertIn("tool_calls", query_msg)

    def test_format_pattern_b_medium(self):
        sample = format_tool_calling_sample(
            db_id="test_db",
            db_path=self.db_path,
            question="Names of singers older than 20?",
            question_id="q2",
            difficulty="medium",
            gold_sql="SELECT name FROM singer WHERE age > 20",
            pipe_sql="FROM singer |> WHERE age > 20 |> SELECT name",
            tables=self.tables,
            rng=self.rng,
        )
        self.assertIsNotNone(sample)
        self.assertIn(sample.metadata["pattern"], ("A", "B"))

    def test_format_pattern_c_hard(self):
        sample = format_tool_calling_sample(
            db_id="test_db",
            db_path=self.db_path,
            question="Concert names with singer names?",
            question_id="q3",
            difficulty="extra",
            gold_sql="SELECT s.name, c.concert_name FROM singer s JOIN concert c ON s.singer_id = c.singer_id WHERE s.age > 20 ORDER BY c.concert_name",
            pipe_sql="FROM singer AS s |> JOIN concert AS c ON s.singer_id = c.singer_id |> WHERE s.age > 20 |> ORDER BY c.concert_name |> SELECT s.name, c.concert_name",
            tables=self.tables,
            rng=self.rng,
        )
        self.assertIsNotNone(sample)
        self.assertEqual(sample.metadata["pattern"], "C")

    def test_tool_call_ids_sequential(self):
        sample = format_tool_calling_sample(
            db_id="test_db",
            db_path=self.db_path,
            question="How many singers are there?",
            question_id="q1",
            difficulty="easy",
            gold_sql="SELECT count(*) FROM singer",
            pipe_sql="FROM singer |> AGGREGATE COUNT(*)",
            tables=self.tables,
            rng=random.Random(42),
        )
        self.assertIsNotNone(sample)
        call_ids = []
        for msg in sample.messages:
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    call_ids.append(tc["id"])
        # Should be sequential: call_1, call_2, ...
        for i, cid in enumerate(call_ids, 1):
            self.assertEqual(cid, f"call_{i}")

    def test_tool_response_matches_call(self):
        sample = format_tool_calling_sample(
            db_id="test_db",
            db_path=self.db_path,
            question="How many singers are there?",
            question_id="q1",
            difficulty="easy",
            gold_sql="SELECT count(*) FROM singer",
            pipe_sql="FROM singer |> AGGREGATE COUNT(*)",
            tables=self.tables,
            rng=random.Random(42),
        )
        self.assertIsNotNone(sample)
        # Every tool_call should be followed by a tool response with matching ID
        for i, msg in enumerate(sample.messages):
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    next_msg = sample.messages[i + 1]
                    self.assertEqual(next_msg["role"], "tool")
                    self.assertEqual(next_msg["tool_call_id"], tc["id"])

    def test_no_tables_returns_none(self):
        sample = format_tool_calling_sample(
            db_id="test_db",
            db_path=self.db_path,
            question="How many singers are there?",
            question_id="q1",
            difficulty="easy",
            gold_sql="NOT VALID SQL !!!",
            pipe_sql="FROM singer |> AGGREGATE COUNT(*)",
            tables=self.tables,
            rng=self.rng,
        )
        self.assertIsNone(sample)


class TestSchemaExtractorCaches(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_dir = os.path.join(self.tmpdir, "test_db")
        os.makedirs(db_dir)
        self.db_path = os.path.join(db_dir, "test_db.sqlite")
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val TEXT)")
        conn.commit()
        conn.close()

    def test_build_tables_cache(self):
        cache = build_tables_cache([self.tmpdir])
        self.assertIn("test_db", cache)
        self.assertIsInstance(cache["test_db"], list)
        self.assertEqual(cache["test_db"][0].name, "t1")

    def test_build_db_path_cache(self):
        cache = build_db_path_cache([self.tmpdir])
        self.assertIn("test_db", cache)
        self.assertTrue(cache["test_db"].endswith("test_db.sqlite"))
        self.assertTrue(os.path.isabs(cache["test_db"]))


class TestWriterWithTools(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_tools_field_in_output(self):
        samples = [
            ChatSample(
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "asst"},
                ],
                metadata={
                    "question_id": "q_0",
                    "db_id": "db_0",
                    "difficulty": "easy",
                    "sample_type": "tool_calling",
                },
                tools=[{"type": "function", "function": {"name": "test"}}],
            ),
            ChatSample(
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "asst"},
                ],
                metadata={
                    "question_id": "q_1",
                    "db_id": "db_0",
                    "difficulty": "easy",
                    "step_index": 0,
                    "total_steps": 1,
                    "op_type": "FROM",
                },
            ),
        ]
        stats = write_output(samples, self.tmpdir, train_ratio=0.95, seed=42)
        self.assertEqual(stats["tool_calling_samples"], 1)

        # Check that tools field appears in JSONL
        found_tools = False
        found_no_tools = False
        for fname in ("train.jsonl", "dev.jsonl"):
            path = os.path.join(self.tmpdir, fname)
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    if "tools" in d:
                        found_tools = True
                    else:
                        found_no_tools = True
        self.assertTrue(found_tools, "Should have at least one sample with tools")
        self.assertTrue(found_no_tools, "Should have at least one sample without tools")


class TestGenerateCLI(unittest.TestCase):
    def test_main_with_limit(self):
        """End-to-end test with a small synthetic database and golden pairs."""
        tmpdir = tempfile.mkdtemp()

        # Create a test DB
        db_dir = os.path.join(tmpdir, "dbs", "test_db")
        os.makedirs(db_dir)
        db_path = os.path.join(db_dir, "test_db.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.commit()
        conn.close()

        # Create golden pairs file
        pairs_path = os.path.join(tmpdir, "pairs.jsonl")
        with open(pairs_path, "w") as f:
            f.write(json.dumps({
                "question_id": "test_0",
                "db_id": "test_db",
                "difficulty": "easy",
                "gold_sql": "SELECT count(*) FROM singer",
                "pipe_sql": "FROM singer |> AGGREGATE COUNT(*)",
                "round_tripped_sql": "SELECT COUNT(*) FROM singer",
                "validation": "MATCH",
                "question": "How many singers are there?",
            }) + "\n")
            f.write(json.dumps({
                "question_id": "test_1",
                "db_id": "test_db",
                "difficulty": "medium",
                "gold_sql": "SELECT name FROM singer WHERE age > 20",
                "pipe_sql": "FROM singer |> WHERE age > 20 |> SELECT name",
                "round_tripped_sql": "SELECT name FROM singer WHERE age > 20",
                "validation": "MATCH",
                "question": "Names of singers older than 20?",
            }) + "\n")

        output_dir = os.path.join(tmpdir, "output")
        from training_data.generate import main

        main([
            "--golden-pairs", pairs_path,
            "--db-dir", os.path.join(tmpdir, "dbs"),
            "--output-dir", output_dir,
            "--limit", "2",
        ])

        # Check output files exist
        self.assertTrue(os.path.exists(os.path.join(output_dir, "train.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "dev.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "stats.json")))

        # Check stats
        with open(os.path.join(output_dir, "stats.json")) as f:
            stats = json.load(f)

        # 2 queries: first has 2 steps, second has 3 steps = 5 total samples
        self.assertEqual(stats["total_samples"], 5)
        self.assertEqual(stats["total_queries"], 2)

    def test_main_with_tool_calling(self):
        """End-to-end test with tool-calling enabled."""
        tmpdir = tempfile.mkdtemp()

        # Create a test DB with data
        db_dir = os.path.join(tmpdir, "dbs", "test_db")
        os.makedirs(db_dir)
        db_path = os.path.join(db_dir, "test_db.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.execute("INSERT INTO singer VALUES (1, 'Alice', 25)")
        conn.execute("INSERT INTO singer VALUES (2, 'Bob', 30)")
        conn.commit()
        conn.close()

        # Create golden pairs file
        pairs_path = os.path.join(tmpdir, "pairs.jsonl")
        with open(pairs_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "question_id": f"test_{i}",
                    "db_id": "test_db",
                    "difficulty": "easy",
                    "gold_sql": "SELECT count(*) FROM singer",
                    "pipe_sql": "FROM singer |> AGGREGATE COUNT(*)",
                    "round_tripped_sql": "SELECT COUNT(*) FROM singer",
                    "validation": "MATCH",
                    "question": "How many singers are there?",
                }) + "\n")

        output_dir = os.path.join(tmpdir, "output")
        from training_data.generate import main

        main([
            "--golden-pairs", pairs_path,
            "--db-dir", os.path.join(tmpdir, "dbs"),
            "--output-dir", output_dir,
            "--tool-calling",
            "--tool-ratio", "1.0",  # 100% to guarantee tool samples
            "--seed", "42",
        ])

        with open(os.path.join(output_dir, "stats.json")) as f:
            stats = json.load(f)

        # 10 queries * 2 trajectory steps = 20, plus 10 tool-calling samples = 30
        self.assertEqual(stats["total_samples"], 30)
        self.assertEqual(stats["tool_calling_samples"], 10)

        # Verify tool-calling samples have tools field in JSONL
        tool_count = 0
        for fname in ("train.jsonl", "dev.jsonl"):
            path = os.path.join(output_dir, fname)
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    if "tools" in d:
                        tool_count += 1
                        self.assertEqual(len(d["tools"]), 5)
        self.assertEqual(tool_count, 10)


if __name__ == "__main__":
    unittest.main()
