"""CLI entry point for training data generation.

Usage:
    python -m training_data.generate \
        --golden-pairs validation_output/golden_pairs_consolidated.jsonl \
        --db-dir data/spider/database --db-dir data/bird/databases \
        --output-dir training_data_output

    # With tool-calling samples:
    python -m training_data.generate \
        --golden-pairs validation_output/golden_pairs_consolidated.jsonl \
        --db-dir data/spider/database --db-dir data/bird/databases \
        --output-dir training_data_output \
        --tool-calling --tool-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from training_data.formatter import ChatSample, format_trajectory
from training_data.schema_extractor import build_db_path_cache, build_schema_cache, build_tables_cache
from training_data.tool_formatter import format_tool_calling_sample
from training_data.trajectory import decompose_trajectory
from training_data.writer import write_output


def _load_golden_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _verify_consistency(gold_sql: str, pipe_sql: str, steps: list, dialect: str = "sqlite") -> bool:
    """Verify that reassembled trajectory matches the golden pipe_sql."""
    if not steps:
        return False
    # Reassemble from operator texts
    parts = []
    for step in steps:
        parts.append(step.operator_text)
    reassembled = " ".join(parts)

    # Add CTE prefix if present
    if steps[0].cte_prefix:
        reassembled = f"{steps[0].cte_prefix} {reassembled}"

    return reassembled.strip() == pipe_sql.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate training data from golden pairs")
    parser.add_argument("--golden-pairs", required=True, help="Path to golden_pairs_consolidated.jsonl")
    parser.add_argument("--db-dir", action="append", required=True, help="DB directory (repeatable)")
    parser.add_argument("--output-dir", default="training_data_output", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.95, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=0, help="Process only N entries (0 = all)")
    parser.add_argument("--verify", action="store_true", help="Verify re-emitted pipe_sql matches golden pair")
    parser.add_argument("--tool-calling", action="store_true", help="Enable tool-calling sample generation")
    parser.add_argument("--tool-ratio", type=float, default=0.3, help="Fraction of golden pairs that also get a tool-calling sample")
    args = parser.parse_args(argv)

    print(f"Loading golden pairs from {args.golden_pairs}...")
    pairs = _load_golden_pairs(args.golden_pairs)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"Loaded {len(pairs)} golden pairs")

    print(f"Building schema cache from {args.db_dir}...")
    schema_cache = build_schema_cache(args.db_dir)
    print(f"Cached schemas for {len(schema_cache)} databases")

    # Build additional caches for tool-calling
    tables_cache: dict[str, list] = {}
    db_path_cache: dict[str, str] = {}
    if args.tool_calling:
        print("Building tables and DB path caches for tool-calling...")
        tables_cache = build_tables_cache(args.db_dir)
        db_path_cache = build_db_path_cache(args.db_dir)
        print(f"Cached tables for {len(tables_cache)} databases, paths for {len(db_path_cache)}")

    rng = random.Random(args.seed)
    all_samples: list[ChatSample] = []
    errors_skipped = 0
    missing_db = 0
    verify_failures = 0
    tool_samples_generated = 0

    for i, entry in enumerate(pairs):
        db_id = entry["db_id"]
        question_id = entry["question_id"]

        # Look up schema
        schema_str = schema_cache.get(db_id)
        if not schema_str:
            missing_db += 1
            continue

        # Decompose trajectory
        try:
            steps = decompose_trajectory(entry["gold_sql"], entry["pipe_sql"])
        except Exception as e:
            print(f"  Error decomposing {question_id}: {e}", file=sys.stderr)
            errors_skipped += 1
            continue

        if not steps:
            errors_skipped += 1
            continue

        # Verify consistency if requested
        if args.verify:
            if not _verify_consistency(entry["gold_sql"], entry["pipe_sql"], steps):
                verify_failures += 1
                if verify_failures <= 10:
                    print(f"  Verify mismatch: {question_id}", file=sys.stderr)

        # Integrity checks
        assert schema_str, f"Empty schema for {db_id}"
        assert len(steps) >= 1, f"No steps for {question_id}"
        first_word = steps[0].operator_text.strip().split()[0].upper()
        if len(steps) == 1 and "|>" not in entry["pipe_sql"]:
            pass  # Passthrough (set operations) — no prefix constraint
        else:
            assert first_word == "FROM", (
                f"First step should start with FROM, got '{first_word}' for {question_id}"
            )
        for step in steps[1:]:
            assert step.operator_text.startswith("|>"), (
                f"Step {step.step_index} should start with |> for {question_id}"
            )

        # Format trajectory chat samples (always generated)
        samples = format_trajectory(
            steps=steps,
            schema_str=schema_str,
            question=entry.get("question", ""),
            question_id=question_id,
            db_id=db_id,
            difficulty=entry.get("difficulty", ""),
        )

        # Check non-empty content
        for sample in samples:
            for msg in sample.messages:
                assert msg["content"], f"Empty message content in {question_id}"

        all_samples.extend(samples)

        # Optionally generate tool-calling sample
        if args.tool_calling and rng.random() < args.tool_ratio:
            tables = tables_cache.get(db_id)
            db_path = db_path_cache.get(db_id)
            question = entry.get("question", "")
            if tables and question:
                try:
                    tool_sample = format_tool_calling_sample(
                        db_id=db_id,
                        db_path=db_path,
                        question=question,
                        question_id=question_id,
                        difficulty=entry.get("difficulty", ""),
                        gold_sql=entry["gold_sql"],
                        pipe_sql=entry["pipe_sql"],
                        tables=tables,
                        rng=rng,
                    )
                    if tool_sample:
                        all_samples.append(tool_sample)
                        tool_samples_generated += 1
                except Exception as e:
                    print(f"  Error generating tool sample for {question_id}: {e}", file=sys.stderr)

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} entries, {len(all_samples)} samples so far")

    print(f"\nProcessed {len(pairs)} entries -> {len(all_samples)} samples")
    if missing_db:
        print(f"  Skipped {missing_db} entries (missing DB)")
    if errors_skipped:
        print(f"  Skipped {errors_skipped} entries (errors)")
    if args.verify:
        print(f"  Verify failures: {verify_failures}")
    if args.tool_calling:
        print(f"  Tool-calling samples: {tool_samples_generated}")

    print(f"\nWriting output to {args.output_dir}...")
    stats = write_output(
        samples=all_samples,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        errors_skipped=errors_skipped,
        missing_db=missing_db,
    )

    print(f"\nStats:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Train: {stats['train_samples']}, Dev: {stats['dev_samples']}")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Databases: {stats['database_count']}")
    print(f"  Operator distribution: {stats['operator_distribution']}")
    if stats.get("tool_calling_samples"):
        print(f"  Tool-calling samples: {stats['tool_calling_samples']}")


if __name__ == "__main__":
    main()
