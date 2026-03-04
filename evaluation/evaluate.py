"""Post-hoc evaluation: execution accuracy for pipe SQL predictions.

Reads results.json from the agent benchmark run, transpiles predicted pipe SQL,
executes against SQLite databases, and compares with gold results.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

# Add project root to path for imports when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlglot

from validation.compare import compare_with_tolerance


def transpile_pipe_sql(pipe_sql: str) -> str:
    """Transpile pipe SQL to standard SQLite SQL."""
    return sqlglot.transpile(pipe_sql, read="sqlite", write="sqlite")[0]


def execute_sql(db_path: str, sql: str) -> list[tuple]:
    """Execute SQL against a SQLite database and return results."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        conn.close()


def find_db_path(db_id: str, db_dirs: list[str]) -> str | None:
    """Find the SQLite database file for a given db_id."""
    for db_dir in db_dirs:
        path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def evaluate_single(
    result: dict,
    db_dirs: list[str],
) -> dict:
    """Evaluate a single prediction against gold."""
    qid = result["id"]
    db_id = result["db_id"]
    gold_sql = result["gold_sql"]
    predicted_pipe_sql = result.get("predicted_pipe_sql")

    eval_result = {
        "id": qid,
        "db_id": db_id,
        "question": result["question"],
        "gold_sql": gold_sql,
        "predicted_pipe_sql": predicted_pipe_sql,
        "status": "unknown",
        "detail": "",
    }

    # No prediction
    if not predicted_pipe_sql:
        eval_result["status"] = "no_prediction"
        eval_result["detail"] = f"Agent status: {result.get('status', 'unknown')}"
        return eval_result

    # Find database
    db_path = find_db_path(db_id, db_dirs)
    if not db_path:
        eval_result["status"] = "db_not_found"
        eval_result["detail"] = f"Database {db_id} not found"
        return eval_result

    # Transpile predicted pipe SQL
    try:
        transpiled_sql = transpile_pipe_sql(predicted_pipe_sql)
        eval_result["transpiled_sql"] = transpiled_sql
    except Exception as e:
        eval_result["status"] = "transpile_error"
        eval_result["detail"] = str(e)
        return eval_result

    # Execute gold SQL
    try:
        gold_results = execute_sql(db_path, gold_sql)
    except Exception as e:
        eval_result["status"] = "gold_error"
        eval_result["detail"] = f"Gold SQL error: {e}"
        return eval_result

    # Execute predicted SQL
    try:
        pred_results = execute_sql(db_path, transpiled_sql)
    except Exception as e:
        eval_result["status"] = "exec_error"
        eval_result["detail"] = f"Predicted SQL error: {e}"
        return eval_result

    # Compare results
    try:
        comparison = compare_with_tolerance(gold_results, pred_results)
        eval_result["match"] = comparison.match
        eval_result["match_type"] = comparison.match_type
        eval_result["status"] = "match" if comparison.match else "mismatch"
        eval_result["gold_rows"] = comparison.row_count_a
        eval_result["pred_rows"] = comparison.row_count_b
        if not comparison.match:
            eval_result["detail"] = comparison.detail
            eval_result["mismatch_type"] = comparison.mismatch_type
            eval_result["f1_score"] = comparison.f1_score
    except Exception as e:
        eval_result["status"] = "compare_error"
        eval_result["detail"] = str(e)

    return eval_result


def run_evaluation(
    results_path: str,
    db_dirs: list[str],
    output_dir: str,
) -> None:
    """Run evaluation on all results."""
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"Evaluating {len(results)} predictions...")
    print(f"Database dirs: {db_dirs}")

    eval_results = []
    start_time = time.time()

    for i, result in enumerate(results):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Evaluating {i + 1}/{len(results)}...")

        eval_result = evaluate_single(result, db_dirs)
        eval_results.append(eval_result)

    elapsed = time.time() - start_time

    # Compute summary
    total = len(eval_results)
    statuses = {}
    for r in eval_results:
        s = r["status"]
        statuses[s] = statuses.get(s, 0) + 1

    matches = statuses.get("match", 0)
    gold_errors = statuses.get("gold_error", 0)
    denominator = total - gold_errors  # Exclude gold errors from accuracy

    accuracy = matches / denominator if denominator > 0 else 0.0

    summary = {
        "total_questions": total,
        "execution_accuracy": round(accuracy, 4),
        "matches": matches,
        "denominator": denominator,
        "gold_errors_excluded": gold_errors,
        "status_breakdown": statuses,
        "evaluation_time_s": round(elapsed, 1),
    }

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console report
    print(f"\n{'=' * 50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"  Total questions:      {total}")
    print(f"  Execution accuracy:   {accuracy:.1%} ({matches}/{denominator})")
    print(f"  Gold errors excluded: {gold_errors}")
    print(f"{'=' * 50}")
    print(f"  Status breakdown:")
    for status, count in sorted(statuses.items()):
        pct = count / total * 100
        print(f"    {status:20s} {count:4d} ({pct:.1f}%)")
    print(f"{'=' * 50}")
    print(f"  Evaluation time: {elapsed:.1f}s")
    print(f"  Results: {eval_path}")
    print(f"  Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipe SQL predictions")
    parser.add_argument(
        "--results",
        default="evaluation_output/results.json",
        help="Path to agent results JSON",
    )
    parser.add_argument(
        "--db-dirs",
        nargs="+",
        default=["data/spider/database"],
        help="Database directories to search",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_output",
        help="Output directory for evaluation results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)

    run_evaluation(args.results, args.db_dirs, args.output_dir)


if __name__ == "__main__":
    main()
