/**
 * CLI entry point for the evaluation agent.
 *
 * Modes:
 *   --benchmark [--limit N]   Run benchmark on all/subset of questions
 *   <question> <db_id>        Interactive single-question mode
 */

import * as fs from "fs";
import * as path from "path";
import { runAgent } from "./agent.js";
import type { BenchmarkQuestion, BenchmarkResult } from "./types.js";

const SERVER_URL = process.env.SERVER_URL || "http://localhost:8000";
// Default output to project root's evaluation_output/
const OUTPUT_DIR = process.env.OUTPUT_DIR || path.resolve(__dirname, "../../../evaluation_output");

async function fetchQuestions(
  limit: number = 0,
  offset: number = 0,
): Promise<BenchmarkQuestion[]> {
  const params = new URLSearchParams();
  if (limit > 0) params.set("limit", String(limit));
  if (offset > 0) params.set("offset", String(offset));

  const url = `${SERVER_URL}/benchmark/questions?${params}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch questions: ${resp.status}`);

  const data = (await resp.json()) as {
    questions: BenchmarkQuestion[];
    total: number;
  };
  console.log(`Fetched ${data.questions.length} questions (total: ${data.total})`);
  return data.questions;
}

async function runBenchmark(limit: number = 0): Promise<void> {
  const questions = await fetchQuestions(limit);
  const results: BenchmarkResult[] = [];
  const startTime = Date.now();

  console.log(
    `\nRunning benchmark on ${questions.length} questions...\n`,
  );

  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    const qStart = Date.now();

    console.log(
      `[${i + 1}/${questions.length}] ${q.db_id}: ${q.question.slice(0, 80)}...`,
    );

    try {
      const result = await runAgent(q.question, q.db_id);
      const durationMs = Date.now() - qStart;

      results.push({
        id: q.id,
        question: q.question,
        db_id: q.db_id,
        gold_sql: q.gold_sql,
        predicted_pipe_sql: result.predictedPipeSQL,
        conversation: result.conversation,
        num_turns: result.numTurns,
        duration_ms: durationMs,
        status: result.status,
        error: result.error,
      });

      const sqlPreview = result.predictedPipeSQL
        ? result.predictedPipeSQL.slice(0, 60) + "..."
        : "NONE";
      console.log(
        `  -> ${result.status} (${result.numTurns} turns, ${durationMs}ms) SQL: ${sqlPreview}`,
      );
    } catch (err) {
      const durationMs = Date.now() - qStart;
      results.push({
        id: q.id,
        question: q.question,
        db_id: q.db_id,
        gold_sql: q.gold_sql,
        predicted_pipe_sql: null,
        conversation: [],
        num_turns: 0,
        duration_ms: durationMs,
        status: "error",
        error: String(err),
      });
      console.log(`  -> ERROR: ${err}`);
    }
  }

  // Write results
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const outputPath = path.join(OUTPUT_DIR, "results.json");
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));

  // Summary
  const totalMs = Date.now() - startTime;
  const success = results.filter((r) => r.status === "success").length;
  const noPred = results.filter((r) => r.status === "no_prediction").length;
  const errors = results.filter((r) => r.status === "error").length;
  const timeouts = results.filter((r) => r.status === "timeout").length;

  console.log(`\n=== Benchmark Complete ===`);
  console.log(`Total: ${results.length} questions in ${(totalMs / 1000).toFixed(1)}s`);
  console.log(`Success: ${success}, No prediction: ${noPred}, Error: ${errors}, Timeout: ${timeouts}`);
  console.log(`Results saved to: ${outputPath}`);
}

async function runInteractive(question: string, dbId: string): Promise<void> {
  console.log(`\nQuestion: ${question}`);
  console.log(`Database: ${dbId}\n`);

  const result = await runAgent(question, dbId);

  console.log("\n=== Conversation ===");
  for (const msg of result.conversation) {
    const prefix = msg.role.toUpperCase().padEnd(10);
    const content = msg.content.slice(0, 200);
    console.log(`${prefix} ${content}`);
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        console.log(
          `           -> ${tc.function.name}(${tc.function.arguments.slice(0, 100)})`,
        );
      }
    }
  }

  console.log(`\n=== Result ===`);
  console.log(`Status: ${result.status}`);
  console.log(`Turns: ${result.numTurns}`);
  console.log(`Predicted SQL: ${result.predictedPipeSQL || "NONE"}`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  // Parse flags
  const isBenchmark = args.includes("--benchmark");
  let limit = 0;
  const limitIdx = args.indexOf("--limit");
  if (limitIdx >= 0 && limitIdx + 1 < args.length) {
    limit = parseInt(args[limitIdx + 1], 10);
  }

  if (isBenchmark) {
    await runBenchmark(limit);
  } else if (args.length >= 2 && !args[0].startsWith("--")) {
    // Interactive mode: question db_id
    const question = args[0];
    const dbId = args[1];
    await runInteractive(question, dbId);
  } else {
    console.log("Usage:");
    console.log("  tsx src/main.ts --benchmark [--limit N]");
    console.log('  tsx src/main.ts "question text" db_id');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
