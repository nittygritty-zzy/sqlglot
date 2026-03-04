# Pipe SQL Evaluation System

End-to-end evaluation for the fine-tuned pipe SQL model using agentic tool-calling.

## Architecture

```
TypeScript Agent (OpenAI client)
  │ POST /v1/chat/completions     │ POST /tools/{name}
  ▼                                ▼
Python Server (FastAPI)
  ├─ Model inference (training chat template)
  ├─ Tool execution (reuses tool_executor.py)
  └─ Benchmark questions (Spider dev set)

evaluate.py → transpile + execute + compare
```

## Prerequisites

1. **Fine-tuned model** at `finetuning_output/merged/` (run `python -m finetuning.train --merge` first)
2. **Spider databases** at `data/spider/database/`
3. **Python dependencies**: `pip install -r evaluation/requirements.txt`
4. **Node.js** 18+ with npm

## Quick Start

### Run full pipeline
```bash
# All 1,034 questions
bash evaluation/run_all.sh

# Smoke test with 5 questions
bash evaluation/run_all.sh --limit 5
```

### Run components separately

**Start server:**
```bash
python -m evaluation.server.app
# Server runs at http://localhost:8000
```

**Run agent benchmark:**
```bash
cd evaluation/agent
npm install
npx tsx src/main.ts --benchmark --limit 10
```

**Run single question interactively:**
```bash
cd evaluation/agent
npx tsx src/main.ts "How many singers do we have?" concert_singer
```

**Evaluate results:**
```bash
python evaluation/evaluate.py --results evaluation_output/results.json
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `finetuning_output/merged` | Path to merged model |
| `SPIDER_DB_DIR` | `data/spider/database` | Spider database directory |
| `SPIDER_DIR` | `data/spider` | Spider data directory |
| `SPIDER2_DB_DIR` | (none) | Optional Spider2-lite database directory |
| `PORT` | `8000` | Server port |
| `SERVER_URL` | `http://localhost:8000` | Agent → server URL |
| `OUTPUT_DIR` | `evaluation_output` | Agent output directory |

## Output Files

```
evaluation_output/
├── results.json        # Agent predictions (from TypeScript agent)
├── eval_results.json   # Per-question evaluation details
└── eval_summary.json   # Aggregate metrics
```

## Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (uses training template) |
| `/tools/{name}` | POST | Execute tool (`list_tables`, `describe_table`, `sample_data`, `execute_pipe_sql`, `validate_pipe_sql`) |
| `/benchmark/questions` | GET | Load benchmark questions (`?limit=N&offset=M`) |
| `/health` | GET | Health check |
