#!/bin/bash
set -e

# Orchestrator: start server -> run agent benchmark -> evaluate
# Usage: bash evaluation/run_all.sh [--limit N]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LIMIT="${1:---limit}"
LIMIT_N="${2:-0}"

# Parse --limit flag
if [ "$LIMIT" = "--limit" ] && [ "$LIMIT_N" != "0" ]; then
    AGENT_ARGS="--benchmark --limit $LIMIT_N"
else
    AGENT_ARGS="--benchmark"
fi

echo "=== Pipe SQL Evaluation Pipeline ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# 1. Start Python server in background
echo "[1/4] Starting evaluation server..."
python -m evaluation.server.app &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  Server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ERROR: Server failed to start after 60s"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Ensure server is killed on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Server stopped."
}
trap cleanup EXIT

# 2. Install agent dependencies
echo ""
echo "[2/4] Installing agent dependencies..."
cd "$PROJECT_ROOT/evaluation/agent"
npm install --silent
cd "$PROJECT_ROOT"

# 3. Run agent benchmark
echo ""
echo "[3/4] Running agent benchmark ($AGENT_ARGS)..."
cd "$PROJECT_ROOT/evaluation/agent"
npx tsx src/main.ts $AGENT_ARGS
cd "$PROJECT_ROOT"

# 4. Run evaluation
echo ""
echo "[4/4] Running evaluation..."
python evaluation/evaluate.py --results evaluation_output/results.json

echo ""
echo "=== Pipeline Complete ==="
