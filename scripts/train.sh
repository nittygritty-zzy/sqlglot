#!/usr/bin/env bash
# Fine-tune Qwen2.5-Coder on pipe SQL training data
#
# Usage:
#   bash scripts/train.sh                    # Full training (CUDA GPU recommended)
#   bash scripts/train.sh --smoke-test       # Quick smoke test (works on MPS/CPU)
#   bash scripts/train.sh --generate-only    # Only regenerate training data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Defaults
MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH=4096
BATCH_SIZE=4
GRAD_ACCUM=8
NUM_EPOCHS=3
LIMIT=2000
TOOL_RATIO=0.3
OUTPUT_DIR="finetuning_output"
DATA_DIR="training_data_output"
GOLDEN_PAIRS="validation_output/golden_pairs_consolidated.jsonl"
SMOKE_TEST=false
GENERATE_ONLY=false

for arg in "$@"; do
    case $arg in
        --smoke-test)
            SMOKE_TEST=true
            ;;
        --generate-only)
            GENERATE_ONLY=true
            ;;
    esac
done

if $SMOKE_TEST; then
    LIMIT=100
    NUM_EPOCHS=1
    BATCH_SIZE=1
    GRAD_ACCUM=4
    OUTPUT_DIR="/tmp/ft_smoke_test"
    DATA_DIR="/tmp/ft_smoke_data"
fi

# ── Step 1: Generate training data ──
echo "=== Generating Training Data ==="
echo "  Golden pairs: $GOLDEN_PAIRS"
echo "  Limit: $LIMIT"
echo "  Tool ratio: $TOOL_RATIO"

DB_DIRS=()
for d in data/spider/database data/bird/train/train_databases data/bird/dev_20240627/dev_databases; do
    if [ -d "$d" ]; then
        DB_DIRS+=(--db-dir "$d")
    fi
done

if [ ${#DB_DIRS[@]} -eq 0 ]; then
    echo "Error: No database directories found. Run setup scripts first:"
    echo "  bash scripts/setup_data.sh"
    echo "  bash scripts/setup_bird_data.sh"
    exit 1
fi

python -m training_data.generate \
    --golden-pairs "$GOLDEN_PAIRS" \
    "${DB_DIRS[@]}" \
    --output-dir "$DATA_DIR" \
    --tool-calling --tool-ratio "$TOOL_RATIO" \
    --limit "$LIMIT" \
    --verify

if $GENERATE_ONLY; then
    echo ""
    echo "=== Data generation complete. Skipping training. ==="
    exit 0
fi

# ── Step 2: Train ──
echo ""
echo "=== Starting Fine-tuning ==="
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE × $GRAD_ACCUM (grad accum)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Max seq length: $MAX_SEQ_LENGTH"
echo "  Output: $OUTPUT_DIR"

python -m finetuning.train \
    --model-name "$MODEL_NAME" \
    --train-data "$DATA_DIR/train.jsonl" \
    --dev-data "$DATA_DIR/dev.jsonl" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --per-device-train-batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-epochs "$NUM_EPOCHS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Training Complete ==="
echo "  Adapter saved to: $OUTPUT_DIR/final"
echo ""
echo "To merge adapter into base model:"
echo "  python -m finetuning.train --merge --output-dir $OUTPUT_DIR"
