#!/usr/bin/env bash
# Download and set up BIRD benchmark datasets for validation and training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/bird"

BIRD_DEV_URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_TRAIN_URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"

echo "=== BIRD Benchmark Dataset Setup ==="

# ── Dev set ──────────────────────────────────────────────────────────────────
DEV_DIR="$DATA_DIR/dev_20240627"
if [ -d "$DEV_DIR/dev_databases" ]; then
    echo "BIRD dev dataset already exists at $DEV_DIR"
else
    mkdir -p "$DATA_DIR"
    TMPFILE="$(mktemp /tmp/bird_dev_XXXXXX.zip)"
    trap 'rm -f "$TMPFILE"' EXIT

    echo "Downloading BIRD dev set..."
    curl -L -o "$TMPFILE" "$BIRD_DEV_URL"

    echo "Extracting to $DEV_DIR..."
    unzip -q "$TMPFILE" -d "$DATA_DIR"
    rm -rf "$DATA_DIR/__MACOSX"

    # Extract nested dev_databases.zip
    if [ -f "$DEV_DIR/dev_databases.zip" ]; then
        echo "Extracting dev databases..."
        unzip -q "$DEV_DIR/dev_databases.zip" -d "$DEV_DIR"
        rm -f "$DEV_DIR/dev_databases.zip"
        rm -rf "$DEV_DIR/__MACOSX"
    fi

    rm -f "$TMPFILE"
    trap - EXIT

    DEV_DB_COUNT=$(find "$DEV_DIR/dev_databases" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    echo "  Dev databases: $DEV_DB_COUNT"
fi

# ── Train set ────────────────────────────────────────────────────────────────
TRAIN_DIR="$DATA_DIR/train"
TRAIN_DB_DIR="$TRAIN_DIR/train_databases"
if [ -d "$TRAIN_DB_DIR" ] && [ "$(find "$TRAIN_DB_DIR" -name '*.sqlite' | head -1)" ]; then
    EXISTING=$(find "$TRAIN_DB_DIR" -name '*.sqlite' | wc -l | tr -d ' ')
    echo "BIRD train databases already exist at $TRAIN_DB_DIR ($EXISTING databases)"
else
    mkdir -p "$TRAIN_DIR"
    TMPFILE="$(mktemp /tmp/bird_train_XXXXXX.zip)"
    trap 'rm -f "$TMPFILE"' EXIT

    echo "Downloading BIRD train set (~8.3 GB)..."
    curl -L -o "$TMPFILE" "$BIRD_TRAIN_URL"

    echo "Extracting outer archive..."
    unzip -q "$TMPFILE" "train/train_databases.zip" -d "$DATA_DIR"
    rm -f "$TMPFILE"
    trap - EXIT

    # Extract the nested train_databases.zip
    if [ -f "$TRAIN_DIR/train_databases.zip" ]; then
        echo "Extracting train databases (~9 GB uncompressed)..."
        unzip -q "$TRAIN_DIR/train_databases.zip" -d "$TRAIN_DIR"
        rm -f "$TRAIN_DIR/train_databases.zip"
        rm -rf "$TRAIN_DIR/__MACOSX"
    fi

    TRAIN_DB_COUNT=$(find "$TRAIN_DB_DIR" -name '*.sqlite' | wc -l | tr -d ' ')
    echo "  Train databases: $TRAIN_DB_COUNT"
fi

echo ""
echo "=== Setup Complete ==="
echo "  Dev databases:   $DEV_DIR/dev_databases/"
echo "  Train databases: $TRAIN_DB_DIR/"
echo ""
echo "Generate training data with:"
echo "  python -m training_data.generate \\"
echo "      --golden-pairs validation_output/golden_pairs_consolidated.jsonl \\"
echo "      --db-dir data/spider/database \\"
echo "      --db-dir $TRAIN_DB_DIR \\"
echo "      --db-dir $DEV_DIR/dev_databases \\"
echo "      --output-dir training_data_output"
