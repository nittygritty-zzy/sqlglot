#!/usr/bin/env bash
# Regenerates the ANTLR Python parser from grammar files.
# Requires: Java 11+, antlr4-python3-runtime pip package
#
# Usage: bash tests/oracle_grammar/regenerate.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAMMAR_DIR="$SCRIPT_DIR/grammars"
GENERATED_DIR="$SCRIPT_DIR/generated"
ANTLR_VERSION="4.13.2"
ANTLR_JAR="/tmp/antlr-${ANTLR_VERSION}-complete.jar"

if [ ! -f "$ANTLR_JAR" ]; then
    echo "Downloading ANTLR $ANTLR_VERSION..."
    curl -sL "https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar" -o "$ANTLR_JAR"
fi

WORK_DIR="$(mktemp -d)"
trap "rm -rf $WORK_DIR" EXIT

cp "$GRAMMAR_DIR/PlSqlLexer.g4" "$GRAMMAR_DIR/PlSqlParser.g4" \
   "$SCRIPT_DIR/PlSqlLexerBase.py" "$SCRIPT_DIR/PlSqlParserBase.py" \
   "$WORK_DIR/"

echo "Generating Python parser..."
java -jar "$ANTLR_JAR" -Dlanguage=Python3 -visitor -o "$WORK_DIR/out" \
    "$WORK_DIR/PlSqlLexer.g4" "$WORK_DIR/PlSqlParser.g4"

# Copy generated Python files (skip .interp and .tokens)
for f in PlSqlLexer.py PlSqlParser.py PlSqlParserListener.py PlSqlParserVisitor.py; do
    cp "$WORK_DIR/out/$f" "$GENERATED_DIR/$f"
done

# Copy base classes into generated dir
cp "$SCRIPT_DIR/PlSqlLexerBase.py" "$GENERATED_DIR/"
cp "$SCRIPT_DIR/PlSqlParserBase.py" "$GENERATED_DIR/"

echo "Done. Generated files in $GENERATED_DIR"
