# Pipe SQL Training Journey: Implementation Record

This document records the actual implementation of the pipe SQL fine-tuning pipeline described in the [design document](pipe-sql-fine-tuning-design-doc.md). It covers what was built, how to run it, and the results observed at each phase.

---

## Pipeline Overview

```
Phase 1              Phase 2               Phase 3                Phase 4
Decompiler    →    Validation Loop    →    Training Data    →    Fine-Tuning
pipe_decompiler/      validation/        training_data/         finetuning/

Standard SQL        Gold SQL ↔ Pipe SQL    Trajectory +          LoRA on
→ Pipe SQL          semantic equivalence   Tool-calling samples   Qwen 1.5B/7B
```

**Status**: Phases 1–3 complete. Phase 4 (fine-tuning) in progress with two backends operational.

---

## Phase 1: Pipe SQL Decompiler

**Package**: `pipe_decompiler/`

Converts standard SQL into pipe SQL by walking SQLGlot's qualified AST and emitting a linear chain of `|>` operators.

**Entry point**: `emit_pipe_query(ast, dialect)` → `PipeQuery` (typed list of `PipeOperator` objects)
**Serializer**: `serialize(pipe_query)` → pipe SQL string

The emitter follows a canonical operator ordering:

```
FROM → JOINs → WHERE (pre-agg) → EXTEND (computed cols) →
AGGREGATE → WHERE (post-agg) → EXTEND (windows) →
WHERE (post-window) → SELECT (final projection) →
ORDER BY → LIMIT
```

---

## Phase 2: Validation Loop

**Package**: `validation/`

Validates semantic equivalence via round-trip: Gold SQL → Pipe SQL → SQLGlot transpile back to standard SQL → execute both → compare result sets.

**Output**: `validation_output/golden_pairs_consolidated.jsonl`
- 15,443 validated pairs across 245 databases
- Sources: Spider 1.0 (166 DBs) + BIRD train (70 DBs) + BIRD dev (11 DBs)

Each record:
```json
{
  "question_id": "...",
  "db_id": "...",
  "difficulty": "easy|medium|hard|extra",
  "gold_sql": "SELECT ...",
  "pipe_sql": "FROM ... |> WHERE ... |> SELECT ...",
  "round_tripped_sql": "SELECT ...",
  "validation": "pass",
  "question": "Which departments have average salary above 80K?"
}
```

---

## Phase 3: Training Data Generation

**Package**: `training_data/`

Generates two types of training samples in OpenAI chat JSONL format:

### Trajectory Samples (~62K from full dataset)

One-step-at-a-time pipe operator prediction. Each N-operator pipe query decomposes into N training samples:

```json
{
  "messages": [
    {"role": "system", "content": "You are a SQL assistant..."},
    {"role": "user", "content": "Question: ...\nSchema: ...\nQuery so far: FROM t |> WHERE ..."},
    {"role": "assistant", "content": "|> AGGREGATE AVG(salary) AS avg_sal GROUP BY department"}
  ]
}
```

### Tool-Calling Samples (~4.5K at 30% ratio)

Multi-turn explore → generate → verify conversations using 5 tools:
- `list_tables` — enumerate tables in a database
- `describe_table` — get column names, types, and sample values
- `sample_data` — preview rows from a table
- `execute_pipe_sql` — run a pipe query and return results
- `validate_pipe_sql` — check pipe SQL syntax via SQLGlot parse

Three conversation patterns (A/B/C) based on difficulty, all ending with `execute_pipe_sql` verification.

### Generation Command

```bash
python -m training_data.generate \
    --golden-pairs validation_output/golden_pairs_consolidated.jsonl \
    --db-dir data/spider/database \
    --db-dir data/bird/train/train_databases \
    --db-dir data/bird/dev_20240627/dev_databases \
    --output-dir training_data_output \
    --tool-calling --tool-ratio 0.3 \
    --limit 2000
```

**Output**: `training_data_output/{train,dev}.jsonl` + `stats.json`

Key details:
- Train/dev split is stratified by `db_id`, grouped by `question_id`
- Tool-calling samples include a `tools` field; trajectory samples do not
- `--limit 2000` → ~9K samples; no limit → ~66K samples
- p90 sequence length: ~1,500 tokens; p99: ~3,500 tokens

---

## Phase 4: Fine-Tuning

**Package**: `finetuning/`

Two training backends are available:

### Backend 1: Transformers + PEFT (CUDA / MPS / CPU)

**Module**: `finetuning/train.py`

Uses HuggingFace TRL's `SFTTrainer` with LoRA/QLoRA. Best suited for CUDA GPUs.

**Key features**:
- Custom Jinja chat template with `{% generation %}` markers for assistant-only loss
- QLoRA 4-bit on CUDA, float16 LoRA on MPS, float32 on CPU
- Handles system/user/assistant/tool roles (tool responses get no loss)
- Saves original tokenizer chat template (with tool_calls support) after training

**Usage**:
```bash
# CUDA GPU (recommended)
python -m finetuning.train \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --max-seq-length 4096 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --num-epochs 3 \
    --output-dir finetuning_output

# Merge adapter
python -m finetuning.train --merge --output-dir finetuning_output
```

### Backend 2: MLX-LM (Apple Silicon)

**Module**: `finetuning/train_mlx.py`

Native Apple Silicon training via MLX-LM with automatic 4-bit quantization and LoRA. Dramatically faster than PyTorch MPS.

**Pipeline**:
1. **Data prep** — reads `train.jsonl`/`dev.jsonl`, strips `metadata`, writes MLX-LM format (`train.jsonl`/`valid.jsonl`)
2. **Quantize** — `mlx_lm convert` creates a local 4-bit quantized model (skips if already exists)
3. **Train** — `mlx_lm lora` with `--mask-prompt` (built-in assistant-only loss)
4. **Fuse** — `mlx_lm fuse` merges adapter into base model

**Usage**:
```bash
# Smoke test
python -m finetuning.train_mlx \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --limit 100 --num-epochs 1

# Full training (recommended settings for Apple Silicon)
python -m finetuning.train_mlx \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --limit 2000 --num-epochs 3 \
    --batch-size 1 --grad-accumulation-steps 4 \
    --grad-checkpoint

# Fuse adapter into base model
python -m finetuning.train_mlx --fuse --output-dir finetuning_output_mlx
```

**Design decisions**:
- Uses `subprocess.run()` to call `mlx_lm` CLI (documented interface; internal API is not stable)
- LoRA rank configured via YAML config file (not exposed as CLI flag)
- Iteration count computed as `(num_samples × num_epochs) / batch_size`

### Wrapper Script

`scripts/train.sh` handles both data generation and training:

```bash
bash scripts/train.sh                    # Transformers+PEFT (CUDA)
bash scripts/train.sh --mlx              # MLX-LM (Apple Silicon)
bash scripts/train.sh --smoke-test       # Quick test (100 samples, 1 epoch)
bash scripts/train.sh --smoke-test --mlx # Quick test with MLX
bash scripts/train.sh --generate-only    # Only regenerate training data
```

---

## Performance Benchmarks

### Training Speed by Backend

| Backend | Device | Batch | Speed | Memory | Notes |
|---|---|---|---|---|---|
| **MLX-LM** | Apple Silicon (M-series) | 1 × 4 grad_accum | **~1 it/sec** | ~22 GB (smoke), ~25-30 GB (full) | Recommended for Mac |
| **Transformers+PEFT** | CUDA (RTX 4080) | 4 × 8 grad_accum | ~2-4 s/step | ~16 GB (QLoRA 4-bit) | Recommended for full training |
| **Transformers+PEFT** | MPS (Mac) | 1 × 16 grad_accum | ~37 s/step | ~20 GB (float16) | Smoke tests only |

### Wall-Clock Estimates (Qwen 1.5B)

| Scenario | Samples | Backend | Estimated Time |
|---|---|---|---|
| `--limit 2000` | ~9K | CUDA (RTX 4080) | ~40 min |
| `--limit 2000` | ~9K | MLX-LM (Apple Silicon) | ~1.5–3 hrs |
| Full (no limit) | ~66K | CUDA (RTX 4080) | ~10 hrs |
| Full (no limit) | ~66K | MLX-LM (Apple Silicon) | ~10–18 hrs |

### Smoke Test Results

**MLX-LM** (100 samples, 1 epoch, batch=4, 25 iterations):
- Loss: 1.219 → 0.390 (iter 10→20)
- Val loss: 3.502 → 0.371
- Throughput: ~1.75 it/sec (short sequences)
- Peak memory: 22 GB
- Total time: ~88s

**Transformers+PEFT** (100 samples, 1 epoch, MPS):
- Loss: 1.9 → 0.15 in 25 steps
- Eval loss: 0.46
- Speed: ~37 s/step

---

## Memory Considerations

MLX-LM on Apple Silicon shares unified memory between CPU and GPU. Long sequences (p99 ~3,500 tokens) at batch_size=4 can hit 39+ GB and OOM on machines with 36 GB unified memory.

**Recommended settings for machines with ≤36 GB**:
- `--batch-size 1 --grad-accumulation-steps 4` (effective batch = 4)
- `--grad-checkpoint` (gradient checkpointing trades compute for memory)
- `--max-seq-length 4096` (covers p99; increasing further wastes memory)

---

## Data Locations

| Artifact | Path |
|---|---|
| Spider databases | `data/spider/database/{db_id}/{db_id}.sqlite` (166 DBs) |
| BIRD train databases | `data/bird/train/train_databases/{db_id}/{db_id}.sqlite` (70 DBs) |
| BIRD dev databases | `data/bird/dev_20240627/dev_databases/{db_id}/{db_id}.sqlite` (11 DBs) |
| Golden pairs | `validation_output/golden_pairs_consolidated.jsonl` |
| Training data | `training_data_output/{train,dev}.jsonl` |
| Training stats | `training_data_output/stats.json` |

Setup:
```bash
bash scripts/setup_data.sh        # Spider
bash scripts/setup_bird_data.sh   # BIRD
```

---

## Technical Notes

- **Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct (fast, sufficient for narrow task). Also supports 7B variant (better quality, much slower).
- **BIRD question_ids** are integers, not strings — sort with `key=str`.
- **109 golden pairs** have no matching DB (minor data gaps, silently skipped).
- **Trajectory decomposition** uses `emit_pipe_query()` with string-split fallback for edge cases.
- **One outlier sample** has 360K tokens (huge table dump) — gets truncated at `max_seq_length`, harmless.
- **MLX-LM** invoked via CLI subprocess (`python -m mlx_lm <command>`) rather than Python API for stability.

---

## What's Next

- [ ] Complete `--limit 2000` training run on Apple Silicon with MLX-LM
- [ ] Evaluate trained model on held-out dev set (execution accuracy)
- [ ] Full dataset training on CUDA GPU
- [ ] GRPO reinforcement learning pass (Phase 5 in design doc)
- [ ] Agentic inference integration with tool loop (Section 9 in design doc)
