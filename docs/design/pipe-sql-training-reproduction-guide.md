# Pipe SQL Fine-Tuning: Reproduction Guide

This document describes how to reproduce the pipe SQL fine-tuning pipeline end-to-end, from a fresh clone of the repository to a trained model. It covers environment setup, data preparation, training data generation, and model fine-tuning.

For the design rationale behind this system, see [pipe-sql-fine-tuning-design-doc.md](pipe-sql-fine-tuning-design-doc.md).

---

## Prerequisites

- **GPU**: NVIDIA GPU with >=16 GB VRAM (tested on RTX 4080 16 GB)
- **NVIDIA Driver**: 525+ (CUDA 12.x compatible)
- **OS**: Windows 11 or Linux (commands below use bash; on Windows, use Git Bash or WSL)
- **uv**: Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Disk**: ~15 GB for benchmark databases, ~15 GB for model weights (cached by HuggingFace)

---

## Step 1: Clone and Create Python Environment

```bash
git clone <repo-url>
cd sqlglot

# Create a Python 3.11 virtual environment
uv venv .venv --python 3.11
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # Linux/macOS
```

## Step 2: Install Dependencies

```bash
# Install sqlglot in editable mode (puts training_data/ and finetuning/ on sys.path)
uv pip install -e .

# Install PyTorch with CUDA 12.6 support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install ML training stack
uv pip install transformers peft trl datasets bitsandbytes accelerate

# For Spider dataset download (Google Drive)
uv pip install gdown
```

**Verify CUDA**:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 4080
```

> **Note**: PyTorch cu126 wheels bundle their own CUDA runtime. You do NOT need to upgrade your system CUDA toolkit — any NVIDIA driver >=525 works.

## Step 3: Download Benchmark Databases

The training data generation requires SQLite databases from Spider 1.0 and BIRD benchmarks to extract schemas.

```bash
# Spider 1.0 (~1 GB, downloads from Google Drive via gdown)
bash scripts/setup_data.sh

# BIRD dev + train sets (~9 GB, downloads via curl)
bash scripts/setup_bird_data.sh
```

**Verify**:
```bash
ls data/spider/database | wc -l           # ~166 databases
ls data/bird/train/train_databases | wc -l # ~70 databases
ls data/bird/dev_20240627/dev_databases | wc -l  # ~11 databases
```

## Step 4: Generate Training Data

This reads the 15,443 validated golden pairs (standard SQL ↔ pipe SQL) and generates incremental chat training samples. Each N-operator pipe query is decomposed into N training samples where the model learns to emit one pipe operator at a time.

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

| Flag | Description |
|------|-------------|
| `--golden-pairs` | JSONL file with `{gold_sql, pipe_sql, db_id, question_id, question}` entries |
| `--db-dir` | Directories containing SQLite databases (repeatable) |
| `--tool-calling` | Also generate agentic tool-calling training samples |
| `--tool-ratio 0.3` | 30% of golden pairs get an additional tool-calling sample |
| `--limit 2000` | Process only the first 2000 pairs (remove for full dataset) |

**Output** (`training_data_output/`):

| File | Description |
|------|-------------|
| `train.jsonl` | 6,860 training samples |
| `dev.jsonl` | 498 validation samples |
| `stats.json` | Operator distribution, step counts, database count |

With `--limit 2000`, this produces 7,358 total samples (including 579 tool-calling samples) from 1,680 unique queries across 78 databases. Remove `--limit` to process all 15,443 golden pairs for ~50K+ samples.

### Training Data Format

Each sample is a chat conversation in OpenAI format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a SQL assistant that writes pipe SQL..."},
    {"role": "user", "content": "Question: ... Schema: ... Query so far: FROM t |> WHERE ..."},
    {"role": "assistant", "content": "|> AGGREGATE COUNT(*) AS cnt GROUP BY department"}
  ]
}
```

## Step 5: Fine-Tune the Model

### Quick Start (One Command)

The `scripts/train.sh` wrapper handles data generation + training:

```bash
# Smoke test (~5 min, 1 epoch, 100 samples)
bash scripts/train.sh --smoke-test

# Full training (1.5B model, 3 epochs, ~2 hours)
bash scripts/train.sh
```

### Manual Training Commands

#### 5a. Smoke Test (1.5B, 1 epoch)

Validates the pipeline works end-to-end:

```bash
python -m finetuning.train \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --train-data training_data_output/train.jsonl \
    --dev-data training_data_output/dev.jsonl \
    --max-seq-length 4096 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --num-epochs 1 \
    --no-4bit \
    --output-dir finetuning_output_smoke
```

Expected: loss drops from ~2.1 to ~0.2, token accuracy rises to ~96%.

#### 5b. Full 1.5B Training (3 epochs)

```bash
python -m finetuning.train \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --max-seq-length 4096 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --num-epochs 3 \
    --no-4bit \
    --output-dir finetuning_output
```

#### 5c. 7B QLoRA Training (3 epochs)

For the full-size model using 4-bit quantization to fit in 16 GB VRAM:

```bash
python -m finetuning.train \
    --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
    --max-seq-length 4096 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --num-epochs 3 \
    --load-in-4bit \
    --output-dir finetuning_output_7b
```

### Key Training Parameters

| Parameter | 1.5B (float16) | 7B (QLoRA 4-bit) |
|-----------|----------------|-------------------|
| `--per-device-train-batch-size` | 4 | 1 |
| `--gradient-accumulation-steps` | 8 | 16 |
| `--load-in-4bit` / `--no-4bit` | `--no-4bit` | `--load-in-4bit` |
| Effective batch size | 32 | 16 |
| VRAM usage | ~7 GB | ~12.5 GB |
| Training time (2K samples, 3 epochs) | ~1h 44min | ~3h |

### What the Trainer Does

1. Loads the base model (Qwen2.5-Coder) with LoRA adapters targeting all attention + MLP projections
2. Applies a custom chat template with `{% generation %}` markers so loss is computed only on assistant responses (`assistant_only_loss=True`)
3. Uses gradient checkpointing to reduce VRAM usage
4. For QLoRA: uses bitsandbytes 4-bit NF4 quantization with bf16 compute
5. Saves checkpoints every 500 steps, keeps the 3 most recent
6. Restores the original Qwen chat template (with tool-call support) before saving the final adapter

## Step 6: Merge LoRA Adapter

After training, merge the LoRA adapter into the base model for standalone inference:

```bash
# For 1.5B model
python -m finetuning.train --merge \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output-dir finetuning_output

# For 7B model
python -m finetuning.train --merge \
    --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
    --output-dir finetuning_output_7b
```

The merged model is saved to `<output-dir>/merged/` and can be loaded directly with `AutoModelForCausalLM.from_pretrained()`.

> **Important**: Always specify `--model-name` matching the model used for training. The default is 7B, so for 1.5B merges you must pass it explicitly.

---

## Training Results (Reference)

Results from training with `--limit 2000` (7,358 samples) on an RTX 4080 16 GB:

### 1.5B Smoke Test (1 epoch, float16)

| Metric | Start | End |
|--------|-------|-----|
| Train loss | 2.126 | 0.200 |
| Token accuracy | 67.4% | 96.3% |
| Eval loss | — | 0.197 |
| Runtime | — | 35 min |

### 1.5B Full Training (3 epochs, float16)

| Metric | Start | End |
|--------|-------|-----|
| Train loss | 2.172 | 0.132 |
| Token accuracy | 66.9% | 98.4% |
| Eval loss | — | 0.191 |
| Runtime | — | 1h 44min |

### VRAM Budget (RTX 4080 — 16 GB)

| Model | Quantization | Model VRAM | Training Overhead | Total |
|-------|-------------|------------|-------------------|-------|
| 1.5B | float16 | ~3 GB | ~4 GB | ~7 GB |
| 7B | QLoRA 4-bit | ~4.5 GB | ~8 GB | ~12.5 GB |

---

## Project Structure

```
sqlglot/
├── validation_output/
│   └── golden_pairs_consolidated.jsonl   # 15,443 validated (gold_sql, pipe_sql) pairs
├── training_data/
│   ├── __main__.py                       # Entry: python -m training_data.generate
│   ├── generate.py                       # Main data generation pipeline
│   ├── formatter.py                      # Chat sample formatting (incremental trajectory)
│   ├── tool_formatter.py                 # Tool-calling sample generation
│   ├── trajectory.py                     # Pipe query → step decomposition
│   ├── schema_extractor.py              # SQLite schema → text representation
│   ├── tool_executor.py                 # Simulated tool execution for training
│   └── writer.py                        # Train/dev split and JSONL output
├── finetuning/
│   ├── train.py                          # Main fine-tuning script
│   ├── config.py                         # TrainConfig dataclass with CLI parsing
│   └── data.py                           # JSONL dataset loader
├── scripts/
│   ├── setup_data.sh                     # Downloads Spider 1.0
│   ├── setup_bird_data.sh                # Downloads BIRD dev + train
│   └── train.sh                          # One-command data gen + training
├── training_data_output/                 # Generated training data (not committed)
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── stats.json
├── finetuning_output/                    # Training outputs (not committed)
│   ├── checkpoint-*/                     # Intermediate checkpoints
│   ├── final/                            # Final LoRA adapter
│   └── merged/                           # Merged standalone model
└── docs/design/
    ├── pipe-sql-fine-tuning-design-doc.md
    ├── pipe-sql-decompiler-design-doc.md
    ├── pipe-sql-validation-loop-design-doc.md
    └── pipe-sql-training-reproduction-guide.md  # This file
```

---

## Troubleshooting

### BFloat16 / FP16 AMP Error with QLoRA

**Error**: `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`

**Cause**: bitsandbytes 4-bit quantization produces BFloat16 parameters, which are incompatible with the FP16 AMP gradient scaler.

**Fix**: The training script automatically detects this and uses `bf16=True` when `--load-in-4bit` is set on CUDA. If you see this error, ensure you're using the latest `finetuning/train.py`.

### Model Loading on CPU Instead of GPU

**Symptom**: Training is extremely slow; logs show "Using float32 on CPU" despite having a CUDA GPU.

**Cause**: When using `--no-4bit` on CUDA, an earlier version of the code was missing the `elif use_cuda` branch in `load_model_and_tokenizer()`.

**Fix**: The current code includes proper device detection for all CUDA modes (4-bit and float16).

### Wrong Base Model During Merge

**Symptom**: `RuntimeError` or size mismatch when running `--merge`.

**Cause**: The default `--model-name` is `Qwen/Qwen2.5-Coder-7B-Instruct`. If you trained the 1.5B model, you must specify the correct base model during merge.

**Fix**: Always pass `--model-name` matching the model used for training:
```bash
python -m finetuning.train --merge \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output-dir finetuning_output
```

### First Run Downloads Are Slow

The first time you run training, HuggingFace downloads the model weights (~3 GB for 1.5B, ~15 GB for 7B). Subsequent runs use the cached weights from `~/.cache/huggingface/`. For faster downloads, set a HuggingFace token:

```bash
huggingface-cli login
```

---

## Full Reproduction Checklist

- [ ] Python 3.11 virtual environment created
- [ ] PyTorch with CUDA support installed and verified
- [ ] Spider 1.0 databases downloaded (~166 DBs)
- [ ] BIRD databases downloaded (~81 DBs)
- [ ] Training data generated from golden pairs
- [ ] Smoke test passed (loss decreases, accuracy >90%)
- [ ] Full 1.5B training completed (eval_loss < 0.20)
- [ ] 7B QLoRA training completed
- [ ] LoRA adapters merged into standalone models
