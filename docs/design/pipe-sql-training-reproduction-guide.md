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
# Full dataset (recommended for production training)
python -m training_data.generate \
    --golden-pairs validation_output/golden_pairs_consolidated.jsonl \
    --db-dir data/spider/database \
    --db-dir data/bird/train/train_databases \
    --db-dir data/bird/dev_20240627/dev_databases \
    --output-dir training_data_output \
    --tool-calling --tool-ratio 0.3

# Subset for quick iteration (add --limit)
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
| `--limit N` | Process only the first N pairs (omit for full dataset) |

**Expected output**:

| Input | Total Samples | Train (95%) | Dev (5%) | Tool-calling |
|-------|--------------|-------------|----------|--------------|
| `--limit 2000` | ~7,400 | ~6,900 | ~500 | ~580 |
| All 15,443 pairs | ~57,000 | ~54,000 | ~2,800 | ~4,600 |

Each golden pair produces ~3.7 training samples on average (trajectory decomposition amplification). Output files: `train.jsonl`, `dev.jsonl`, `stats.json`.

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

#### 5a. Smoke Test (1.5B, 1 epoch, small subset)

Validates the pipeline works end-to-end. Use a small dataset generated with `--limit 2000`:

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

#### 5b. Full 1.5B Training (recommended: full dataset, 2 epochs)

```bash
python -m finetuning.train \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --train-data training_data_output/train.jsonl \
    --dev-data training_data_output/dev.jsonl \
    --max-seq-length 4096 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --num-epochs 2 \
    --no-4bit \
    --output-dir finetuning_output_1.5b
```

#### 5c. 7B QLoRA Training (recommended: full dataset, 2 epochs)

For the full-size model using 4-bit quantization to fit in 16 GB VRAM:

```bash
python -m finetuning.train \
    --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train-data training_data_output/train.jsonl \
    --dev-data training_data_output/dev.jsonl \
    --max-seq-length 4096 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 32 \
    --learning-rate 5e-5 \
    --num-epochs 2 \
    --load-in-4bit \
    --save-steps 1000 \
    --eval-steps 1000 \
    --output-dir finetuning_output_7b
```

> **Important**: The lower learning rate (5e-5 vs default 2e-4) is critical for 7B stability. An earlier run with 2e-4 collapsed to NaN at epoch ~1.5. See the Troubleshooting section for details.

### Recommended Configurations

The table below shows recommended settings for both dataset sizes. With the full dataset (15,443 pairs → ~54K train samples), 2 epochs is optimal — 7.7x more data reduces overfitting risk, and eval loss plateaus by epoch 2. With the smaller subset, 3 epochs compensates for limited data.

**1.5B (float16, `--no-4bit`)**:

| Parameter | Subset (2K pairs) | Full (15K pairs) |
|-----------|-------------------|-------------------|
| `--num-epochs` | 3 | **2** |
| `--per-device-train-batch-size` | 4 | 4 |
| `--gradient-accumulation-steps` | 8 | 8 |
| Effective batch size | 32 | 32 |
| Steps/epoch | ~215 | ~1,690 |
| Total steps | ~645 | ~3,380 |
| VRAM usage | ~7 GB | ~7 GB |
| Est. time (RTX 4080) | ~1h 44min | **~3.5 hours** |

**7B QLoRA (4-bit, `--load-in-4bit`)**:

| Parameter | Subset (2K pairs) | Full (15K pairs) |
|-----------|-------------------|-------------------|
| `--num-epochs` | 2 | **2** |
| `--per-device-train-batch-size` | 1 | 1 |
| `--gradient-accumulation-steps` | 32 | **32** |
| `--learning-rate` | **5e-5** | **5e-5** |
| Effective batch size | 32 | **32** |
| `--save-steps` / `--eval-steps` | 500 | **1000** |
| Steps/epoch | ~429 | ~1,690 |
| Total steps | ~858 | ~3,380 |
| VRAM usage | ~12.5 GB | ~12.5 GB |
| Est. time (RTX 4080) | ~2 hours | **~17 hours** |

> **Note**: Earlier runs with `--learning-rate 2e-4` and `--gradient-accumulation-steps 16` over 3 epochs caused a training collapse at epoch ~1.5 (loss → NaN). The settings above reflect the corrected configuration.

> **Tip**: Run 1.5B first as a quick validation (~3.5h). If eval loss improves over the subset baseline (0.191), the full dataset is working well. Then kick off the 7B overnight.

### Why 2 Epochs for Full Dataset?

With the 2K subset (3 epochs), we observed:
- Train loss 0.132 vs eval loss 0.191 → gap of 0.059 indicates mild overfitting
- Eval loss plateaued between epoch 2 and 3

With 7.7x more training data, the model sees far more diverse examples per epoch. 2 epochs provides sufficient coverage while avoiding diminishing returns. More data > more epochs.

### Why grad_accum=32 for Full 7B?

Doubling gradient accumulation from 16 to 32 (effective batch 32) halves the number of optimizer steps while keeping total forward/backward passes identical. Each optimizer step uses a lower-variance gradient estimate, giving more stable training. This doesn't change wall-clock time but produces better-calibrated updates.

### What the Trainer Does

1. Loads the base model (Qwen2.5-Coder) with LoRA adapters targeting all attention + MLP projections (r=16, alpha=32)
2. Applies a custom chat template with `{% generation %}` markers so loss is computed only on assistant responses (`assistant_only_loss=True`)
3. Uses gradient checkpointing to reduce VRAM usage
4. For QLoRA: uses bitsandbytes 4-bit NF4 quantization with bf16 compute
5. Saves checkpoints periodically, keeps the 3 most recent
6. Restores the original Qwen chat template (with tool-call support) before saving the final adapter

## Step 6: Merge LoRA Adapter

After training, merge the LoRA adapter into the base model for standalone inference:

```bash
# For 1.5B model
python -m finetuning.train --merge \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output-dir finetuning_output_1.5b

# For 7B model
python -m finetuning.train --merge \
    --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
    --output-dir finetuning_output_7b
```

The merged model is saved to `<output-dir>/merged/` and can be loaded directly with `AutoModelForCausalLM.from_pretrained()`.

> **Important**: Always specify `--model-name` matching the model used for training. The default is 7B, so for 1.5B merges you must pass it explicitly.

---

## Training Results (Reference)

All results on RTX 4080 16 GB, subset dataset (2K pairs → 7,358 samples).

### 1.5B Smoke Test (1 epoch, float16)

| Metric | Start | End |
|--------|-------|-----|
| Train loss | 2.126 | 0.200 |
| Token accuracy | 67.4% | 96.1% |
| Steps | — | 429 |
| Runtime | — | ~35 min |

Smooth training curve. No eval configured (single epoch validation run).

### 1.5B Full (3 epochs, float16)

| Metric | Start | End |
|--------|-------|-----|
| Train loss | 2.172 | 0.191 |
| Token accuracy | 66.9% | 97.7% |
| Best eval loss | — | **0.191** (step 500, epoch 2.3) |
| Eval token accuracy | — | 95.8% |
| Steps | — | 645 |
| Runtime | — | ~1h 44min |

Training converged well. Best checkpoint at step 500. Final train loss (0.132 at step 630) vs eval loss (0.191) shows a gap of 0.059, indicating mild overfitting in the third epoch. LoRA adapter merged successfully.

### 7B QLoRA (3 epochs, 4-bit) — FAILED (Training Collapse)

| Metric | Start | Best (step 500) | Collapse (step 680) |
|--------|-------|-----------------|---------------------|
| Train loss | 2.271 | 0.253 | **7.05 → NaN** |
| Token accuracy | 66.5% | 97.4% | **58.6% → 0.0%** |
| Eval loss | — | **0.224** | NaN (step 1000) |
| Eval token accuracy | — | 95.8% | 0.0% |
| Grad norm | 0.11 | 0.031 | **NaN** |
| Steps | — | 500/1287 | 680/1287 |

**What happened**: Training progressed normally through step 610 (epoch ~1.42), then catastrophically collapsed:

| Step | Epoch | Loss | Accuracy | Grad Norm |
|------|-------|------|----------|-----------|
| 610 | 1.42 | 0.25 | 96.6% | 0.24 |
| 620 | 1.45 | 0.87 | 90.3% | 0.92 |
| 630 | 1.47 | 2.15 | 72.6% | 2.47 |
| 640 | 1.49 | 2.77 | 67.8% | 1.66 |
| 650 | 1.52 | 3.52 | 55.9% | 1.66 |
| 660 | 1.54 | 3.70 | 45.0% | 1.84 |
| 670 | 1.56 | 3.95 | 54.6% | 0.86 |
| 680 | 1.59 | **7.05** | 58.6% | **NaN** |
| 690+ | 1.61+ | 0.0 | 0.0% | NaN |

The model weights went to NaN at step 680 and remained dead for the remaining ~600 steps. The loss spike correlates with gradient norm explosion (0.24 → 2.47 over 20 steps).

**Likely causes**:
1. Learning rate (2e-4) too aggressive for the 7B model
2. Batch size of 1 (even with grad_accum=16) causes high gradient variance
3. Possible numerical instability in 4-bit quantization + bf16 compute

**Salvageable**: The **checkpoint-500** (before collapse) is still viable — eval_loss=0.224, accuracy=95.8%. To use it:
```bash
python -m finetuning.train --merge \
    --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
    --output-dir finetuning_output_7b \
    --checkpoint checkpoint-500
```

**Recommended fixes for re-training** (see Troubleshooting section below):
- Lower learning rate to 5e-5
- Increase gradient accumulation to 32 (effective batch 32)
- Add explicit gradient clipping (`max_grad_norm=0.5`)

### Full Dataset Expectations (15K pairs → ~57K samples)

With 7.7x more data, we expect:
- **Lower eval loss** than the 0.191 subset baseline (better generalization from more diverse examples)
- **Smaller train-eval gap** (less overfitting with 2 epochs on more data)
- **1.5B**: ~3.5 hours for 2 epochs
- **7B QLoRA**: ~17 hours for 2 epochs (best run overnight)
- **Important**: Use the reduced learning rate (5e-5) and higher grad_accum (32) for 7B to avoid the collapse observed in the subset run

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

### 7B QLoRA Training Collapse (Loss → NaN)

**Symptom**: Training loss spikes dramatically around epoch 1.4–1.6, gradient norm explodes, then all metrics go to NaN/0.0 for the remaining steps.

**Cause**: The combination of a high learning rate (2e-4), small per-device batch size (1), and 4-bit quantization creates conditions for numerical instability. A single bad gradient update can cascade — once gradient norms exceed ~1.0, the model enters an irrecoverable divergence loop that ends in NaN weights.

**Fix**: Apply all three mitigations:

```bash
python -m finetuning.train \
    --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train-data training_data_output/train.jsonl \
    --dev-data training_data_output/dev.jsonl \
    --max-seq-length 4096 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 32 \
    --num-epochs 2 \
    --learning-rate 5e-5 \
    --load-in-4bit \
    --save-steps 500 \
    --eval-steps 500 \
    --output-dir finetuning_output_7b
```

Key changes from the failed run:
| Parameter | Failed Run | Recommended |
|-----------|-----------|-------------|
| `--learning-rate` | 2e-4 (default) | **5e-5** |
| `--gradient-accumulation-steps` | 16 | **32** |
| `--num-epochs` | 3 | **2** |
| `max_grad_norm` | 1.0 (default) | **0.5** (if supported) |

**Recovery**: If training has already collapsed, the last good checkpoint before the spike is still usable. Check `trainer_state.json` in each checkpoint directory — look for the last one with normal loss values and merge from there.

### First Run Downloads Are Slow

The first time you run training, HuggingFace downloads the model weights (~3 GB for 1.5B, ~15 GB for 7B). Subsequent runs use the cached weights from `~/.cache/huggingface/`. For faster downloads, set a HuggingFace token:

```bash
huggingface-cli login
```

---

## Full Reproduction Checklist

- [x] Python 3.11 virtual environment created
- [x] PyTorch with CUDA support installed and verified
- [x] Spider 1.0 databases downloaded (~166 DBs)
- [x] BIRD databases downloaded (~81 DBs)
- [x] Training data generated from golden pairs
- [x] Smoke test passed (1.5B, 1 epoch — loss 2.13→0.20, accuracy 96.1%)
- [x] Full 1.5B training completed (3 epochs — eval_loss=0.191, accuracy 95.8%)
- [x] 1.5B LoRA adapter merged (`finetuning_output/merged/`)
- [ ] 7B QLoRA training — **collapsed at epoch 1.5** (checkpoint-500 salvageable, needs re-run with lower LR)
- [ ] 7B LoRA adapter merged
- [ ] Full dataset training (15K pairs) — pending 7B fix
