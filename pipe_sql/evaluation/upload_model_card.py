"""Upload model card README.md and design docs to HuggingFace."""

from huggingface_hub import HfApi
import tempfile
import os

REPO_ID = "nittygritty-zzy/pipe-sql-1.5b"

README_CONTENT = """---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
tags:
- text-to-sql
- pipe-sql
- sqlglot
- tool-calling
- qwen2
datasets:
- spider
pipeline_tag: text-generation
model-index:
- name: pipe-sql-1.5b
  results:
  - task:
      type: text-to-sql
      name: Text-to-SQL
    dataset:
      type: spider
      name: Spider 1.0 Dev
    metrics:
    - type: execution_accuracy
      value: 60.66
      name: Execution Accuracy
---

# Pipe SQL 1.5B

A fine-tuned [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) model for generating **Pipe SQL** through multi-turn tool-calling conversations.

**GitHub**: [nittygritty-zzy/sqlglot](https://github.com/nittygritty-zzy/sqlglot)

## What is Pipe SQL?

Pipe SQL is a more readable SQL syntax that uses the `|>` (pipe) operator to chain operations in a linear, top-to-bottom flow:

```sql
FROM employees
|> WHERE department = 'Engineering'
|> AGGREGATE AVG(salary) AS avg_salary GROUP BY level
|> ORDER BY avg_salary DESC
```

This is transpiled to standard SQL via [sqlglot](https://github.com/tobymao/sqlglot), an open-source SQL parser and transpiler.

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen2.5-Coder-1.5B-Instruct |
| **Architecture** | Qwen2ForCausalLM |
| **Parameters** | 1.5B |
| **Hidden Size** | 1536 |
| **Layers** | 28 |
| **Attention Heads** | 12 (2 KV heads) |
| **Context Length** | 2048 tokens (training) |

## Design Documents

The full design and methodology behind this project is documented in the following design docs (also available in [docs/design/](https://github.com/nittygritty-zzy/sqlglot/tree/main/docs/design) on GitHub):

| Document | Description |
|----------|-------------|
| [Fine-Tuning Design Doc](docs/pipe-sql-fine-tuning-design-doc.md) | End-to-end system design for incremental pipe SQL synthesis and specialized fine-tuning of 1.5B-7B models |
| [Decompiler Design Doc](docs/pipe-sql-decompiler-design-doc.md) | Standard SQL to pipe SQL decompiler — the deterministic data generation component |
| [Validation Loop Design Doc](docs/pipe-sql-validation-loop-design-doc.md) | SQLite round-trip validation and feedback loop to ensure semantic correctness |
| [Training Reproduction Guide](docs/pipe-sql-training-reproduction-guide.md) | Step-by-step guide to reproduce the full training pipeline from scratch |

## Training

The model was fine-tuned using **QLoRA** on multi-turn tool-calling conversations for text-to-SQL generation.

### Training Data

Conversations were generated from the [Spider 1.0](https://yale-lily.github.io/spider) training set, where each conversation follows an agentic workflow:
1. **Explore** the database schema using `list_tables`, `describe_table`, and `sample_data` tools
2. **Write** pipe SQL queries using `execute_pipe_sql` and `validate_pipe_sql` tools
3. **Iterate** based on execution results until the query is correct

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Method** | QLoRA (4-bit NF4) |
| **LoRA rank** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.05 |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Epochs** | 3 |
| **Learning rate** | 2e-4 |
| **LR scheduler** | Cosine |
| **Warmup ratio** | 0.05 |
| **Batch size** | 2 (per device) |
| **Gradient accumulation** | 8 steps |
| **Weight decay** | 0.01 |
| **Loss** | Assistant-only (tool responses masked) |

## Evaluation Results

Evaluated on the **Spider 1.0 dev set** (1,034 questions) using an agentic benchmark pipeline. The agent autonomously explores database schemas via tool calls, writes pipe SQL, and iterates until correct — matching the training workflow.

### Execution Accuracy

| Metric | Value |
|--------|-------|
| **Execution Accuracy** | **60.66%** (626 / 1,032) |
| **Prediction Rate** | 99.7% (1,031 / 1,034) |
| **Total Questions** | 1,034 |
| **Gold Errors Excluded** | 2 |

### Detailed Breakdown

| Status | Count | % of Total | Description |
|--------|------:|------------|-------------|
| **Match** | 626 | 60.5% | Predicted SQL produces identical results to gold SQL |
| **Mismatch** | 209 | 20.2% | SQL executes but results differ from gold |
| **Execution Error** | 170 | 16.4% | Transpiled SQL fails to execute against SQLite |
| **Transpile Error** | 24 | 2.3% | Pipe SQL cannot be transpiled to standard SQL |
| **No Prediction** | 3 | 0.3% | Agent did not produce a pipe SQL query |
| **Gold Error** | 2 | 0.2% | Reference gold SQL fails (excluded from denominator) |

### Evaluation Methodology

1. The TypeScript agent runs each question through a multi-turn tool-calling loop (max 10 turns, 120s timeout)
2. The agent's final `execute_pipe_sql` call is extracted as the predicted pipe SQL
3. Predicted pipe SQL is transpiled to standard SQL using `sqlglot.transpile()`
4. Both predicted and gold SQL are executed against the Spider SQLite databases
5. Result sets are compared using order-insensitive set comparison with numeric tolerance

> **Note**: This is an **in-distribution** evaluation — the model was trained on Spider training data, and the dev set uses the same 20 databases.

## Tools

The model was trained to use 5 tools in a multi-turn conversation:

| Tool | Description |
|------|-------------|
| `list_tables` | List all tables in a database |
| `describe_table` | Get column names, types, and constraints for a table |
| `sample_data` | Retrieve sample rows from a table |
| `execute_pipe_sql` | Execute a pipe SQL query against the database |
| `validate_pipe_sql` | Validate pipe SQL syntax without executing |

## Usage

### Chat Template

The model uses a custom chat template with `<tool_call>` tags for tool invocations:

```
<|im_start|>assistant
Let me explore the database first.
<tool_call>
list_tables({"db_id": "concert_singer"})
</tool_call><|im_end|>
```

Tool responses are formatted as:

```
<|im_start|>user
<tool_response>
Tables in database 'concert_singer':
- stadium
- singer
- concert
- singer_in_concert
</tool_response><|im_end|>
```

### Inference

For inference with the correct chat template, see the [evaluation server code](https://github.com/nittygritty-zzy/sqlglot/tree/main/pipe_sql/evaluation/server) on GitHub.

## Reproducing the Benchmark

### Prerequisites

- **GPU**: NVIDIA GPU with >= 6 GB VRAM (model runs in float16)
- **Python**: 3.11+ with pip/uv
- **Node.js**: 18+ with npm
- **Disk**: ~1 GB for Spider databases, ~3 GB for model weights

### Step 1: Clone the Repository

```bash
git clone https://github.com/nittygritty-zzy/sqlglot.git
cd sqlglot
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate       # Linux/macOS
# source .venv/Scripts/activate # Windows (Git Bash)

# Install sqlglot (editable)
uv pip install -e .

# Install evaluation server dependencies
uv pip install fastapi uvicorn pydantic

# Install PyTorch with CUDA support
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install model loading dependencies
uv pip install transformers accelerate
```

Verify CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX ...
```

### Step 3: Download Spider 1.0 Dataset

The benchmark uses Spider 1.0 dev set (1,034 questions across 20 SQLite databases).

```bash
# Install gdown for Google Drive downloads
uv pip install gdown

# Download and extract Spider 1.0 (~1 GB)
bash scripts/setup_data.sh
```

Verify:
```bash
ls data/spider/dev.json           # 1,034 questions
ls data/spider/database/ | wc -l  # ~166 databases (20 used by dev set)
```

### Step 4: Download the Model

```bash
# Option A: Use huggingface_hub (recommended)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nittygritty-zzy/pipe-sql-1.5b', local_dir='pipe_sql/finetuning_output/merged')
"

# Option B: Use git-lfs
git lfs install
git clone https://huggingface.co/nittygritty-zzy/pipe-sql-1.5b pipe_sql/finetuning_output/merged
```

### Step 5: Install Node.js Agent Dependencies

```bash
cd pipe_sql/evaluation/agent
npm install
cd ../../..
```

### Step 6: Run the Benchmark

#### Option A: Full Pipeline (Recommended)

```bash
# Run all 1,034 questions (takes ~2 hours on RTX 4080)
bash pipe_sql/evaluation/run_all.sh

# Smoke test with 5 questions first
bash pipe_sql/evaluation/run_all.sh --limit 5
```

This script:
1. Starts the Python evaluation server (model inference + tool execution)
2. Waits for the server to be ready
3. Runs the TypeScript agent benchmark
4. Evaluates results and prints execution accuracy

#### Option B: Run Components Separately

**Start the evaluation server:**
```bash
# Default: loads model from pipe_sql/finetuning_output/merged/
python -m pipe_sql.evaluation.server.app

# Custom model path:
MODEL_PATH=path/to/model python -m pipe_sql.evaluation.server.app
```

Wait for `Server ready` in the logs, then in a separate terminal:

**Run the agent benchmark:**
```bash
cd pipe_sql/evaluation/agent
npx tsx src/main.ts --benchmark           # All 1,034 questions
npx tsx src/main.ts --benchmark --limit 5 # Smoke test
```

**Run single question interactively:**
```bash
cd pipe_sql/evaluation/agent
npx tsx src/main.ts "How many singers do we have?" concert_singer
```

**Evaluate results:**
```bash
python pipe_sql/evaluation/evaluate.py --results pipe_sql/output/results.json
```

### Step 7: Review Results

Results are saved to `pipe_sql/output/`:

| File | Description |
|------|-------------|
| `results.json` | Agent predictions with conversation traces |
| `eval_results.json` | Per-question evaluation details (match/mismatch/error) |
| `eval_summary.json` | Aggregate metrics |

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_PATH` | `pipe_sql/finetuning_output/merged` | Path to merged model directory |
| `SPIDER_DB_DIR` | `data/spider/database` | Spider database directory |
| `SPIDER_DIR` | `data/spider` | Spider data directory (contains dev.json) |
| `PORT` | `8000` | Evaluation server port |
| `SERVER_URL` | `http://localhost:8000` | Agent to server connection URL |
| `OUTPUT_DIR` | `pipe_sql/output` | Agent output directory |

### Troubleshooting

**Server fails to load model**: Ensure `pipe_sql/finetuning_output/merged/` contains `config.json`, `model.safetensors`, and `tokenizer.json`. If using a different path, set `MODEL_PATH`.

**CUDA out of memory**: The 1.5B model needs ~3 GB VRAM in float16. Close other GPU processes or use `CUDA_VISIBLE_DEVICES=0` to select a specific GPU.

**Agent produces garbled tool calls**: The 1.5B model sometimes generates garbled special tokens instead of proper `<tool_call>` tags. The inference server includes fallback parsing for bare function calls — this is handled automatically.

**Spider databases not found**: Run `bash scripts/setup_data.sh` to download Spider 1.0. The script downloads from Google Drive via `gdown`.

## Limitations

- Trained and evaluated only on Spider 1.0 (SQLite databases)
- Context window limited to 2,048 tokens during training
- The 1.5B model may generate garbled special tokens instead of proper `<tool_call>` tags — the inference server includes fallback parsing for bare function calls
- Performance on out-of-distribution databases (different schemas/domains) has not been extensively tested
- This is an in-distribution evaluation; real-world performance on unseen databases will likely be lower

## License

This model is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0), consistent with the base Qwen2.5-Coder model license.
"""

# Design docs to upload (local path -> repo path)
DESIGN_DOCS = [
    ("docs/design/pipe-sql-fine-tuning-design-doc.md", "docs/pipe-sql-fine-tuning-design-doc.md"),
    ("docs/design/pipe-sql-decompiler-design-doc.md", "docs/pipe-sql-decompiler-design-doc.md"),
    ("docs/design/pipe-sql-validation-loop-design-doc.md", "docs/pipe-sql-validation-loop-design-doc.md"),
    ("docs/design/pipe-sql-training-reproduction-guide.md", "docs/pipe-sql-training-reproduction-guide.md"),
]


if __name__ == "__main__":
    api = HfApi()

    # Upload README
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(README_CONTENT)
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        commit_message="Update model card: add GitHub link, design docs, and benchmark setup guide",
    )
    os.unlink(tmp_path)
    print("README.md uploaded")

    # Upload design docs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for local_rel, repo_path in DESIGN_DOCS:
        local_path = os.path.join(project_root, local_rel)
        if not os.path.isfile(local_path):
            print(f"  WARNING: {local_path} not found, skipping")
            continue
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            commit_message=f"Add design doc: {os.path.basename(local_path)}",
        )
        print(f"  Uploaded {repo_path}")

    print("\nAll files uploaded successfully!")
