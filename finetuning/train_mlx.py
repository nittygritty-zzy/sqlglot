"""Fine-tune Qwen-2.5-Coder on pipe SQL training data using MLX-LM (Apple Silicon).

MLX-LM provides native Apple Silicon acceleration with automatic QLoRA support,
yielding ~5-10x speedup over PyTorch MPS for LoRA fine-tuning.

Usage:
    python -m finetuning.train_mlx [--flags]              # train
    python -m finetuning.train_mlx --fuse                  # merge adapter into base model
    python -m finetuning.train_mlx --limit 100 --num-epochs 1  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import typing as t
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(
    train_path: str,
    dev_path: str,
    output_dir: str,
    limit: t.Optional[int] = None,
) -> int:
    """Read our train/dev JSONL, strip metadata, write MLX-LM format.

    MLX-LM expects a data directory with train.jsonl and valid.jsonl,
    each line containing {"messages": [...]}.

    Returns the number of training samples written.
    """
    os.makedirs(output_dir, exist_ok=True)

    num_train = 0
    for src, dst_name in [(train_path, "train.jsonl"), (dev_path, "valid.jsonl")]:
        if not os.path.exists(src):
            logger.warning(f"Source file not found, skipping: {src}")
            continue

        dst = os.path.join(output_dir, dst_name)
        count = 0
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                if limit and count >= limit:
                    break
                row = json.loads(line)
                # MLX-LM only needs the messages field
                out = {"messages": row["messages"]}
                # Include tools if present (for tool-calling samples)
                if "tools" in row:
                    out["tools"] = row["tools"]
                fout.write(json.dumps(out) + "\n")
                count += 1

        logger.info(f"Wrote {count} samples to {dst}")
        if dst_name == "train.jsonl":
            num_train = count

    return num_train


def quantize_model(model_name: str, output_dir: str, bits: int = 4) -> str:
    """Convert a HuggingFace model to MLX 4-bit quantized format.

    Returns the path to the quantized model directory.
    """
    # Use a clean directory name based on the model
    model_short = model_name.replace("/", "--")
    quantized_dir = os.path.join(output_dir, f"{model_short}-{bits}bit-mlx")

    if os.path.exists(os.path.join(quantized_dir, "config.json")):
        logger.info(f"Quantized model already exists: {quantized_dir}")
        return quantized_dir

    logger.info(f"Quantizing {model_name} to {bits}-bit MLX format...")
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", model_name,
        "--mlx-path", quantized_dir,
        "-q",
        "--q-bits", str(bits),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"Quantized model saved to: {quantized_dir}")
    return quantized_dir


def train(
    model_path: str,
    data_dir: str,
    num_train_samples: int,
    num_epochs: int = 3,
    batch_size: int = 4,
    lora_rank: int = 16,
    learning_rate: float = 2e-4,
    max_seq_length: int = 4096,
    output_dir: str = "finetuning_output",
    seed: int = 42,
) -> None:
    """Run MLX-LM LoRA fine-tuning."""
    adapter_dir = os.path.join(output_dir, "adapters")
    os.makedirs(adapter_dir, exist_ok=True)

    # MLX uses --iters not epochs
    iters = max((num_train_samples * num_epochs) // batch_size, 1)
    steps_per_eval = max(iters // 10, 50)
    steps_per_report = max(iters // 50, 10)

    # Write a YAML config for LoRA rank (not exposed as CLI flag)
    config_path = os.path.join(output_dir, "lora_config.yaml")
    lora_config = {
        "lora_parameters": {
            "rank": lora_rank,
            "alpha": lora_rank * 2,
            "dropout": 0.05,
            "scale": 1.0,
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(lora_config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--data", data_dir,
        "--train",
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", "16",
        "--learning-rate", str(learning_rate),
        "--steps-per-eval", str(steps_per_eval),
        "--steps-per-report", str(steps_per_report),
        "--adapter-path", adapter_dir,
        "--max-seq-length", str(max_seq_length),
        "--mask-prompt",
        "--seed", str(seed),
        "-c", config_path,
    ]

    logger.info(f"Training for {iters} iterations ({num_train_samples} samples × {num_epochs} epochs / {batch_size} batch)")
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"Training complete. Adapter saved to: {adapter_dir}")


def fuse(
    model_path: str,
    output_dir: str,
) -> None:
    """Fuse LoRA adapter into base model."""
    adapter_dir = os.path.join(output_dir, "adapters")
    fused_dir = os.path.join(output_dir, "fused")

    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_path,
        "--adapter-path", adapter_dir,
        "--save-path", fused_dir,
    ]

    logger.info(f"Fusing adapter into base model...")
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"Fused model saved to: {fused_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with MLX-LM (Apple Silicon)")

    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--train-data", default="training_data_output/train.jsonl")
    parser.add_argument("--dev-data", default="training_data_output/dev.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples")

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--quantize-bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--output-dir", default="finetuning_output_mlx")

    parser.add_argument("--fuse", action="store_true", help="Fuse LoRA adapter into base model")

    args = parser.parse_args()

    # Prepare MLX data directory
    mlx_data_dir = os.path.join(args.output_dir, "data")

    if args.fuse:
        # Find quantized model path
        model_short = args.model_name.replace("/", "--")
        quantized_dir = os.path.join(args.output_dir, f"{model_short}-{args.quantize_bits}bit-mlx")
        if not os.path.exists(quantized_dir):
            logger.error(f"Quantized model not found at {quantized_dir}. Run training first.")
            sys.exit(1)
        fuse(model_path=quantized_dir, output_dir=args.output_dir)
        return

    # Step 1: Prepare data
    logger.info("=== Preparing data for MLX-LM ===")
    num_train = prepare_data(
        train_path=args.train_data,
        dev_path=args.dev_data,
        output_dir=mlx_data_dir,
        limit=args.limit,
    )
    if num_train == 0:
        logger.error("No training samples found. Generate training data first.")
        sys.exit(1)

    # Step 2: Quantize model
    logger.info("=== Quantizing model ===")
    quantized_dir = quantize_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        bits=args.quantize_bits,
    )

    # Step 3: Train
    logger.info("=== Starting MLX-LM training ===")
    train(
        model_path=quantized_dir,
        data_dir=mlx_data_dir,
        num_train_samples=num_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    logger.info("=== Done ===")
    logger.info(f"To fuse adapter: python -m finetuning.train_mlx --fuse --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
