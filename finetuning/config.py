from __future__ import annotations

import argparse
import typing as t
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Data
    train_data: str = "training_data_output/train.jsonl"
    dev_data: str = "training_data_output/dev.jsonl"
    max_seq_length: int = 2048
    limit: t.Optional[int] = None

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Quantization (CUDA only)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Eval & Saving
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10

    # Output
    output_dir: str = "finetuning_output"
    seed: int = 42

    # Merge mode
    merge: bool = False
    merged_dir: str = ""

    @classmethod
    def from_cli(cls, args: t.Optional[list[str]] = None) -> TrainConfig:
        parser = argparse.ArgumentParser(description="Fine-tune Qwen on pipe SQL data")
        defaults = cls()

        parser.add_argument("--model-name", default=defaults.model_name)
        parser.add_argument("--train-data", default=defaults.train_data)
        parser.add_argument("--dev-data", default=defaults.dev_data)
        parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
        parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples (for smoke tests)")

        parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
        parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
        parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)

        parser.add_argument("--load-in-4bit", action="store_true", default=defaults.load_in_4bit)
        parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")

        parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
        parser.add_argument("--per-device-train-batch-size", type=int, default=defaults.per_device_train_batch_size)
        parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.gradient_accumulation_steps)
        parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
        parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
        parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
        parser.add_argument("--lr-scheduler-type", default=defaults.lr_scheduler_type)

        parser.add_argument("--eval-steps", type=int, default=defaults.eval_steps)
        parser.add_argument("--save-steps", type=int, default=defaults.save_steps)
        parser.add_argument("--logging-steps", type=int, default=defaults.logging_steps)

        parser.add_argument("--output-dir", default=defaults.output_dir)
        parser.add_argument("--seed", type=int, default=defaults.seed)

        parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter into base model")
        parser.add_argument("--merged-dir", default="", help="Output directory for merged model")

        parsed = parser.parse_args(args)

        return cls(
            model_name=parsed.model_name,
            train_data=parsed.train_data,
            dev_data=parsed.dev_data,
            max_seq_length=parsed.max_seq_length,
            limit=parsed.limit,
            lora_r=parsed.lora_r,
            lora_alpha=parsed.lora_alpha,
            lora_dropout=parsed.lora_dropout,
            load_in_4bit=parsed.load_in_4bit,
            num_epochs=parsed.num_epochs,
            per_device_train_batch_size=parsed.per_device_train_batch_size,
            gradient_accumulation_steps=parsed.gradient_accumulation_steps,
            learning_rate=parsed.learning_rate,
            warmup_ratio=parsed.warmup_ratio,
            weight_decay=parsed.weight_decay,
            lr_scheduler_type=parsed.lr_scheduler_type,
            eval_steps=parsed.eval_steps,
            save_steps=parsed.save_steps,
            logging_steps=parsed.logging_steps,
            output_dir=parsed.output_dir,
            seed=parsed.seed,
            merge=parsed.merge,
            merged_dir=parsed.merged_dir,
        )
