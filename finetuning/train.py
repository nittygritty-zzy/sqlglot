"""Fine-tune Qwen-2.5-Coder-7B on pipe SQL training data.

Usage:
    python -m finetuning.train [--flags]
    python -m finetuning.train --merge --output-dir finetuning_output/final --merged-dir finetuning_output/merged
"""

from __future__ import annotations

import os
import logging

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from finetuning.config import TrainConfig
from finetuning.data import load_chat_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Qwen chat template with {% generation %} markers for assistant_only_loss.
# Handles system/user/assistant/tool roles for tool-calling training data.
# Assistant messages may have optional tool_calls; tool responses use the user role
# with <tool_response> tags (no loss computed on tool responses).
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'user' %}"
    "{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{% generation %}{{- '<|im_start|>assistant\\n' + message.content }}"
    "{%- if message.tool_calls %}"
    "{%- for tc in message.tool_calls %}"
    "{{- '\\n<tool_call>\\n' + tc.function.name + '(' + tc.function.arguments + ')\\n</tool_call>' }}"
    "{%- endfor %}"
    "{%- endif %}"
    "{{- '<|im_end|>\\n' }}{% endgeneration %}"
    "{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{% generation %}{{- '<|im_start|>assistant\\n' }}{% endgeneration %}"
    "{%- endif %}"
)


def detect_device() -> tuple[str, bool, bool]:
    """Detect available hardware. Returns (device, use_mps, use_cuda)."""
    if torch.cuda.is_available():
        return "cuda", False, True
    if torch.backends.mps.is_available():
        return "mps", True, False
    return "cpu", False, False


def load_model_and_tokenizer(config: TrainConfig):
    """Load the base model and tokenizer with appropriate quantization settings."""
    device, use_mps, use_cuda = detect_device()
    logger.info(f"Detected device: {device} (mps={use_mps}, cuda={use_cuda})")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save original template (with tool call support) for restoring at save time
    tokenizer._original_chat_template = tokenizer.chat_template
    # Use simplified template with {% generation %} markers for assistant_only_loss during training
    tokenizer.chat_template = CHAT_TEMPLATE
    logger.info("Set chat template with {% generation %} markers for assistant-only loss")

    model_kwargs = {
        "trust_remote_code": True,
    }

    if use_cuda and config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
        logger.info("Using QLoRA with 4-bit quantization (CUDA)")
    elif use_cuda:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        logger.info("Using float16 LoRA on CUDA (no quantization)")
    elif use_mps:
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = {"": device}
        logger.info("Using float16 LoRA on MPS (no quantization)")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    else:
        model_kwargs["torch_dtype"] = torch.float32
        logger.info("Using float32 on CPU")

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.config.use_cache = False

    return model, tokenizer, device


def make_lora_config(config: TrainConfig) -> LoraConfig:
    """Build LoRA config from training config."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def merge_adapter(config: TrainConfig):
    """Merge LoRA adapter into base model for standalone inference."""
    adapter_dir = os.path.join(config.output_dir, "final")
    merged_dir = config.merged_dir or os.path.join(config.output_dir, "merged")

    logger.info(f"Loading base model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu",
    )

    logger.info(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {merged_dir}")
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merge complete.")


def train(config: TrainConfig):
    """Run the fine-tuning loop."""
    device, use_mps, use_cuda = detect_device()

    # Load model + tokenizer
    model, tokenizer, device = load_model_and_tokenizer(config)

    # Load datasets
    logger.info(f"Loading training data: {config.train_data}")
    train_dataset = load_chat_dataset(config.train_data, limit=config.limit)
    logger.info(f"Training samples: {len(train_dataset)}")

    dev_dataset = None
    if config.dev_data and os.path.exists(config.dev_data):
        dev_dataset = load_chat_dataset(config.dev_data, limit=config.limit)
        logger.info(f"Dev samples: {len(dev_dataset)}")

    # Training config — TRL 0.29 uses SFTConfig which combines TrainingArguments + SFT params
    # QLoRA with bitsandbytes produces BFloat16 params, which are incompatible with FP16 AMP scaler.
    # Use bf16 when doing 4-bit quantization on CUDA, fp16 otherwise.
    use_bf16 = use_cuda and config.load_in_4bit
    use_fp16 = (use_cuda or use_mps) and not use_bf16
    has_eval = dev_dataset is not None

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=config.eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        seed=config.seed,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=not use_mps,
        # SFT-specific
        max_length=config.max_seq_length,
        assistant_only_loss=True,
    )

    # Create trainer — pass peft_config so SFTTrainer handles LoRA setup
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        peft_config=make_lora_config(config),
    )

    # Train
    logger.info("Starting training...")
    result = trainer.train()

    # Log results
    logger.info(f"Training complete. Metrics: {result.metrics}")

    # Restore original chat template (with tool call support) before saving
    if hasattr(tokenizer, "_original_chat_template"):
        tokenizer.chat_template = tokenizer._original_chat_template
        logger.info("Restored original chat template with tool call support")

    # Save final adapter + tokenizer
    final_dir = os.path.join(config.output_dir, "final")
    logger.info(f"Saving final adapter to: {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Eval on dev set
    if dev_dataset:
        eval_metrics = trainer.evaluate()
        logger.info(f"Final eval metrics: {eval_metrics}")

    logger.info("Done.")


def main():
    config = TrainConfig.from_cli()

    if config.merge:
        merge_adapter(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
