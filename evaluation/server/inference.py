"""Model loading and inference using transformers.

Loads the merged fine-tuned model and generates responses using
the training chat template (NOT the saved tokenizer template).
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.server.chat_template import format_messages, parse_assistant_response

logger = logging.getLogger(__name__)


class PipeSQLModel:
    """Wraps a fine-tuned model for pipe SQL generation."""

    def __init__(self, model_path: str, device: str | None = None):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = device

    def load(self):
        """Load model and tokenizer."""
        logger.info(f"Loading tokenizer from {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Loading model from {self.model_path}")
        device = self._device

        model_kwargs = {"trust_remote_code": True}

        if device == "cuda" or (device is None and torch.cuda.is_available()):
            model_kwargs["dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
            self._device = "cuda"
        elif device == "mps" or (
            device is None and torch.backends.mps.is_available()
        ):
            model_kwargs["dtype"] = torch.float16
            model_kwargs["device_map"] = {"": "mps"}
            self._device = "mps"
        else:
            model_kwargs["dtype"] = torch.float32
            self._device = "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **model_kwargs
        )
        self._model.eval()
        logger.info(f"Model loaded on {self._device}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict:
        """Generate a response for the given message history.

        Uses the training chat template (format_messages) instead of
        the tokenizer's saved template.

        Returns:
            OpenAI-compatible response dict with content and optional tool_calls.
        """
        # Build prompt using training template
        prompt = format_messages(messages, add_generation_prompt=True)

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs["attention_mask"].to(self._model.device)
        input_length = input_ids.shape[1]

        # im_end token for stopping
        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [self._tokenizer.eos_token_id]
        if isinstance(im_end_id, int) and im_end_id != self._tokenizer.unk_token_id:
            eos_ids.append(im_end_id)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),  # avoid 0
                do_sample=temperature > 0.01,
                eos_token_id=eos_ids,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][input_length:]
        raw_text = self._tokenizer.decode(new_tokens, skip_special_tokens=False)

        logger.info(f"Raw token IDs: {new_tokens.tolist()[:50]}")
        logger.info(f"Raw decoded text: {repr(raw_text[:500])}")

        # Parse into content + tool_calls
        parsed = parse_assistant_response(raw_text)
        logger.info(f"Parsed: content={repr(parsed.get('content', '')[:100])}, tool_calls={parsed.get('tool_calls') is not None}")

        return parsed
