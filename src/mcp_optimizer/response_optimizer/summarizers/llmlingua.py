"""LLMLingua-2 summarizer using ONNX Runtime for inference."""

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import structlog
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mcp_optimizer.config import get_config
from mcp_optimizer.response_optimizer.summarizers.base import BaseSummarizer

logger = structlog.get_logger(__name__)

# Model folder name within the configured model path
LLMLINGUA_MODEL_FOLDER = "llmlingua2-onnx"


class LLMLinguaSummarizer(BaseSummarizer):
    """
    LLMLingua-2 summarizer using ONNX Runtime.

    Uses token classification to determine which tokens to preserve.
    The model outputs probabilities for each token being important,
    and we keep tokens above a threshold based on the target compression rate.

    Algorithm:
    1. Tokenize input text
    2. Run ONNX inference to get logits
    3. Apply softmax to get "keep" probabilities
    4. Calculate threshold based on target compression
    5. Keep tokens above threshold
    6. Reconstruct text from kept tokens
    """

    def __init__(
        self,
        force_tokens: list[str] | None = None,
    ):
        """
        Initialize the LLMLingua-2 summarizer.

        Args:
            force_tokens: Tokens to always preserve (e.g., ["\n", ".", "?", "!"])
        """
        config = get_config()
        if config.llmlingua_model_path:
            self.model_path = Path(config.llmlingua_model_path) / LLMLINGUA_MODEL_FOLDER
        else:
            # Default to models directory relative to this file
            self.model_path = Path(__file__).parent.parent / "models" / LLMLINGUA_MODEL_FOLDER
        self.force_tokens = force_tokens or ["\n", ".", "?", "!", ","]

        self._session: "ort.InferenceSession | None" = None
        self._tokenizer: "PreTrainedTokenizerBase | None" = None
        self._loaded = False
        self._available = False

    def _load_model(self) -> bool:
        """Load the ONNX model and tokenizer."""
        if self._loaded:
            return self._available

        try:
            model_file = self.model_path / "model.onnx"
            if not model_file.exists():
                logger.warning(
                    "LLMLingua ONNX model not found",
                    model_path=str(model_file),
                )
                self._loaded = True
                self._available = False
                return False

            # Load ONNX model
            self._session = ort.InferenceSession(
                str(model_file),
                providers=["CPUExecutionProvider"],
            )

            # Load tokenizer
            # Try to load from local path first, fall back to HuggingFace
            tokenizer_path = self.model_path
            if (tokenizer_path / "tokenizer_config.json").exists():
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                # Fall back to HuggingFace
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
                )

            self._loaded = True
            self._available = True
            logger.info("LLMLingua model loaded successfully", model_path=str(self.model_path))
            return True

        except Exception as e:
            logger.error("Failed to load LLMLingua model", error=str(e))
            self._loaded = True
            self._available = False
            return False

    def is_available(self) -> bool:
        """Check if the summarizer is available."""
        self._load_model()
        return self._available

    def _run_inference(self, inputs: Any) -> Any:
        """Run ONNX model inference and return logits."""
        if self._session is None:
            raise RuntimeError("ONNX session not initialized")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add token_type_ids if the model expects it
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            ort_inputs["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(input_ids))

        outputs = self._session.run(None, ort_inputs)
        return outputs[0]  # Shape: (batch, seq_len, 2)

    def _compute_keep_probabilities(self, logits: Any) -> Any:
        """Compute keep probabilities from logits using softmax."""
        # Use amax with explicit typing to work around numpy typing limitations
        logits_max: Any = np.amax(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return probs[0, :, 1]  # Batch index 0, class 1

    def _filter_tokens(
        self, tokens: list[Any], keep_probs: Any, attention_mask: Any, threshold: float
    ) -> list[str]:
        """Filter tokens based on keep probabilities and threshold."""
        kept_tokens = []
        for token, prob, mask in zip(tokens, keep_probs, attention_mask, strict=False):
            if mask == 0:
                continue  # Skip padding

            # Always keep special tokens and force tokens
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if self._should_force_keep(token):
                kept_tokens.append(token)
            elif prob >= threshold:
                kept_tokens.append(token)
        return kept_tokens

    async def summarize(self, text: str, target_tokens: int) -> str:
        """
        Summarize text using LLMLingua-2 token classification.

        Args:
            text: The text to summarize
            target_tokens: Target maximum token count

        Returns:
            Compressed text with important tokens preserved
        """
        if not self._load_model():
            return self._fallback_summarize(text, target_tokens)

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized after successful model load")
        if self._session is None:
            raise RuntimeError("ONNX session not initialized after successful model load")

        try:
            inputs = self._tokenizer(
                text, return_tensors="np", truncation=True, max_length=512, padding=True
            )

            logits = self._run_inference(inputs)
            keep_probs = self._compute_keep_probabilities(logits)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0])

            # Calculate threshold based on compression rate
            original_tokens = len([t for t, m in zip(tokens, attention_mask[0], strict=False) if m])
            target_keep = min(target_tokens, original_tokens)
            reduce_rate = 1.0 - (target_keep / original_tokens)

            valid_probs = keep_probs[attention_mask[0] == 1]
            if len(valid_probs) == 0:
                return text

            threshold = np.percentile(valid_probs, int(100 * reduce_rate))
            kept_tokens = self._filter_tokens(tokens, keep_probs, attention_mask[0], threshold)

            return self._reconstruct_text(kept_tokens)

        except Exception as e:
            logger.error("LLMLingua summarization failed", error=str(e))
            return self._fallback_summarize(text, target_tokens)

    def _should_force_keep(self, token: str) -> bool:
        """Check if a token should always be kept."""
        # Clean token (remove ## prefix from wordpiece)
        clean_token = token.replace("##", "")

        for force_token in self.force_tokens:
            if force_token in clean_token:
                return True

        # Keep tokens with digits
        if any(c.isdigit() for c in clean_token):
            return True

        return False

    def _reconstruct_text(self, tokens: list[str]) -> str:
        """Reconstruct text from kept tokens."""
        result = []
        for token in tokens:
            if token.startswith("##"):
                # Wordpiece continuation - append without space
                if result:
                    result[-1] += token[2:]
                else:
                    result.append(token[2:])
            else:
                result.append(token)

        return " ".join(result)

    def _fallback_summarize(self, text: str, target_tokens: int) -> str:
        """Simple fallback when model is not available."""
        # Rough estimate: 4 chars per token
        max_chars = target_tokens * 4

        if len(text) <= max_chars:
            return text

        # Keep first portion with truncation marker
        return text[: max_chars - 20] + " [...TRUNCATED]"
