"""Optional local multilingual prompt-injection classifier.

Llama Prompt Guard 2 complements the deterministic CP2 rules; it does not
replace topic filtering, output redaction, rate limits, or egress controls.
The model is loaded lazily so unit tests and the baseline lab path do not need
network access or GPU memory.  If its optional dependencies or gated weights
are unavailable, callers continue using deterministic guardrails.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any, Literal


PROMPT_GUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
_CHUNK_TOKENS = 510  # Model card specifies a 512-token input window.

PromptGuardStatus = Literal["MALICIOUS", "BENIGN", "UNAVAILABLE"]


@dataclass(frozen=True)
class PromptGuardVerdict:
    """Classifier result without leaking raw input into logs."""

    status: PromptGuardStatus
    score: float | None = None


def prompt_guard_enabled() -> bool:
    """Allow an operator to disable the optional semantic detector."""
    value = os.environ.get("PROMPT_GUARD_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def _load_classifier() -> tuple[Any, Any]:
    """Load the tokenizer and classifier once, only when semantic detection runs.

    Meta distributes the weights behind an access agreement.  ``HF_TOKEN`` is
    read only by the Hugging Face library and must remain in the environment,
    never in source or audit artifacts.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(PROMPT_GUARD_MODEL, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        PROMPT_GUARD_MODEL, token=token
    )
    model.eval()
    return tokenizer, model


def _chunk_text(text: str, tokenizer: Any) -> list[str]:
    """Split long input on tokenizer boundaries so no content is silently lost."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return [text]
    return [
        tokenizer.decode(token_ids[start : start + _CHUNK_TOKENS])
        for start in range(0, len(token_ids), _CHUNK_TOKENS)
    ]


def classify_prompt(text: str) -> PromptGuardVerdict:
    """Classify all input chunks, returning unavailable rather than failing open.

    The caller deliberately falls back to the deterministic CP2 rules when
    this optional local model cannot be used.  Operational logs can separately
    monitor ``UNAVAILABLE`` without storing the prompt text.
    """
    if not prompt_guard_enabled() or not text.strip():
        return PromptGuardVerdict("UNAVAILABLE")

    try:
        import torch

        tokenizer, model = _load_classifier()
        chunks = _chunk_text(text, tokenizer)
        best_score: float | None = None
        with torch.no_grad():
            for chunk in chunks:
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                logits = model(**inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1)
                class_id = int(probabilities.argmax().item())
                label = str(model.config.id2label[class_id]).upper()
                score = float(probabilities[class_id].item())
                best_score = max(best_score or 0.0, score)
                if label == "MALICIOUS":
                    return PromptGuardVerdict("MALICIOUS", score)
        return PromptGuardVerdict("BENIGN", best_score)
    except (ImportError, OSError, ValueError, RuntimeError):
        # Missing optional runtime, gated model access, or local inference error.
        # Do not make a banking request fail solely because this extra layer is down.
        return PromptGuardVerdict("UNAVAILABLE")
