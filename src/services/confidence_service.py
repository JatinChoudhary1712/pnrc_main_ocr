# src/services/confidence_service.py
import math

from src.config import (
    LOW_CONFIDENCE_FRACTION_THRESHOLD,
    LOW_CONFIDENCE_RUN_LENGTH,
    TOKEN_CONFIDENCE_THRESHOLD,
)


def score_page_confidence(
    token_logprobs,
    token_threshold=TOKEN_CONFIDENCE_THRESHOLD,
    fraction_threshold=LOW_CONFIDENCE_FRACTION_THRESHOLD,
    min_run_length=LOW_CONFIDENCE_RUN_LENGTH,
):
    """`token_logprobs` is a list of per-token logprob floats (or None/[] for empty pages)."""
    if not token_logprobs:
        return {
            "is_low_confidence": False,
            "has_low_confidence_run": False,
        }

    confidences = [math.exp(lp) for lp in token_logprobs]

    fraction_below_threshold = sum(1 for c in confidences if c < token_threshold) / len(confidences)

    run = 0
    has_low_confidence_run = False
    for c in confidences:
        run = run + 1 if c < token_threshold else 0
        if run >= min_run_length:
            has_low_confidence_run = True
            break

    return {
        "is_low_confidence": fraction_below_threshold > fraction_threshold or has_low_confidence_run,
        "has_low_confidence_run": has_low_confidence_run,
    }
