from typing import Counter

import numpy as np

from manual_analyser.transcription.types import TranscriptSegment

# Hook phrase n-gram size
HOOK_NGRAM = 3


# Minimum repetitions for a phrase to be considered a hook
MIN_HOOK_REPETITIONS = 2


def _extract_hook(
    segments: list[TranscriptSegment],
    ngram_size: int = HOOK_NGRAM,
) -> tuple[str | None, int, float | None]:
    """
    Find the most repeated phrase across all transcript segments.

    Uses n-gram counting across all segment text. The most repeated
    n-gram that appears at least MIN_HOOK_REPETITIONS times is the hook.

    Args:
        segments: Transcript segments with timestamps.
        ngram_size: Size of n-gram (default: 3 words).

    Returns:
        (hook_phrase, repetition_count, first_appearance_time)
        All None/0 if no hook found.
    """
    if not segments:
        return None, 0, None

    # Collect all n-grams with their first timestamp
    ngram_times: dict[str, float] = {}
    ngram_counts: Counter = Counter()

    for seg in segments:
        words = seg.text.lower().split()
        if len(words) < ngram_size:
            continue

        for i in range(len(words) - ngram_size + 1):
            ngram = " ".join(words[i : i + ngram_size])
            # Record first appearance time
            if ngram not in ngram_times:
                ngram_times[ngram] = seg.start
            ngram_counts[ngram] += 1

    if not ngram_counts:
        return None, 0, None

    # Find most common phrase meeting minimum threshold
    most_common = ngram_counts.most_common(1)[0]
    phrase, count = most_common

    if count < MIN_HOOK_REPETITIONS:
        return None, 0, None

    first_time = ngram_times.get(phrase)
    return phrase, count, first_time


# ---------------------------------------------------------------------------
# Lyric statistics
# ---------------------------------------------------------------------------


def _compute_unique_word_ratio(text: str) -> float:
    """
    Compute the ratio of unique words to total words.

    Low ratio = high repetition = positive signal for The Manual criteria.

    Args:
        text: Full transcript text.

    Returns:
        Ratio 0.0–1.0. Returns 0.5 if text is empty (neutral).
    """
    if not text.strip():
        return 0.5

    words = text.lower().split()
    if not words:
        return 0.5

    unique_count = len(set(words))
    total_count = len(words)
    return float(np.clip(unique_count / total_count, 0.0, 1.0))
