"""Evaluation metrics: overfitting (held-out loss) and creative-generation proxies.

Generation quality is summarised on pitch sequences along four axes:

* **novelty**    - fraction of generated n-grams unseen in training (high = not copying)
* **copy length**- longest verbatim run shared with any training clip (low = not memorising)
* **diversity**  - distinct-n across generated samples (high = not mode-collapsed)
* **style match**- histogram overlap of pitch-classes / intervals vs training (high = in-idiom)

The creative "sweet spot" is high novelty + diversity *and* high style match (inventing new lines in
the right idiom) with short copy lengths. Pure copying => low novelty / long copies; pure noise =>
high novelty but low style match.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import torch

from osc_genai.core.vocab import EventCodec
from osc_genai.training.train import collate


def dataset_loss(
    model,
    event_sequences,
    codec: EventCodec | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """Mean per-batch loss over a dataset (no grad) — compare train vs held-out to see overfitting."""
    codec = codec or EventCodec(model.vocab)
    model.eval()
    encoded = [codec.encode_sequence(s, add_eos=True) for s in event_sequences if s]
    total, batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            targets, mask = collate(encoded[start : start + batch_size], codec.eos)
            total += model.loss(targets.to(device), mask.to(device)).item()
            batches += 1
    return total / max(1, batches)


def _ngrams(seq: Sequence[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def ngram_novelty(
    generated: list[list[int]], train: list[list[int]], n: int = 4
) -> float:
    """Fraction of generated n-grams not present anywhere in training (1 = fully novel)."""
    train_ngrams: set[tuple[int, ...]] = set()
    for seq in train:
        train_ngrams.update(_ngrams(seq, n))
    total = novel = 0
    for seq in generated:
        for gram in _ngrams(seq, n):
            total += 1
            novel += gram not in train_ngrams
    return novel / total if total else 0.0


def longest_copied_run(seq: Sequence[int], train: list[list[int]]) -> int:
    """Length of the longest contiguous run in ``seq`` that appears verbatim in any training clip."""
    best = 0
    for ref in train:
        prev = [0] * (len(ref) + 1)
        for a in seq:
            cur = [0] * (len(ref) + 1)
            for j, b in enumerate(ref, 1):
                if a == b:
                    cur[j] = prev[j - 1] + 1
                    best = max(best, cur[j])
            prev = cur
    return best


def distinct_n(sequences: list[list[int]], n: int = 4) -> float:
    """Distinct-n: unique n-grams / total n-grams across all sequences (1 = no repetition)."""
    grams = [gram for seq in sequences for gram in _ngrams(seq, n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def histogram_overlap(a: Iterable[int], b: Iterable[int], low: int, high: int) -> float:
    """Overlap of two normalised integer histograms over ``[low, high]`` (1 = identical, 0 = disjoint)."""
    ca, cb = Counter(a), Counter(b)
    sa, sb = sum(ca.values()) or 1, sum(cb.values()) or 1
    return sum(min(ca.get(k, 0) / sa, cb.get(k, 0) / sb) for k in range(low, high + 1))
