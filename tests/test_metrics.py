"""Tests for the generation-quality metrics."""

from __future__ import annotations

import pytest

from osc_genai.training.metrics import (
    distinct_n,
    histogram_overlap,
    longest_copied_run,
    ngram_novelty,
)


def test_novelty_is_zero_when_identical():
    seqs = [[60, 62, 64, 65, 67]]
    assert ngram_novelty(seqs, seqs, n=3) == 0.0


def test_novelty_is_one_when_disjoint():
    assert ngram_novelty([[1, 2, 3, 4]], [[60, 61, 62, 63]], n=3) == 1.0


def test_longest_copied_run():
    train = [[60, 62, 64, 65, 67]]
    assert longest_copied_run([10, 62, 64, 65, 11], train) == 3  # 62,64,65
    assert longest_copied_run([1, 2, 3], train) == 0


def test_distinct_n():
    assert distinct_n([[1, 1, 1, 1]], n=2) == pytest.approx(
        1 / 3
    )  # (1,1) x3 -> 1 unique
    assert distinct_n([[1, 2, 3]], n=2) == 1.0  # (1,2),(2,3) all unique


def test_histogram_overlap():
    assert histogram_overlap([0, 1, 2], [0, 1, 2], 0, 2) == pytest.approx(1.0)
    assert histogram_overlap([0, 0], [2, 2], 0, 2) == 0.0
    assert histogram_overlap([0, 0, 1, 1], [0, 1, 1, 1], 0, 2) == pytest.approx(0.75)
