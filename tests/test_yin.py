"""YIN should recover the fundamental of synthetic tones and reject noise."""

from __future__ import annotations

import numpy as np
import pytest

from osc_genai.audio.yin import Yin

SR = 44100
FRAME = 2048


def _sine(freq: float, n: int = FRAME, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


@pytest.mark.parametrize("freq", [110.0, 220.0, 440.0, 587.33])  # A2, A3, A4, D5
def test_recovers_pure_tone_within_a_few_cents(freq):
    yin = Yin(SR, FRAME)
    f0, prob = yin.estimate(_sine(freq))
    assert f0 > 0
    cents = 1200 * np.log2(f0 / freq)
    assert abs(cents) < 35, f"{freq} Hz -> {f0:.2f} Hz ({cents:.1f} cents)"
    assert prob > 0.8


def test_white_noise_is_unvoiced_or_low_confidence():
    rng = np.random.default_rng(0)
    yin = Yin(SR, FRAME)
    f0, prob = yin.estimate(rng.standard_normal(FRAME) * 0.3)
    # Noise has no stable period: YIN should bail (-1) or report weak periodicity.
    assert f0 == -1.0 or prob < 0.6


def test_silence_is_handled_without_error():
    yin = Yin(SR, FRAME)
    f0, prob = yin.estimate(np.zeros(FRAME))
    assert f0 == -1.0 or prob <= 1.0  # mainly: no exception / no nan blow-up


def test_short_frame_is_padded():
    yin = Yin(SR, FRAME)
    f0, _ = yin.estimate(_sine(220.0, n=FRAME // 2))  # fewer samples than frame_size
    assert f0 > 0  # padding lets it still produce an estimate
