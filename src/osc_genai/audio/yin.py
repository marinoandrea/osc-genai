"""YIN fundamental-frequency estimation, vectorized in NumPy.

A faithful port of the classic YIN algorithm (de Cheveigné & Kawahara 2002), following the C
reference at https://github.com/ashokfernandez/Yin-Pitch-Tracking. YIN tracks the pitch of a
**monophonic** signal — one fundamental per frame — so it cannot resolve chords.

The five steps (all vectorized):

1. **Difference function** ``d(τ) = Σ_j (x[j] − x[j+τ])²`` for ``τ < frame_size/2``.
2. **Cumulative mean normalized difference** ``d'`` (``d'[0] = 1``) — flattens the falloff so a fixed
   absolute threshold works across pitches.
3. **Absolute threshold** — the first ``τ`` where ``d'`` dips below ``threshold``, then descend to the
   local minimum. ``probability = 1 − d'[τ]`` (aperiodicity-based confidence).
4. **Parabolic interpolation** around ``τ`` for sub-sample period precision.
5. ``f0 = sample_rate / τ``.

The hot step-1 loop is isolated as :func:`_difference` (an FFT-based form) so a compiled backend
(Cython/CFFI/C) can replace just that function later without touching the rest.
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.15  # the reference's YIN_DEFAULT_THRESHOLD


def _difference(frame: np.ndarray, half: int) -> np.ndarray:
    """The YIN difference function ``d(τ)`` for ``τ`` in ``[0, half)``, via FFT autocorrelation.

    ``d(τ) = Σ_{j<half} (x[j] − x[j+τ])² = e₁ + e₂(τ) − 2·r(τ)`` where ``e₁`` is the energy of the
    first half, ``e₂(τ)`` the energy of the window starting at ``τ`` (sliding sum via cumsum), and
    ``r(τ) = Σ_j x[j]·x[j+τ]`` the cross-correlation (one FFT pair). This is the standard O(n log n)
    replacement for the reference's O(n²) double loop. The seam: swap this for a C kernel later.
    """
    x = np.asarray(frame, dtype=np.float64)[: 2 * half]
    a = x[:half]
    energy_first = float(np.dot(a, a))  # e₁ — constant in τ

    cumsq = np.concatenate(([0.0], np.cumsum(x * x)))
    energy_win = cumsq[half : 2 * half] - cumsq[:half]  # e₂(τ), length == half

    fft_size = 1
    while fft_size < 2 * half:
        fft_size <<= 1
    corr = np.fft.irfft(np.fft.rfft(x, fft_size) * np.conj(np.fft.rfft(a, fft_size)), fft_size)
    corr = corr[:half]  # r(τ) for τ in [0, half)

    diff = energy_first + energy_win - 2.0 * corr
    np.maximum(diff, 0.0, out=diff)  # FFT rounding can dip a hair below zero
    return diff


def _cumulative_mean_normalized_difference(diff: np.ndarray) -> np.ndarray:
    """Normalize ``d`` into ``d'``: ``d'[0]=1``, ``d'[τ] = d[τ]·τ / Σ_{j=1}^{τ} d[j]``."""
    cmnd = np.empty_like(diff)
    cmnd[0] = 1.0
    taus = np.arange(1, diff.shape[0], dtype=np.float64)
    running = np.cumsum(diff[1:])
    safe = np.where(running == 0.0, np.finfo(np.float64).tiny, running)
    cmnd[1:] = diff[1:] * taus / safe
    return cmnd


def _absolute_threshold(cmnd: np.ndarray, threshold: float) -> int:
    """First ``τ ≥ 2`` whose ``d'`` dips below ``threshold``, walked down to the local min; else -1."""
    below = np.nonzero(cmnd[2:] < threshold)[0]
    if below.size == 0:
        return -1
    tau = int(below[0]) + 2
    n = cmnd.shape[0]
    while tau + 1 < n and cmnd[tau + 1] < cmnd[tau]:
        tau += 1
    return tau


def _parabolic_interpolation(cmnd: np.ndarray, tau: int) -> float:
    """Refine ``tau`` by fitting a parabola to ``(tau-1, tau, tau+1)`` (reference's betterTau)."""
    n = cmnd.shape[0]
    x0 = tau - 1 if tau > 0 else tau
    x2 = tau + 1 if tau + 1 < n else tau
    if x0 == tau:
        return float(tau if cmnd[tau] <= cmnd[x2] else x2)
    if x2 == tau:
        return float(tau if cmnd[tau] <= cmnd[x0] else x0)
    s0, s1, s2 = float(cmnd[x0]), float(cmnd[tau]), float(cmnd[x2])
    denom = 2.0 * (2.0 * s1 - s2 - s0)
    if denom == 0.0:
        return float(tau)
    return tau + (s2 - s0) / denom


class Yin:
    """Estimate the fundamental of a monophonic frame.

    ``frame_size`` is the analysis window in samples; pitch resolution and the lowest detectable
    frequency scale with it (``f_min ≈ 2·sample_rate / frame_size``). Higher ``threshold`` accepts
    weaker periodicity (more sensitive, more octave errors); the reference default is 0.15.
    """

    def __init__(self, sample_rate: int, frame_size: int, threshold: float = DEFAULT_THRESHOLD) -> None:
        if frame_size < 4:
            raise ValueError("frame_size must be >= 4")
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.half = self.frame_size // 2
        self.threshold = float(threshold)

    def estimate(self, frame: np.ndarray) -> tuple[float, float]:
        """Return ``(f0_hz, probability)``; ``f0_hz == -1.0`` when no pitch is found.

        ``probability`` (0–1) is ``1 − d'[τ]`` at the chosen lag: high when the frame is strongly
        periodic. Note YIN says nothing about loudness — gate silence on RMS upstream.
        """
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape[0] < self.frame_size:
            frame = np.pad(frame, (0, self.frame_size - frame.shape[0]))
        diff = _difference(frame, self.half)
        cmnd = _cumulative_mean_normalized_difference(diff)
        tau = _absolute_threshold(cmnd, self.threshold)
        if tau == -1:
            return -1.0, 0.0
        better_tau = _parabolic_interpolation(cmnd, tau)
        if better_tau <= 0.0:
            return -1.0, 0.0
        return self.sample_rate / better_tau, float(1.0 - cmnd[tau])
