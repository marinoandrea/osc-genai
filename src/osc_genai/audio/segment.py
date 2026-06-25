"""Turn a stream of per-frame pitch estimates into discrete MIDI notes (monophonic).

YIN gives a ``(f0, probability)`` per frame; a player gives *notes*. :class:`NoteSegmenter` bridges
the two: it gates voicing (confidence + loudness), smooths out single-frame octave errors, and emits
``note_on(pitch, velocity)`` / ``note_off(pitch)`` callbacks — the exact shape ``HumanStream``
consumes, so the audio path reuses the duet's note bookkeeping verbatim.

It is **monophonic** by construction: at most one note sounds at a time, so a new stable pitch closes
the previous note before opening the next.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable

import numpy as np


def hz_to_midi(hz: float) -> float:
    """Frequency in Hz to a (fractional) MIDI note number. 440 Hz → 69 (A4)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


class NoteSegmenter:
    """Segment ``(f0, probability, rms)`` frames into note on/off events.

    Voicing requires ``probability >= confidence`` **and** ``rms >= noise_floor``. A note is only
    opened once ``smoothing`` consecutive voiced frames agree on a (median-smoothed) pitch — this
    both rejects octave/jitter blips and enforces a minimum stable duration. A note is closed when
    the pitch changes or after ``release_frames`` unvoiced frames. Velocity is mapped from the onset
    RMS between ``noise_floor`` and ``rms_ceiling``.
    """

    def __init__(
        self,
        *,
        note_on: Callable[[int, int], None],
        note_off: Callable[[int], None],
        confidence: float = 0.5,
        noise_floor: float = 0.01,
        smoothing: int = 3,
        release_frames: int = 3,
        rms_ceiling: float = 0.2,
        velocity_floor: int = 32,
    ) -> None:
        self.note_on = note_on
        self.note_off = note_off
        self.confidence = float(confidence)
        self.noise_floor = float(noise_floor)
        self.release_frames = int(release_frames)
        self.rms_ceiling = float(rms_ceiling)
        self.velocity_floor = int(velocity_floor)
        self._hist: deque[int] = deque(maxlen=max(1, int(smoothing)))
        self._current: int | None = None  # pitch currently sounding
        self._silence = 0

    def _velocity(self, rms: float) -> int:
        span = max(1e-9, self.rms_ceiling - self.noise_floor)
        frac = min(1.0, max(0.0, (rms - self.noise_floor) / span))
        return int(round(self.velocity_floor + frac * (127 - self.velocity_floor)))

    def _open(self, pitch: int, rms: float) -> None:
        self._current = pitch
        self.note_on(pitch, self._velocity(rms))

    def _close(self) -> None:
        if self._current is not None:
            self.note_off(self._current)
            self._current = None

    def process(self, f0_hz: float, probability: float, rms: float) -> None:
        """Feed one analysis frame; fires note_on/note_off callbacks as the line evolves."""
        voiced = f0_hz > 0.0 and probability >= self.confidence and rms >= self.noise_floor
        if not voiced:
            self._silence += 1
            self._hist.clear()
            if self._current is not None and self._silence >= self.release_frames:
                self._close()
            return

        self._silence = 0
        pitch = int(min(127, max(0, round(hz_to_midi(f0_hz)))))
        self._hist.append(pitch)
        if len(self._hist) < self._hist.maxlen:
            return  # not yet stable — wait for the smoothing window to fill
        smoothed = int(np.median(self._hist))
        if self._current is None:
            self._open(smoothed, rms)
        elif smoothed != self._current:
            self._close()
            self._open(smoothed, rms)

    def flush(self) -> None:
        """Close any note left sounding (call when the audio stream ends)."""
        self._close()
