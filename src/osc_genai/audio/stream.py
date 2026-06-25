"""Glue: buffer incoming audio blocks into overlapping YIN frames and drive a segmenter.

Audio arrives in fixed blocks (whatever the capture device hands us); YIN wants analysis frames of
``frame_size`` advanced by ``hop``. :class:`PitchTracker` owns that rolling buffer so both the live
duet input and the standalone ``audio-track`` tool share one code path.
"""

from __future__ import annotations

import numpy as np

from osc_genai.audio.segment import NoteSegmenter
from osc_genai.audio.yin import Yin


class PitchTracker:
    """Accumulate audio blocks; for each full ``frame_size`` window run YIN and feed the segmenter."""

    def __init__(
        self, yin: Yin, segmenter: NoteSegmenter, *, frame_size: int, hop: int
    ) -> None:
        self._yin = yin
        self._seg = segmenter
        self._frame_size = int(frame_size)
        self._hop = int(hop)
        self._buf = np.zeros(0, dtype=np.float64)

    def feed(self, block: np.ndarray) -> None:
        """Append a mono block (float samples) and process every frame it completes."""
        self._buf = np.concatenate(
            [self._buf, np.asarray(block, dtype=np.float64).ravel()]
        )
        while self._buf.shape[0] >= self._frame_size:
            frame = self._buf[: self._frame_size]
            rms = float(np.sqrt(np.mean(frame * frame)))
            f0, prob = self._yin.estimate(frame)
            self._seg.process(f0, prob, rms)
            self._buf = self._buf[self._hop :]
