"""The 'ML magic' seam.

For now ``generate_notes`` returns a hardcoded melody. Later this is where a model will
produce notes — keep the return type (a list of :class:`Note`) model-agnostic so the rest
of the pipeline never has to change.
"""

from __future__ import annotations

from typing import NamedTuple


class Note(NamedTuple):
    """A single MIDI note, in the shape AbletonOSC expects.

    ``start`` and ``duration`` are in beats; ``pitch`` and ``velocity`` are 0-127.
    """

    pitch: int
    start: float
    duration: float
    velocity: int = 100
    mute: bool = False


# C major scale, one note per beat. MIDI 60 == middle C.
_C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]


def generate_notes() -> list[Note]:
    """Return the notes to write into the clip (stubbed ML output)."""
    return [
        Note(pitch=pitch, start=float(beat), duration=1.0, velocity=100)
        for beat, pitch in enumerate(_C_MAJOR)
    ]


def total_beats(notes: list[Note]) -> float:
    """Length in beats needed to hold ``notes`` (used to size the clip)."""
    if not notes:
        return 0.0
    return max(note.start + note.duration for note in notes)
