"""Factored per-event representation: clip ``Note`` lists <-> ordered event sequences.

The model consumes music as a sequence of *events*, one per note, each factored into independent
fields — pitch, onset delta, duration, velocity, **channel**, and **source** — rather than a flat
token stream. The channel field lets one stream address multiple instruments at once; the source
field tags which line of a duet an event belongs to (``PARTNER`` = the observed/other player,
``SELF`` = the line the model generates), so two time-aligned tracks can be interleaved into one
clock-shared stream (see :func:`pairs.interleave`).

Time is quantised to a grid of ``steps_per_beat`` (default 4 = sixteenth notes); ``dt`` is the gap
in steps from the previous note's onset, so absolute timing is recovered by accumulation. Encoding
on-grid notes and decoding them again is exact; off-grid input snaps. Muted notes are dropped on
encode. Simultaneous notes (``dt == 0``) — polyphony, including across channels — are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

from osc_genai.core.note import Note

DEFAULT_STEPS_PER_BEAT = 4

PARTNER = 0  # the observed/other line of a duet (conditioning input)
SELF = 1  # the line the model generates


@dataclass(frozen=True)
class Event:
    """One note as factored fields. ``dt`` and ``dur`` are in grid steps, not beats."""

    pitch: int  # 0-127
    dt: int  # steps since the previous event's onset (>= 0)
    dur: int  # note length in steps (>= 1)
    velocity: int  # 0-127
    channel: int = 0  # MIDI channel 0-15 — lets the model address multiple instruments
    source: int = 0  # duet role: PARTNER (observed) or SELF (generated)


def _quantize(beats: float, steps_per_beat: int) -> int:
    """Quantise a beat value to the nearest whole grid step."""
    return int(round(beats * steps_per_beat))


def notes_to_events(
    notes: list[Note], steps_per_beat: int = DEFAULT_STEPS_PER_BEAT
) -> list[Event]:
    """Convert clip notes to an onset-ordered event sequence (carrying each note's channel)."""
    ordered = sorted((n for n in notes if not n.mute), key=lambda n: (n.start, n.pitch))
    events: list[Event] = []
    prev_onset = 0
    for note in ordered:
        onset = _quantize(note.start, steps_per_beat)
        events.append(
            Event(
                pitch=note.pitch,
                dt=max(0, onset - prev_onset),
                dur=max(1, _quantize(note.duration, steps_per_beat)),
                velocity=note.velocity,
                channel=note.channel,
            )
        )
        prev_onset = onset
    return events


def events_to_notes(
    events: list[Event], steps_per_beat: int = DEFAULT_STEPS_PER_BEAT
) -> list[Note]:
    """Inverse of :func:`notes_to_events`: recover onset times by accumulating ``dt``."""
    notes: list[Note] = []
    onset = 0
    for event in events:
        onset += event.dt
        notes.append(
            Note(
                pitch=event.pitch,
                start=onset / steps_per_beat,
                duration=event.dur / steps_per_beat,
                velocity=event.velocity,
                channel=event.channel,
            )
        )
    return notes
