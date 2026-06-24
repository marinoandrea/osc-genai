"""Core music representation: the ``Note``/``Event`` types and the model-field codec.

Pure data — no torch, no IO. ``from osc_genai.core import Note, Event, EventCodec``.
"""

from osc_genai.core.note import Note
from osc_genai.core.event import (
    DEFAULT_STEPS_PER_BEAT,
    PARTNER,
    SELF,
    Event,
    events_to_notes,
    notes_to_events,
)
from osc_genai.core.vocab import EventCodec, Fields, MIDI_RANGE, VocabConfig

__all__ = [
    "Note",
    "Event",
    "notes_to_events",
    "events_to_notes",
    "DEFAULT_STEPS_PER_BEAT",
    "PARTNER",
    "SELF",
    "EventCodec",
    "VocabConfig",
    "Fields",
    "MIDI_RANGE",
]
