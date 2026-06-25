"""Tests for the factored per-event representation (Note <-> Event round-trips)."""

from __future__ import annotations

from osc_genai.core.event import events_to_notes, notes_to_events
from osc_genai.core.note import Note


def test_roundtrip_on_grid_melody_is_exact():
    notes = [Note(60, 0.0, 1.0, 100), Note(62, 1.0, 1.0, 100), Note(64, 2.0, 0.5, 80)]
    assert events_to_notes(notes_to_events(notes)) == notes


def test_first_event_preserves_absolute_start():
    events = notes_to_events([Note(67, 2.0, 1.0, 90)])
    assert events[0].dt == 8  # 2 beats * 4 steps/beat
    assert events_to_notes(events)[0].start == 2.0


def test_simultaneous_notes_get_zero_dt_and_sort_by_pitch():
    events = notes_to_events([Note(64, 0.0, 1.0, 100), Note(60, 0.0, 1.0, 100)])
    assert [e.pitch for e in events] == [60, 64]
    assert [e.dt for e in events] == [0, 0]


def test_deep_polyphony_roundtrips_no_cap():
    chord = [Note(40 + i, 0.0, 0.5, 100) for i in range(16)]  # a 16-note cluster
    events = notes_to_events(chord)
    assert sum(1 for e in events if e.dt == 0) == 16  # all simultaneous, no cap
    rt = events_to_notes(events)
    assert len(rt) == 16 and {n.start for n in rt} == {0.0}  # still one 16-note chord


def test_muted_notes_are_dropped():
    notes = [Note(60, 0.0, 1.0, 100), Note(62, 1.0, 1.0, 100, mute=True)]
    assert [e.pitch for e in notes_to_events(notes)] == [60]


def test_duration_quantises_to_at_least_one_step():
    # 0.05 beat * 4 = 0.2 -> rounds to 0 -> clamped to 1 so the note survives.
    assert notes_to_events([Note(60, 0.0, 0.05, 100)])[0].dur == 1


def test_off_grid_start_snaps_to_nearest_step():
    # 0.1 beat * 4 = 0.4 -> rounds to step 0.
    assert events_to_notes(notes_to_events([Note(60, 0.1, 1.0, 100)]))[0].start == 0.0


def test_empty_sequences():
    assert notes_to_events([]) == []
    assert events_to_notes([]) == []


def test_steps_per_beat_is_configurable():
    notes = [Note(60, 0.5, 0.25, 100)]
    events = notes_to_events(notes, steps_per_beat=4)
    assert events[0].dt == 2 and events[0].dur == 1
    assert events_to_notes(events, steps_per_beat=4) == notes
