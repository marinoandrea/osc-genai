"""Tests for the fake-human loop player (pure parts + a fast bounded run)."""

from __future__ import annotations

from midisommar.core.note import Note
from midisommar.realtime.clock import WallClock
from midisommar.realtime.fake_human import _ACID, loop_length_beats, play_loop


class _FakeOut:
    def __init__(self):
        self.messages = []

    def send(self, msg):
        self.messages.append(msg)


def test_loop_length_rounds_up_to_beats():
    assert loop_length_beats([Note(60, 0.0, 0.25, 100)]) == 1.0
    assert loop_length_beats([Note(60, 0.0, 1.0, 100), Note(62, 3.5, 0.25, 100)]) == 4.0


def test_builtin_acid_pattern_is_valid():
    assert len(_ACID) == 16
    assert all(0 <= n.pitch <= 127 for n in _ACID)


def test_play_loop_sends_notes_and_cleans_up():
    out = _FakeOut()
    # Very fast tempo + bar of 1 beat, so the anchored loop fires within the bounded run.
    # The time budget is generous (not 0.05s): the run is real-time, and a loaded CI runner
    # can oversleep enough to miss a tight deadline before any note is emitted.
    play_loop(out, [Note(60, 0.0, 0.1, 100)], WallClock(6000), quantum=1, seconds=0.5)
    types = [m.type for m in out.messages]
    assert "note_on" in types
    # every note that started is also ended (offs flushed live or in cleanup)
    assert types.count("note_off") >= types.count("note_on")
