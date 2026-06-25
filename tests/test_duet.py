"""Tests for the duet engine's pure parts (human-stream capture on the shared clock)."""

from __future__ import annotations

import mido

from osc_genai.realtime.duet import HumanStream


class FakeClock:
    """A hand-advanced beat clock, so capture tests don't depend on wall time."""

    def __init__(self) -> None:
        self.beat = 0.0


def test_human_stream_records_noteons_as_partner_notes():
    clk = FakeClock()
    s = HumanStream(lambda: clk.beat, channel=0)
    for i, pitch in enumerate((60, 64, 67)):
        clk.beat = i * 0.5
        s.on_message(mido.Message("note_on", note=pitch, velocity=100))
    notes, count = s.window(0.0)
    assert [n.pitch for n in notes] == [60, 64, 67]
    assert [n.start for n in notes] == [
        0.0,
        0.5,
        1.0,
    ]  # captured on the shared beat clock
    assert all(n.channel == 0 and n.velocity == 100 for n in notes)
    assert count == 3


def test_human_stream_ignores_offs_for_the_count():
    clk = FakeClock()
    s = HumanStream(lambda: clk.beat, channel=0)
    s.on_message(mido.Message("note_on", note=60, velocity=0))  # zero-velocity = off
    s.on_message(mido.Message("note_off", note=60, velocity=64))
    s.on_message(mido.Message("control_change", control=1, value=10))
    notes, count = s.window(0.0)
    assert notes == [] and count == 0


def test_note_off_finalises_duration():
    clk = FakeClock()
    s = HumanStream(lambda: clk.beat, channel=0)
    s.on_message(mido.Message("note_on", note=60, velocity=100))
    clk.beat = 1.5
    s.on_message(mido.Message("note_off", note=60, velocity=0))
    notes, _ = s.window(0.0)
    assert len(notes) == 1 and notes[0].duration == 1.5


def test_window_trims_to_trailing_notes():
    clk = FakeClock()
    s = HumanStream(lambda: clk.beat, channel=0)
    s.on_message(mido.Message("note_on", note=60, velocity=100))
    notes, _ = s.window(since_beats=1e6)  # far-future cutoff drops everything
    assert notes == []
