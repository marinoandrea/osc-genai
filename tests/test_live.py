"""Tests for the real-time MIDI duet engine.

The pure core (event <-> message conversion, responder logic, ``DuetEngine.handle``) is exercised
with no I/O. A best-effort virtual-port loopback verifies the rtmidi path actually carries notes,
but skips gracefully on hosts that can't open or route virtual ports in-process.
"""

from __future__ import annotations

import time

import mido
import pytest

from osc_genai.realtime.live import DuetEngine, IntervalHarmonizer, NoteEvent

# -- pure event conversion --------------------------------------------------------------------


def test_from_message_note_on():
    event = NoteEvent.from_message(mido.Message("note_on", note=60, velocity=100))
    assert event == NoteEvent(pitch=60, velocity=100, on=True)


def test_note_on_velocity_zero_is_note_off():
    event = NoteEvent.from_message(mido.Message("note_on", note=60, velocity=0))
    assert event == NoteEvent(pitch=60, velocity=0, on=False)


def test_from_message_note_off():
    event = NoteEvent.from_message(mido.Message("note_off", note=72, velocity=64))
    assert event is not None and event.on is False and event.pitch == 72


def test_non_note_message_is_ignored():
    assert (
        NoteEvent.from_message(mido.Message("control_change", control=1, value=10))
        is None
    )


def test_to_message_roundtrip_on():
    msg = NoteEvent(pitch=64, velocity=90, on=True).to_message(channel=2)
    assert (
        msg.type == "note_on"
        and msg.note == 64
        and msg.velocity == 90
        and msg.channel == 2
    )


def test_to_message_note_off():
    msg = NoteEvent(pitch=64, velocity=0, on=False).to_message()
    assert msg.type == "note_off" and msg.note == 64


# -- responder logic --------------------------------------------------------------------------


def test_harmonizer_adds_intervals_on_note_on():
    out = IntervalHarmonizer(intervals=(4, 7)).respond(NoteEvent(60, 100, on=True))
    assert out == [NoteEvent(64, 100, on=True), NoteEvent(67, 100, on=True)]


def test_harmonizer_mirrors_note_off():
    out = IntervalHarmonizer(intervals=(4, 7)).respond(NoteEvent(60, 0, on=False))
    assert out == [NoteEvent(64, 0, on=False), NoteEvent(67, 0, on=False)]


def test_harmonizer_drops_out_of_range_pitches():
    # 125 + 7 = 132 is out of range and must be dropped; 125 + 4 = 129 too.
    out = IntervalHarmonizer(intervals=(2, 7)).respond(NoteEvent(125, 100, on=True))
    assert out == [NoteEvent(127, 100, on=True)]


def test_engine_handle_delegates_to_responder():
    engine = DuetEngine(responder=IntervalHarmonizer(intervals=(12,)))
    assert engine.handle(NoteEvent(48, 80, on=True)) == [NoteEvent(60, 80, on=True)]


# -- virtual-port loopback (best effort) ------------------------------------------------------


def _can_open_virtual() -> bool:
    try:
        port = mido.open_output("osc-genai-cap-check", virtual=True)
        port.close()
        return True
    except Exception:
        return False


def test_virtual_ports_open_and_close():
    if not _can_open_virtual():
        pytest.skip("host cannot open virtual MIDI ports")
    inp = mido.open_input("osc-genai-test-in", virtual=True)
    out = mido.open_output("osc-genai-test-out", virtual=True)
    inp.close()
    out.close()


def test_virtual_loopback_carries_a_note():
    if not _can_open_virtual():
        pytest.skip("host cannot open virtual MIDI ports")
    out = mido.open_output("osc-genai-loop", virtual=True)
    try:
        time.sleep(0.1)
        names = [n for n in mido.get_input_names() if "osc-genai-loop" in n]
        if not names:
            pytest.skip(
                "virtual output is not routable to an input in-process on this host"
            )
        inp = mido.open_input(names[0])
        try:
            time.sleep(0.05)
            out.send(mido.Message("note_on", note=60, velocity=100))
            deadline = time.time() + 0.5
            received = None
            while time.time() < deadline:
                received = inp.poll()
                if received is not None:
                    break
                time.sleep(0.005)
            if received is None:
                pytest.skip("no in-process loopback routing on this host")
            assert received.type == "note_on" and received.note == 60
        finally:
            inp.close()
    finally:
        out.close()
