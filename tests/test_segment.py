"""NoteSegmenter turns (f0, prob, rms) frames into clean monophonic note on/off events."""

from __future__ import annotations

import math

from osc_genai.audio.segment import NoteSegmenter, hz_to_midi


def _midi_to_hz(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


class Recorder:
    """Collect (kind, pitch) events so tests can assert the emitted note stream."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def make(self, **kw) -> NoteSegmenter:
        return NoteSegmenter(
            note_on=lambda p, v: self.events.append(("on", p)),
            note_off=lambda p: self.events.append(("off", p)),
            **kw,
        )


def test_hz_to_midi_anchors_on_a440():
    assert round(hz_to_midi(440.0)) == 69
    assert round(hz_to_midi(_midi_to_hz(40))) == 40


def test_stable_tone_then_silence_yields_one_note():
    rec = Recorder()
    seg = rec.make(smoothing=3, release_frames=2)
    hz = _midi_to_hz(45)
    for _ in range(8):  # sustained voiced pitch
        seg.process(hz, 0.95, 0.2)
    for _ in range(3):  # silence -> release
        seg.process(-1.0, 0.0, 0.0)
    assert rec.events == [("on", 45), ("off", 45)]


def test_pitch_change_closes_then_opens():
    rec = Recorder()
    seg = rec.make(smoothing=3)
    for _ in range(5):
        seg.process(_midi_to_hz(45), 0.95, 0.2)
    for _ in range(5):
        seg.process(_midi_to_hz(48), 0.95, 0.2)
    seg.flush()
    assert rec.events == [("on", 45), ("off", 45), ("on", 48), ("off", 48)]


def test_single_frame_octave_blip_is_rejected():
    rec = Recorder()
    seg = rec.make(smoothing=3)
    seq = [45, 45, 57, 45, 45, 45]  # one stray octave-up frame in a steady note
    for p in seq:
        seg.process(_midi_to_hz(p), 0.95, 0.2)
    seg.flush()
    # median smoothing keeps it a single 45 note — the 57 never opens its own note.
    assert ("on", 57) not in rec.events
    assert rec.events.count(("on", 45)) == 1


def test_low_confidence_frames_do_not_voice():
    rec = Recorder()
    seg = rec.make(smoothing=2, confidence=0.5)
    for _ in range(6):
        seg.process(_midi_to_hz(45), 0.2, 0.2)  # below confidence
    seg.flush()
    assert rec.events == []


def test_quiet_frames_below_noise_floor_do_not_voice():
    rec = Recorder()
    seg = rec.make(smoothing=2, noise_floor=0.05)
    for _ in range(6):
        seg.process(_midi_to_hz(45), 0.95, 0.001)  # below noise floor
    seg.flush()
    assert rec.events == []


def test_velocity_scales_with_loudness():
    vels: list[int] = []
    seg = NoteSegmenter(
        note_on=lambda p, v: vels.append(v),
        note_off=lambda p: None,
        smoothing=1,
        noise_floor=0.0,
        rms_ceiling=1.0,
        velocity_floor=1,
    )
    seg.process(_midi_to_hz(45), 0.95, 1.0)  # max loudness
    assert vels and vels[0] == 127
