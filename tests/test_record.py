"""Tests for the duet session recorder (note assembly + persistence)."""

from __future__ import annotations

import mido

from osc_genai.core.note import Note
from osc_genai.data.record import StreamRecorder, load_session, save_session


def _on(pitch, vel=100):
    return mido.Message("note_on", note=pitch, velocity=vel)


def _off(pitch):
    return mido.Message("note_off", note=pitch, velocity=0)


def test_stream_recorder_assembles_notes_in_beats():
    rec = StreamRecorder()
    rec.message(_on(60), 1.0)
    rec.message(_off(60), 1.5)  # 0.5s = 1 beat at 120 bpm
    rec.message(_on(64, 90), 2.0)
    rec.message(_off(64), 2.25)  # 0.25s = 0.5 beat
    # t0 = 1.0 (origin), bpm 120 -> 0.5 s/beat
    assert rec.notes(t0=1.0, bpm=120) == [Note(60, 0.0, 1.0, 100), Note(64, 2.0, 0.5, 90)]


def test_unmatched_note_off_is_ignored():
    rec = StreamRecorder()
    rec.message(_off(60), 0.5)  # no preceding note_on
    rec.message(_on(60), 1.0)
    rec.message(_off(60), 2.0)
    assert rec.notes(t0=0.0, bpm=60) == [Note(60, 1.0, 1.0, 100)]  # 60 bpm -> 1 s/beat


def test_note_on_zero_velocity_ends_note():
    rec = StreamRecorder()
    rec.message(_on(60), 0.0)
    rec.message(_on(60, 0), 1.0)  # velocity-0 note_on == note_off
    assert rec.notes(t0=0.0, bpm=60) == [Note(60, 0.0, 1.0, 100)]


def test_save_load_session_roundtrip(tmp_path):
    human = [Note(60, 0.0, 1.0, 100)]
    machine = [Note(67, 0.5, 0.5, 90, False)]
    path = tmp_path / "session.json"
    save_session(human, machine, bpm=128.0, path=path)
    bpm, h, m = load_session(path)
    assert bpm == 128.0 and h == human and m == machine
