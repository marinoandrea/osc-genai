"""Tests for MIDI ingest, augmentation, and persistence (pure / file-based parts)."""

from __future__ import annotations

import mido
import numpy as np

from osc_genai.data.midi import (
    augment,
    jitter_velocity,
    load_midi_file,
    load_sequences,
    save_sequences,
    scale_time,
    transpose,
)
from osc_genai.core.note import Note


def test_load_midi_file_recovers_notes(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))  # 1 beat
    track.append(mido.Message("note_on", note=64, velocity=80, time=0))  # at beat 1
    track.append(mido.Message("note_off", note=64, velocity=0, time=240))  # 0.5 beat
    path = tmp_path / "phrase.mid"
    mid.save(str(path))

    notes = load_midi_file(path)
    assert notes == [Note(60, 0.0, 1.0, 100), Note(64, 1.0, 0.5, 80)]


def test_transpose_shifts_and_drops_out_of_range():
    notes = [Note(60, 0.0, 1.0, 100), Note(125, 0.0, 1.0, 100)]
    up = transpose(notes, 5)
    assert [n.pitch for n in up] == [65]  # 130 dropped


def test_jitter_velocity_is_bounded_and_deterministic():
    rng = np.random.default_rng(0)
    notes = [Note(60, 0.0, 1.0, 100), Note(62, 1.0, 1.0, 1)]
    out = jitter_velocity(notes, amount=10, rng=rng)
    assert all(1 <= n.velocity <= 127 for n in out)
    assert all(abs(o.velocity - n.velocity) <= 10 for o, n in zip(out, notes))


def test_scale_time():
    out = scale_time([Note(60, 1.0, 2.0, 100)], factor=0.5)
    assert out == [Note(60, 0.5, 1.0, 100)]


def test_augment_expands_by_transposition_count():
    seqs = [[Note(60, 0.0, 1.0, 100)]]
    out = augment(seqs, semitones=range(-2, 3))  # 5 transpositions, all in range
    assert len(out) == 5
    assert sorted(seq[0].pitch for seq in out) == [58, 59, 60, 61, 62]


def test_save_load_sequences_roundtrip(tmp_path):
    seqs = [[Note(60, 0.0, 1.0, 100, False)], [Note(62, 0.0, 0.5, 80, True)]]
    path = tmp_path / "data.json"
    save_sequences(seqs, path)
    assert load_sequences(path) == seqs


def test_combine_parts_assigns_channels_and_orders():
    from osc_genai.data.midi import combine_parts

    bass = [Note(40, 0.0, 1.0, 100), Note(43, 1.0, 1.0, 100)]
    drums = [Note(36, 0.0, 0.25, 110), Note(42, 0.5, 0.25, 90)]
    merged = combine_parts([(bass, 0), (drums, 9)])
    assert {n.channel for n in merged} == {0, 9}
    assert [n.start for n in merged] == [0.0, 0.0, 0.5, 1.0]  # onset-ordered across channels
    assert all(n.channel == 9 for n in merged if n.pitch in (36, 42))


def test_cross_pairs_counts_and_is_deterministic():
    from osc_genai.data.midi import cross_pairs

    ctx = [[Note(60, 0, 1, 100)], [Note(62, 0, 1, 100)]]
    tgt = [[Note(36, 0, 0.5, 100)], [Note(38, 0, 0.5, 100)], [Note(42, 0, 0.5, 100)]]
    pairs = cross_pairs(ctx, tgt, k=2, seed=0)
    assert len(pairs) == 2 * 2  # 2 context clips x 2 targets each
    assert cross_pairs(ctx, tgt, k=2, seed=0) == pairs  # deterministic for a fixed seed
    assert all(c in ctx and t in tgt for c, t in pairs)
