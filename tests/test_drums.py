"""Tests for drum role inference + GM normalization."""

from __future__ import annotations

from osc_genai.core.note import Note
from osc_genai.data.drums import infer_drum_map, normalize_drums, regularize_drums


def _kit_clip():
    notes = []
    notes += [
        Note(60, float(b), 0.25, 100) for b in range(4)
    ]  # kick (60) four-on-floor
    notes += [Note(62, float(b), 0.25, 110) for b in (1, 3)]  # snare (62) on backbeat
    notes += [Note(66, i * 0.5, 0.25, 80) for i in range(8)]  # hat (66) on every 8th
    return notes


def test_infer_drum_map_assigns_gm_roles():
    mapping = infer_drum_map(_kit_clip())
    assert mapping[62] == 38  # backbeat note -> snare
    assert mapping[60] == 36  # lowest -> kick
    assert mapping[66] == 42  # densest -> closed hat


def test_normalize_remaps_into_gm_range():
    out = normalize_drums(_kit_clip())
    assert {n.pitch for n in out} <= {36, 38, 42, 46}  # all land on GM kit notes
    assert all(36 <= n.pitch <= 51 for n in out)  # in GM percussion range


def test_explicit_mapping_overrides_inference():
    notes = [Note(7, 0.0, 0.25, 100), Note(22, 0.0, 0.25, 80)]
    out = normalize_drums(notes, mapping={7: 36, 22: 42})
    assert sorted(n.pitch for n in out) == [36, 42]


def test_regularize_snaps_kick_snare_but_not_hats():
    notes = [
        Note(36, 0.06, 0.25, 100),  # kick slightly late -> snaps to 0.0
        Note(38, 1.10, 0.25, 110),  # snare off-grid -> snaps to 1.0 (8th grid)
        Note(42, 0.31, 0.25, 80),  # hat off-grid -> LEFT ALONE (creative lane)
    ]
    out = regularize_drums(notes, grid_beats=0.5)
    by_pitch = {n.pitch: n.start for n in out}
    assert by_pitch[36] == 0.0
    assert by_pitch[38] == 1.0
    assert by_pitch[42] == 0.31  # hat timing preserved


def test_regularize_dedupes_collapsed_hits():
    notes = [Note(36, 0.00, 0.25, 100), Note(36, 0.10, 0.25, 90)]  # both snap to 0.0
    out = regularize_drums(notes, grid_beats=0.5)
    assert len(out) == 1 and out[0].start == 0.0


def test_is_kit_distinguishes_kits_from_stems():
    from osc_genai.data.drums import is_kit

    kit = [Note(36, 0, 0.25, 100), Note(38, 1, 0.25, 100), Note(42, 0, 0.25, 80)]
    stem = [Note(48, float(b), 0.25, 100) for b in range(8)]  # one distinct note
    assert is_kit(kit) and not is_kit(stem)
