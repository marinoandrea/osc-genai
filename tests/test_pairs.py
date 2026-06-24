"""Same-song, time-aligned pairing: window counts, cross-track alignment, drum-safe augmentation."""

from __future__ import annotations

from osc_genai.core.note import Note
from osc_genai.data.pairs import (
    augment_pairs,
    build_aligned_pairs,
    group_by_song,
    interleave,
    window_pairs,
    _save_notes_midi,
)
from osc_genai.core.event import PARTNER, SELF, notes_to_events


def _every_bar(pitch: int, n_bars: int, bpb: float = 4.0, channel: int = 0) -> list[Note]:
    return [Note(pitch, bar * bpb, 1.0, 100, False, channel) for bar in range(n_bars)]


def test_window_count_and_require_both():
    bass = _every_bar(40, 8)            # a note on the downbeat of 8 bars
    drums = _every_bar(36, 8, channel=9)
    chunks = window_pairs(bass, drums, bpb=4.0, chunk_bars=1)  # 1-bar windows, hop = 1 bar
    assert len(chunks) == 8            # one window per bar that has notes
    assert all(c and t for c, t, _ in chunks)
    assert [bar for _, _, bar in chunks] == [0, 1, 2, 3, 4, 5, 6, 7]

    # a 4-bar window holds 4 downbeats, so 8 bars of material -> 2 non-overlapping windows
    assert len(window_pairs(bass, drums, bpb=4.0, chunk_bars=4)) == 2

    # require_both drops windows where one stem is silent
    sparse_drums = _every_bar(36, 1, channel=9)  # drums only in bar 0
    only_both = window_pairs(bass, sparse_drums, bpb=4.0, chunk_bars=1, require_both=True)
    assert len(only_both) == 1 and only_both[0][2] == 0
    kept = window_pairs(bass, sparse_drums, bpb=4.0, chunk_bars=1, require_both=False)
    assert len(kept) == 8              # without require_both, bass-only windows survive


def test_cross_track_alignment():
    # A bass note and a drum hit that sound on the SAME beat must land at the SAME offset/dt
    # within their respective chunks after windowing.
    beat = 10.0  # bar 2 (1-bar window starts at beat 8), offset 2 beats into the chunk
    bass = [Note(40, beat, 1.0, 100, False, 0)]
    drums = [Note(36, beat, 1.0, 100, False, 9)]
    chunks = window_pairs(bass, drums, bpb=4.0, chunk_bars=1)
    assert len(chunks) == 1
    ctx, tgt, bar = chunks[0]
    assert bar == 2
    assert ctx[0].start == tgt[0].start == 2.0          # shifted to shared chunk origin
    # and identical dt once encoded (2 beats * 4 steps/beat = 8)
    assert notes_to_events(ctx)[0].dt == notes_to_events(tgt)[0].dt == 8


def test_augment_leaves_drum_target_untouched():
    pairs = [([Note(40, 0.0, 1.0, 100)], [Note(36, 0.0, 1.0, 100, False, 9)])]
    out = augment_pairs(pairs, semitones=range(-2, 3), target_is_drums=True)
    assert len(out) == 5                                  # 5 transpositions of the context
    target_pitches = {t[0].pitch for _, t in out}
    assert target_pitches == {36}                         # drum pitch never shifted
    assert {c[0].pitch for c, _ in out} == {38, 39, 40, 41, 42}  # context shifted -2..+2


def test_interleave_tags_source_and_shares_clock():
    # context (bass) and target (drums) on a shared origin; interleave merges by onset.
    context = [Note(40, 0.0, 1.0, 100, False, 0), Note(43, 1.0, 1.0, 100, False, 0)]
    target = [Note(36, 0.5, 0.5, 100, False, 9)]
    events = interleave(context, target)  # default 4 steps/beat
    # one stream, onset-ordered: bass@0, drums@0.5(step2), bass@1.0(step4)
    assert [e.source for e in events] == [PARTNER, SELF, PARTNER]
    assert [e.pitch for e in events] == [40, 36, 43]
    assert [e.channel for e in events] == [0, 9, 0]
    assert [e.dt for e in events] == [0, 2, 2]  # dt computed across the MERGED stream


def test_interleave_partner_sorts_before_self_at_equal_onset():
    # A bass note and a drum hit on the same beat: partner must come first so self is conditioned on it.
    context = [Note(40, 0.0, 1.0, 100, False, 0)]
    target = [Note(36, 0.0, 1.0, 100, False, 9)]
    events = interleave(context, target)
    assert [e.source for e in events] == [PARTNER, SELF]
    assert events[1].dt == 0  # simultaneous


def test_build_aligned_pairs_from_disk(tmp_path):
    # Lay out the real store shape: <Instrument>/<Artist>/<song>__<label>.mid
    song = "tune"
    for inst, pitch, ch in [("Bass", 40, 0), ("Drums", 36, 9)]:
        d = tmp_path / inst / "Artist"
        d.mkdir(parents=True)
        _save_notes_midi(_every_bar(pitch, 8, channel=ch), d / f"{song}__{inst}.mid")
    # a Drums-only song must NOT produce pairs (no Bass partner)
    d = tmp_path / "Drums" / "Artist"
    _save_notes_midi(_every_bar(36, 4, channel=9), d / "solo__Drums.mid")

    assert set(group_by_song(tmp_path, "Bass")) == {song}
    pairs = build_aligned_pairs(tmp_path, "Bass", "Drums", chunk_bars=1, normalize_drums=False)
    assert {p.song for p in pairs} == {song}              # only the song with both stems
    assert len(pairs) == 8
    assert all(p.context and p.target and p.bars == 1 for p in pairs)
