"""Same-song, time-aligned instrument-pair clips for conditional training.

The corpus is stored as single-instrument clips under ``data/MIDI/<Instrument>/<Artist>/`` named
``<source_song>__<label>.mid``. To train a directional model (e.g. **bass -> drums**) we need the
two stems *from the same song*, **time-aligned** and cut into fixed-length (4-8 bar) chunks.

This module reconstructs that without restructuring the store:

* :func:`group_by_song` regroups an instrument's clips by their ``source_song`` prefix and merges
  them into one stem (notes already share the song's beat timeline).
* :func:`window_pairs` slides a bar-length window over a song and cuts *both* stems on the **same**
  grid, shifting each chunk to a shared origin — this is what preserves cross-track alignment.
* :func:`build_aligned_pairs` ties it together for an instrument pair; the result is the exact
  ``list[(context_notes, target_notes)]`` shape the existing :func:`train.train_conditional`
  pipeline consumes (after :func:`repr.notes_to_events`).

Pairs can be fed straight to training (dynamic, no duplication) or written out with
:func:`materialize` as a browsable ``<Ctx>_to_<Tgt>/<Artist>/`` tree.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mido

from osc_genai.data.midi import load_midi_file, transpose
from osc_genai.core.note import Note
from osc_genai.core.event import DEFAULT_STEPS_PER_BEAT, PARTNER, SELF, Event, _quantize

DRUM_CHANNEL = 9


# -- grouping ---------------------------------------------------------------------------------

@dataclass
class SongStem:
    artist: str
    notes: list[Note]
    paths: list[Path]


def group_by_song(root: str | Path, instrument: str) -> dict[str, SongStem]:
    """Regroup ``data/MIDI/<instrument>/**`` clips by source song, merging each song's clips.

    The song key is the filename prefix before ``__`` (how the splitter named clips). Notes from
    every clip of that instrument in the song are merged into one onset-ordered stem.
    """
    base = Path(root) / instrument
    songs: dict[str, SongStem] = {}
    for path in sorted([*base.rglob("*.mid"), *base.rglob("*.midi")]):
        if "__" not in path.name:
            continue
        song = path.name.split("__", 1)[0]
        stem = songs.get(song)
        if stem is None:
            stem = songs[song] = SongStem(artist=path.parent.name, notes=[], paths=[])
        stem.notes.extend(load_midi_file(path))
        stem.paths.append(path)
    for stem in songs.values():
        stem.notes.sort(key=lambda n: (n.start, n.pitch))
    return songs


def bar_beats(path: str | Path) -> float:
    """Beats (quarter notes) per bar, read from a clip's time signature (default 4/4 -> 4.0)."""
    num, den = 4, 4
    for track in mido.MidiFile(str(path)).tracks:
        sig = next((m for m in track if m.type == "time_signature"), None)
        if sig is not None:
            num, den = sig.numerator, sig.denominator
            break
    return num * 4.0 / den


# -- windowing --------------------------------------------------------------------------------

def _slice(notes: list[Note], w0: float, w1: float) -> list[Note]:
    """Notes whose onset falls in ``[w0, w1)``, shifted so the window starts at 0."""
    return [n._replace(start=n.start - w0) for n in notes if w0 <= n.start < w1]


def window_pairs(
    context: list[Note],
    target: list[Note],
    bpb: float,
    chunk_bars: int,
    hop_bars: int | None = None,
    require_both: bool = True,
) -> list[tuple[list[Note], list[Note], int]]:
    """Cut both stems into aligned ``chunk_bars``-long windows.

    Returns ``[(context_chunk, target_chunk, bar_index), ...]``. Both chunks are shifted by the
    *same* window start, so a context note and a target note that sound on the same beat keep the
    same offset within their chunks (alignment). ``hop_bars`` defaults to ``chunk_bars``
    (non-overlapping); a smaller hop overlaps windows for more samples. With ``require_both`` a
    window is dropped unless *both* stems have notes in it.
    """
    win = chunk_bars * bpb
    hop = (hop_bars or chunk_bars) * bpb
    end = max((n.start for n in (*context, *target)), default=-1.0)
    out: list[tuple[list[Note], list[Note], int]] = []
    w0 = 0.0
    while w0 <= end:
        ctx_chunk = _slice(context, w0, w0 + win)
        tgt_chunk = _slice(target, w0, w0 + win)
        empty = (require_both and (not ctx_chunk or not tgt_chunk)) or (not ctx_chunk and not tgt_chunk)
        if not empty:
            out.append((ctx_chunk, tgt_chunk, int(round(w0 / bpb))))
        w0 += hop
    return out


# -- pair building ----------------------------------------------------------------------------

@dataclass
class AlignedPair:
    context: list[Note]
    target: list[Note]
    song: str
    artist: str
    bar: int
    bars: int


def build_aligned_pairs(
    root: str | Path,
    ctx_inst: str,
    tgt_inst: str,
    chunk_bars: int = 4,
    hop_bars: int | None = None,
    sizes: Iterable[int] | None = None,
    normalize_drums: bool = True,
    regularize: float | None = None,
    require_both: bool = True,
) -> list[AlignedPair]:
    """Build same-song, time-aligned ``ctx_inst -> tgt_inst`` chunk pairs across the corpus.

    Only songs that contain *both* instruments contribute. ``sizes`` (e.g. ``[4, 8]``) emits several
    chunk lengths from the same songs. When the target is drums it is GM-normalized first
    (reusing :mod:`drums`) so kick/snare/hat land on consistent lanes; ``regularize`` (a grid in
    beats, e.g. 0.5) additionally snaps kick/snare onsets to that grid so they train regular.
    """
    chunk_sizes = list(sizes) if sizes is not None else [chunk_bars]
    ctx_songs = group_by_song(root, ctx_inst)
    tgt_songs = group_by_song(root, tgt_inst)

    pairs: list[AlignedPair] = []
    for song in sorted(set(ctx_songs) & set(tgt_songs)):
        ctx_stem, tgt_stem = ctx_songs[song], tgt_songs[song]
        tgt_notes = tgt_stem.notes
        if tgt_inst.lower() == "drums":
            if normalize_drums:
                from osc_genai.data.drums import normalize_drums as _normalize

                tgt_notes = _normalize(tgt_notes)
            if regularize:
                from osc_genai.data.drums import regularize_drums as _regularize

                tgt_notes = _regularize(tgt_notes, grid_beats=regularize)
        bpb = bar_beats(ctx_stem.paths[0])
        for size in chunk_sizes:
            for ctx_chunk, tgt_chunk, bar in window_pairs(
                ctx_stem.notes, tgt_notes, bpb, size, hop_bars, require_both
            ):
                pairs.append(AlignedPair(ctx_chunk, tgt_chunk, song, ctx_stem.artist, bar, size))
    return pairs


def note_pairs(pairs: list[AlignedPair]) -> list[tuple[list[Note], list[Note]]]:
    """Drop metadata: ``AlignedPair`` list -> ``[(context, target), ...]`` for training."""
    return [(p.context, p.target) for p in pairs]


def augment_pairs(
    pairs: list[tuple[list[Note], list[Note]]],
    semitones: Iterable[int] = range(-5, 6),
    target_is_drums: bool = True,
) -> list[tuple[list[Note], list[Note]]]:
    """Transpose-augment pairs. The drum target is left untouched (drums are unpitched); for a
    pitched target both sides shift together so harmony stays consistent. Timing is unchanged, so
    alignment is preserved. Pairs that transpose to an empty side are dropped."""
    out: list[tuple[list[Note], list[Note]]] = []
    for ctx, tgt in pairs:
        for semi in semitones:
            ctx_t = transpose(ctx, semi)
            tgt_t = tgt if target_is_drums else transpose(tgt, semi)
            if ctx_t and tgt_t:
                out.append((ctx_t, tgt_t))
    return out


# -- interleaving (paired -> single source-tagged stream) -------------------------------------

def interleave(
    context: list[Note], target: list[Note], steps_per_beat: int = DEFAULT_STEPS_PER_BEAT
) -> list[Event]:
    """Merge an aligned ``(context, target)`` pair into one source-tagged event stream.

    ``context`` notes become ``PARTNER`` events, ``target`` notes ``SELF`` events. Both lists already
    share an origin (:func:`window_pairs`), so we just merge by onset and compute ``dt`` across the
    *combined* stream — the shared clock the duet model conditions on. At an equal onset the partner
    note sorts first (``source`` is the tiebreak after ``start``), so a ``SELF`` event is conditioned
    on the partner's concurrent hit. Each note keeps its own channel.
    """
    tagged = [(n, PARTNER) for n in context if not n.mute]
    tagged += [(n, SELF) for n in target if not n.mute]
    tagged.sort(key=lambda ns: (ns[0].start, ns[1], ns[0].pitch))  # PARTNER(0) before SELF(1)
    events: list[Event] = []
    prev_onset = 0
    for note, source in tagged:
        onset = _quantize(note.start, steps_per_beat)
        events.append(
            Event(
                pitch=note.pitch,
                dt=max(0, onset - prev_onset),
                dur=max(1, _quantize(note.duration, steps_per_beat)),
                velocity=note.velocity,
                channel=note.channel,
                source=source,
            )
        )
        prev_onset = onset
    return events


def interleave_pairs(
    pairs: list[tuple[list[Note], list[Note]]], steps_per_beat: int = DEFAULT_STEPS_PER_BEAT
) -> list[list[Event]]:
    """Interleave every ``(context, target)`` pair into one source-tagged event sequence each."""
    return [interleave(ctx, tgt, steps_per_beat) for ctx, tgt in pairs]


# -- materialization (optional, for inspection) -----------------------------------------------

def _save_notes_midi(notes: list[Note], path: Path, ticks_per_beat: int = 480) -> None:
    """Write a multi-channel Note list to a single-track ``.mid`` (channels preserved)."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    events: list[tuple[int, int, Note, bool]] = []
    for note in notes:
        on = int(round(note.start * ticks_per_beat))
        off = on + max(1, int(round(note.duration * ticks_per_beat)))
        events.append((on, 1, note, True))
        events.append((off, 0, note, False))
    events.sort(key=lambda e: (e[0], e[1]))  # note_off before note_on at the same tick
    last = 0
    for tick, _, note, is_on in events:
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=max(0, min(127, note.pitch)),
                velocity=note.velocity if is_on else 0,
                channel=max(0, min(15, note.channel)),
                time=tick - last,
            )
        )
        last = tick
    mid.save(str(path))


def materialize(
    pairs: list[AlignedPair],
    out_dir: str | Path,
    ctx_inst: str,
    tgt_inst: str,
    tgt_channel: int = DRUM_CHANNEL,
) -> Path:
    """Write each pair as one 2-channel ``.mid`` under ``<out>/<Ctx>_to_<Tgt>/<Artist>/`` + a
    ``pairs.csv`` manifest. A browsable view of the dynamically-built pairs; not needed for training.
    """
    root = Path(out_dir) / f"{ctx_inst}_to_{tgt_inst}"
    rows = []
    for pair in pairs:
        dest = root / pair.artist
        dest.mkdir(parents=True, exist_ok=True)
        ctx = [n._replace(channel=0) for n in pair.context]
        tgt = [n._replace(channel=tgt_channel) for n in pair.target]
        combined = sorted(ctx + tgt, key=lambda n: (n.start, n.pitch))
        name = f"{pair.song}_{pair.bars}bar_{pair.bar:03d}.mid"
        _save_notes_midi(combined, dest / name)
        rows.append([f"{ctx_inst}_to_{tgt_inst}", pair.artist, name, pair.song, pair.bars,
                     pair.bar, len(pair.context), len(pair.target)])
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "pairs.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair", "artist", "file", "song", "bars", "bar", "ctx_notes", "tgt_notes"])
        writer.writerows(rows)
    return root


# -- CLI: inspect "how much data" -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build same-song, time-aligned instrument-pair chunks; print stats / materialize."
    )
    parser.add_argument("--data-dir", default="data/MIDI")
    parser.add_argument("--context-inst", default="Bass")
    parser.add_argument("--target-inst", default="Drums")
    parser.add_argument("--chunk-bars", type=int, default=4)
    parser.add_argument("--also-8", action="store_true", help="also emit 8-bar chunks")
    parser.add_argument("--hop-bars", type=int, default=None, help="window hop (default = chunk size)")
    parser.add_argument("--no-require-both", action="store_true", help="keep windows with one stem empty")
    parser.add_argument("--no-normalize-drums", action="store_true")
    parser.add_argument("--materialize", default=None, help="write a <Ctx>_to_<Tgt>/ tree here")
    args = parser.parse_args()

    sizes = [args.chunk_bars] + ([8] if args.also_8 and args.chunk_bars != 8 else [])
    pairs = build_aligned_pairs(
        args.data_dir, args.context_inst, args.target_inst,
        chunk_bars=args.chunk_bars, hop_bars=args.hop_bars, sizes=sizes,
        normalize_drums=not args.no_normalize_drums, require_both=not args.no_require_both,
    )
    by_song: dict[str, int] = {}
    for pair in pairs:
        by_song[f"{pair.artist} / {pair.song}"] = by_song.get(f"{pair.artist} / {pair.song}", 0) + 1
    print(f"{args.context_inst} -> {args.target_inst}: {len(by_song)} songs, {len(pairs)} chunks "
          f"(sizes={sizes} bars)")
    for song, n in sorted(by_song.items()):
        print(f"  {n:4d}  {song}")
    if args.materialize:
        out = materialize(pairs, args.materialize, args.context_inst, args.target_inst)
        print(f"materialized -> {out}")


if __name__ == "__main__":
    main()
