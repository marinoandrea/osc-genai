"""Save the last N bars of a live duet into the personal dataset as a training pair.

A snapshot grabs a trailing window of both parts of a performance — the human's and the model's
— and writes them as two single-instrument ``.mid`` clips that **share a song-id prefix**, one
under ``<root>/<human_inst>/<artist>/`` and one under ``<root>/<machine_inst>/<artist>/``. That
is exactly the on-disk shape :mod:`osc_genai.data.pairs` reconstructs ``(context, target)`` pairs
from (:func:`group_by_song` keys on the ``<song>__`` prefix, :func:`build_aligned_pairs` matches
the same song across two instrument folders), so a snapshot becomes a genuine training pair with
no extra plumbing. This only *produces data*; it does not train.
"""

from __future__ import annotations

import math
from pathlib import Path

from osc_genai.core.note import Note
from osc_genai.data.midi import save_notes_midi
from osc_genai.data.pairs import _slice


def save_snapshot(
    human: list[Note],
    machine: list[Note],
    out_root: str | Path,
    *,
    end_beat: float,
    bars: int,
    beats_per_bar: float,
    human_inst: str,
    machine_inst: str,
    artist: str = "personal",
    song_id: str,
    label_human: str = "human",
    label_machine: str = "machine",
) -> tuple[Path, Path] | None:
    """Slice the last ``bars`` bars and save both parts as one aligned pair.

    The window ends at the last completed bar boundary at or before ``end_beat`` and spans
    ``bars`` bars back. Both parts are sliced and re-origined to a shared 0 with the same
    primitive :func:`pairs.window_pairs` uses (:func:`pairs._slice`), so they stay time-aligned.

    Returns the two written paths ``(human_path, machine_path)``, or ``None`` when the window
    starts before beat 0 (not enough has been played yet) or both slices are empty.
    """
    end = math.floor(end_beat / beats_per_bar) * beats_per_bar
    start = end - bars * beats_per_bar
    if start < 0:
        return None

    human_chunk = _slice(human, start, end)
    machine_chunk = _slice(machine, start, end)
    if not human_chunk and not machine_chunk:
        return None

    human_path = Path(out_root) / human_inst / artist / f"{song_id}__{label_human}.mid"
    machine_path = Path(out_root) / machine_inst / artist / f"{song_id}__{label_machine}.mid"
    for path, chunk in ((human_path, human_chunk), (machine_path, machine_chunk)):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_notes_midi(chunk, path)
    return human_path, machine_path
