"""Getting the musician's MIDI in, and growing a small personal set with augmentation.

Two ingest front-ends, both yielding lists of clip ``Note`` sequences (one list per phrase/clip):

* :func:`load_midi_file` / :func:`load_midi_dir` — parse ``.mid`` files (mido), in beats.
* :func:`capture_from_ableton` — pull clips straight out of a live set via the OSC read path.

Augmentation (transposition is the big multiplier for a one-musician corpus, plus velocity jitter
and time-scaling) expands the data; :func:`save_sequences` / :func:`load_sequences` persist it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import mido
import numpy as np

from osc_genai.osc.ableton import AbletonOSC
from osc_genai.core.note import Note

# -- MIDI files -------------------------------------------------------------------------------

def load_midi_file(path: str | Path) -> list[Note]:
    """Parse a ``.mid`` file into onset-ordered ``Note``s, timed in beats (tempo-independent)."""
    mid = mido.MidiFile(str(path))
    ticks_per_beat = mid.ticks_per_beat
    notes: list[Note] = []
    active: dict[tuple[int, int], tuple[int, int]] = {}  # (channel, pitch) -> (start_tick, vel)
    abs_tick = 0
    for msg in mido.merge_tracks(mid.tracks):
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            started = active.pop((msg.channel, msg.note), None)
            if started is not None:
                start_tick, velocity = started
                notes.append(
                    Note(
                        pitch=msg.note,
                        start=start_tick / ticks_per_beat,
                        duration=max(0.0, (abs_tick - start_tick) / ticks_per_beat),
                        velocity=velocity,
                        channel=msg.channel,
                    )
                )
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def load_midi_dir(directory: str | Path) -> list[list[Note]]:
    """Load every ``.mid`` / ``.midi`` file under a directory (recursively) as one sequence each."""
    root = Path(directory)
    paths = sorted([*root.rglob("*.mid"), *root.rglob("*.midi")])
    return [load_midi_file(p) for p in paths]


# -- Ableton capture --------------------------------------------------------------------------

def capture_from_ableton(
    live: AbletonOSC, slots: int = 8, tracks: Iterable[int] | None = None
) -> list[list[Note]]:
    """Read every non-empty clip (up to ``slots`` per track) out of the live set."""
    track_indices = list(tracks) if tracks is not None else range(live.get_num_tracks())
    sequences: list[list[Note]] = []
    for track in track_indices:
        for slot in range(slots):
            if live.has_clip(track, slot):
                notes = live.get_clip_notes(track, slot)
                if notes:
                    sequences.append(notes)
    return sequences


# -- augmentation -----------------------------------------------------------------------------

def transpose(notes: list[Note], semitones: int) -> list[Note]:
    """Shift pitch; notes pushed outside 0-127 are dropped."""
    out = []
    for note in notes:
        pitch = note.pitch + semitones
        if 0 <= pitch <= 127:
            out.append(note._replace(pitch=pitch))
    return out


def jitter_velocity(notes: list[Note], amount: int, rng: np.random.Generator) -> list[Note]:
    """Add uniform noise in ``[-amount, amount]`` to each velocity (clamped to 1-127)."""
    return [
        note._replace(velocity=int(np.clip(note.velocity + rng.integers(-amount, amount + 1), 1, 127)))
        for note in notes
    ]


def scale_time(notes: list[Note], factor: float) -> list[Note]:
    """Stretch/compress onsets and durations by ``factor`` (tempo feel, same pitches)."""
    return [note._replace(start=note.start * factor, duration=note.duration * factor) for note in notes]


def augment(
    sequences: list[list[Note]], semitones: Iterable[int] = range(-5, 7)
) -> list[list[Note]]:
    """Expand a corpus by transposing each sequence across ``semitones`` (drops empty results)."""
    out: list[list[Note]] = []
    for seq in sequences:
        for semitone in semitones:
            transposed = transpose(seq, semitone)
            if transposed:
                out.append(transposed)
    return out


def cross_pairs(
    context: list[list[Note]], target: list[list[Note]], k: int = 4, seed: int = 0
) -> list[tuple[list[Note], list[Note]]]:
    """Build (context, target) pairs: each context clip paired with k random target clips.

    Used for directional cross-role snapshots (e.g. bass->drums): the model learns to respond in the
    target role given the context role. Drums are key-agnostic so any-to-any pairing is plausible;
    co-performed pairs (via the session recorder) give tighter coupling.
    """
    rng = random.Random(seed)
    pairs: list[tuple[list[Note], list[Note]]] = []
    for ctx in context:
        for tgt in rng.sample(target, k=min(k, len(target))):
            pairs.append((ctx, tgt))
    return pairs


def combine_parts(parts: list[tuple[list[Note], int]]) -> list[Note]:
    """Merge single-instrument sequences onto distinct channels into one multi-channel arrangement.

    ``parts`` is ``[(notes, channel), ...]`` (e.g. bass on 0, drums on 9). Returns one onset-ordered
    Note list spanning all channels — training material for a model that plays several instruments
    at once.
    """
    merged: list[Note] = []
    for notes, channel in parts:
        merged.extend(note._replace(channel=channel) for note in notes)
    return sorted(merged, key=lambda n: (n.start, n.pitch))


# -- persistence ------------------------------------------------------------------------------

def save_sequences(sequences: list[list[Note]], path: str | Path) -> None:
    """Persist Note sequences as JSON (a list of clips, each a list of note tuples)."""
    data = [[list(note) for note in seq] for seq in sequences]
    Path(path).write_text(json.dumps(data))


def load_sequences(path: str | Path) -> list[list[Note]]:
    """Inverse of :func:`save_sequences`."""
    data = json.loads(Path(path).read_text())
    return [[Note(*note) for note in seq] for seq in data]
