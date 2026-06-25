"""Normalize heterogeneous drum MIDI to a consistent General MIDI mapping.

Producer drum clips use arbitrary, kit-specific note numbers (one song's kick is note 60, another's
is 7), so a model trained across them emits incoherent, off-lane notes. This module infers each
clip's roles (kick / snare / hat / ...) from frequency + metric position and remaps them to GM
(36 kick, 38 snare, 42 closed-hat, 46 open-hat, then tom/perc slots). It is heuristic and imperfect
— pair it with :func:`format_report` to audit, and pass an explicit ``mapping`` to override.
"""

from __future__ import annotations

from collections import defaultdict

from osc_genai.core.note import Note

GM_KICK, GM_SNARE, GM_CHAT, GM_OHAT = 36, 38, 42, 46
# Fallback GM slots for the remaining notes, assigned by ascending source pitch.
GM_OTHER = [
    45,
    47,
    48,
    50,
    41,
    43,
    51,
    49,
    39,
    37,
    56,
    54,
]  # toms, ride, crash, clap, rim, cowbell
_ROLE = {36: "kick", 38: "snare", 42: "c-hat", 46: "o-hat"}


def _beat_of_bar(start: float, bar_beats: int = 4) -> int:
    return int(round(start)) % bar_beats


def analyze_drums(notes: list[Note], bar_beats: int = 4) -> list[dict]:
    """Per distinct note: count, fraction on the backbeat (beats 1 & 3) and on beat 0, mean velocity."""
    groups: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        groups[note.pitch].append(note)
    rows = []
    for pitch, hits in groups.items():
        beats = [_beat_of_bar(h.start, bar_beats) for h in hits]
        count = len(hits)
        rows.append(
            {
                "note": pitch,
                "count": count,
                "backbeat": sum(b in (1, 3) for b in beats) / count,
                "beat0": sum(b == 0 for b in beats) / count,
                "vel": sum(h.velocity for h in hits) / count,
            }
        )
    return sorted(rows, key=lambda r: -r["count"])


def infer_drum_map(notes: list[Note], bar_beats: int = 4) -> dict[int, int]:
    """Infer a ``{source_note: gm_note}`` mapping for one drum clip."""
    rows = analyze_drums(notes, bar_beats)
    used: set[int] = set()
    mapping: dict[int, int] = {}

    def pick(key):
        candidates = [r for r in rows if r["note"] not in used]
        return max(candidates, key=key) if candidates else None

    snare = pick(
        lambda r: (r["backbeat"], r["count"])
    )  # backbeat is the strongest role cue
    if snare and snare["count"] >= 2 and snare["backbeat"] > 0:
        mapping[snare["note"]] = GM_SNARE
        used.add(snare["note"])
    kick = pick(lambda r: -r["note"])  # kick usually sits lowest
    if kick:
        mapping[kick["note"]] = GM_KICK
        used.add(kick["note"])
    chat = pick(lambda r: r["count"])  # the densest remaining note carries the hats
    if chat:
        mapping[chat["note"]] = GM_CHAT
        used.add(chat["note"])
    ohat = pick(lambda r: r["count"])
    if ohat:
        mapping[ohat["note"]] = GM_OHAT
        used.add(ohat["note"])
    for i, note in enumerate(sorted(r["note"] for r in rows if r["note"] not in used)):
        mapping[note] = GM_OTHER[min(i, len(GM_OTHER) - 1)]
    return mapping


def normalize_drums(
    notes: list[Note], mapping: dict[int, int] | None = None, bar_beats: int = 4
) -> list[Note]:
    """Remap a drum clip's pitches to GM (inferring the mapping if not given)."""
    mapping = mapping if mapping is not None else infer_drum_map(notes, bar_beats)
    return [n._replace(pitch=mapping.get(n.pitch, n.pitch)) for n in notes]


def regularize_drums(
    notes: list[Note],
    grid_beats: float = 0.5,
    pitches: tuple[int, ...] = (GM_KICK, GM_SNARE),
) -> list[Note]:
    """Snap kick/snare onsets to a coarser grid, leaving hats/percussion free.

    Electronic kick/snare almost never leave the 8th/quarter grid, so 16th-note jitter on those
    lanes is noise the model shouldn't learn. Snapping their onsets to ``grid_beats`` (default 0.5 =
    8th notes) regularizes the training signal for them while ``pitches`` outside the set (hats,
    effects) keep their fine timing. Two snapped hits that collapse onto the same step are de-duped.
    Expects GM-normalized input (see :func:`normalize_drums`), so kick=36 / snare=38.
    """
    out: list[Note] = []
    seen: set[tuple[int, float]] = set()
    for n in notes:
        if n.pitch in pitches:
            start = round(n.start / grid_beats) * grid_beats
            key = (n.pitch, round(start, 6))
            if key in seen:  # a doubled kick/snare on the same grid step is redundant
                continue
            seen.add(key)
            out.append(n._replace(start=start))
        else:
            out.append(n)
    return sorted(out, key=lambda x: (x.start, x.pitch))


def is_kit(notes: list[Note], min_distinct: int = 3) -> bool:
    """A full kit (vs an extracted single-element stem) uses several distinct drum notes."""
    return len({n.pitch for n in notes}) >= min_distinct


def load_drum_kits(
    dirs, min_distinct: int = 3, normalize: bool = True
) -> list[list[Note]]:
    """Load drum clips from ``dirs``, keep only full kits (drop extracted stems), normalize to GM."""
    from osc_genai.data.midi import load_midi_dir

    kits: list[list[Note]] = []
    for directory in dirs:
        for seq in load_midi_dir(directory):
            if seq and is_kit(seq, min_distinct):
                kits.append(normalize_drums(seq) if normalize else seq)
    return kits


def format_report(name: str, notes: list[Note], bar_beats: int = 4) -> str:
    """Human-readable per-clip note histogram + the inferred GM role mapping (for auditing)."""
    rows = analyze_drums(notes, bar_beats)
    mapping = infer_drum_map(notes, bar_beats)
    lines = [f"{name}  ({len(notes)} notes, {len(rows)} distinct)"]
    for r in rows:
        gm = mapping.get(r["note"], r["note"])
        role = _ROLE.get(gm, f"GM{gm}")
        lines.append(
            f"    note {r['note']:>3} x{r['count']:<4} backbeat={r['backbeat']:>4.0%} "
            f"beat0={r['beat0']:>4.0%}  ->  {gm:>3} ({role})"
        )
    return "\n".join(lines)
