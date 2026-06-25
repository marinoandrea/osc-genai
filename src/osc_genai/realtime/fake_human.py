"""A fake human: loop a MIDI line into the duet's input port, for a mock local setup.

No controller or Ableton input-routing needed — this plays a looping sequence (one of your own
clips, a ``.mid`` file, or a built-in acid pattern) out to the port the duet listens on
(``osc-genai in``), so you can hear/verify the duet respond. Two terminals::

    uv run duet --checkpoint models/acid_v1.pt        # terminal 1 (creates the ports)
    uv run fake-human --from-data data/MIDI           # terminal 2 (drives the input)

Then route ``osc-genai out`` to a synth in Ableton to hear the model's response.
"""

from __future__ import annotations

import heapq
import math
import os
import random
import time

# python-rtmidi is mido's real-time backend; select it before mido resolves a default.
os.environ.setdefault("MIDO_BACKEND", "mido.backends.rtmidi")
import mido  # noqa: E402

from osc_genai.cli_spec import REGISTRY, build_parser  # noqa: E402
from osc_genai.core.note import Note  # noqa: E402
from osc_genai.data.midi import load_midi_dir, load_midi_file  # noqa: E402
from osc_genai.realtime.clock import make_clock  # noqa: E402

DEFAULT_TARGET = "osc-genai in"

# A built-in one-bar acid loop (16th notes; root E2 with octave/interval jumps) used when no
# data folder or file is given.
_ACID = [
    Note(40, 0.00, 0.25, 110),
    Note(52, 0.25, 0.25, 90),
    Note(40, 0.50, 0.25, 110),
    Note(43, 0.75, 0.25, 90),
    Note(40, 1.00, 0.25, 110),
    Note(52, 1.25, 0.25, 90),
    Note(45, 1.50, 0.25, 100),
    Note(40, 1.75, 0.25, 110),
    Note(40, 2.00, 0.25, 110),
    Note(55, 2.25, 0.25, 90),
    Note(40, 2.50, 0.25, 110),
    Note(43, 2.75, 0.25, 90),
    Note(40, 3.00, 0.25, 110),
    Note(52, 3.25, 0.25, 90),
    Note(48, 3.50, 0.25, 100),
    Note(40, 3.75, 0.25, 110),
]


def loop_length_beats(notes: list[Note]) -> float:
    """Loop length: the last note's end, rounded up to a whole beat (min 1 beat)."""
    end = max((n.start + n.duration for n in notes), default=0.0)
    return max(1.0, float(math.ceil(end)))


def play_loop(
    out,
    notes: list[Note],
    clock,
    *,
    channel: int = 0,
    quantum: int = 4,
    seconds: float | None = None,
) -> None:
    """Loop ``notes`` out ``out`` on ``clock``'s grid until ``seconds`` elapses (or forever).

    ``clock`` (see :mod:`osc_genai.realtime.clock`) supplies the beat position. Each loop pass is anchored to a
    whole bar (``quantum`` beats), so onsets land on the grid — under a ``LinkClock`` that is Ableton's
    grid, so the looped "human" input shares the duet's clock. The loop only advances while
    ``clock.playing`` is true and restarts cleanly from the next downbeat when the transport resumes.
    """
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    if not notes:
        return
    length = loop_length_beats(notes)
    start = (
        time.perf_counter()
    )  # wall-clock origin, used only for the ``seconds`` time limit
    pending_off: list[tuple[float, int]] = []  # min-heap of (off_beat, pitch)
    anchor: float | None = (
        None  # downbeat (in beats) where the current loop pass begins
    )

    def flush(beat: float) -> None:
        while pending_off and pending_off[0][0] <= beat:
            _, pitch = heapq.heappop(pending_off)
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))

    def silence() -> None:
        while pending_off:
            _, pitch = heapq.heappop(pending_off)
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))

    def expired() -> bool:
        return seconds is not None and (time.perf_counter() - start) >= seconds

    base, stop = 0.0, False
    try:
        while not stop:
            restart = False
            for note in notes:
                while True:  # wait for this note's onset on the shared grid
                    if expired():
                        stop = True
                        break
                    if (
                        not clock.playing
                    ):  # transport stopped: silence, restart from next downbeat
                        silence()
                        anchor = None
                        restart = True
                        time.sleep(0.01)
                        break
                    if anchor is None:  # pin the loop pass to the next whole bar
                        anchor = math.ceil(clock.beat / quantum) * quantum
                    beat = clock.beat
                    flush(beat)
                    target = anchor + base + note.start
                    if beat >= target:
                        break
                    sec_per_beat = 60.0 / clock.tempo
                    wake = (target - beat) * sec_per_beat
                    if pending_off:
                        wake = min(wake, (pending_off[0][0] - beat) * sec_per_beat)
                    time.sleep(
                        max(0.0, min(wake, 0.02))
                    )  # cap so we notice stop/tempo changes
                if stop or restart:
                    break
                out.send(
                    mido.Message(
                        "note_on",
                        note=note.pitch,
                        velocity=note.velocity,
                        channel=channel,
                    )
                )
                heapq.heappush(
                    pending_off,
                    (anchor + base + note.start + note.duration, note.pitch),
                )
            if restart:
                base = 0.0  # next pass re-anchors at the resumed transport's downbeat
                continue
            base += length
    finally:
        silence()


def main() -> None:
    args = build_parser(REGISTRY["fake-human"]).parse_args()

    if args.midi:
        notes = load_midi_file(args.midi)
    elif args.from_data:
        sequences = [s for s in load_midi_dir(args.from_data) if s]
        random.seed(args.seed)
        notes = random.choice(sequences) if sequences else _ACID
    else:
        notes = _ACID

    if not args.virtual and args.to_port not in mido.get_output_names():
        raise SystemExit(
            f"Port {args.to_port!r} not found. Start the duet first "
            "(uv run duet ...), or pass --virtual to create the port."
        )

    clock = make_clock(
        args.link,
        bpm=args.bpm,
        quantum=args.quantum,
        start_stop_sync=args.start_stop_sync,
    )
    tempo = "Ableton Link" if args.link else f"{args.bpm} BPM"
    with mido.open_output(args.to_port, virtual=args.virtual) as out:
        print(
            f"fake human: looping {len(notes)} notes into {args.to_port!r} ({tempo}). Ctrl-C to stop."
        )
        try:
            play_loop(out, notes, clock, quantum=args.quantum, seconds=args.seconds)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
