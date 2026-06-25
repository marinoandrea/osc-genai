"""Record a duet session: capture the human and machine MIDI streams as paired Note sequences.

Listens on two ports — the human's input bus and the model's output — timestamps every note, and
saves the two parts on a shared timeline. This is the data engine for M4: genuine ``(human,
machine)`` pairs to train a *complementary* / anticipatory model on, instead of solo material.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("MIDO_BACKEND", "mido.backends.rtmidi")
import mido  # noqa: E402

from osc_genai.cli_spec import REGISTRY, build_parser  # noqa: E402
from osc_genai.core.note import Note  # noqa: E402


class StreamRecorder:
    """Assemble timestamped note on/off (seconds) into Notes; finalise to beats on a shared clock."""

    def __init__(self) -> None:
        self._active: dict[
            int, tuple[float, int]
        ] = {}  # pitch -> (start_sec, velocity)
        self._raw: list[
            tuple[int, float, float, int]
        ] = []  # (pitch, start_sec, dur_sec, velocity)

    def message(self, msg: mido.Message, now: float) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            self._active[msg.note] = (now, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            started = self._active.pop(msg.note, None)
            if started is not None:
                start, velocity = started
                self._raw.append((msg.note, start, max(0.0, now - start), velocity))

    def notes(self, t0: float, bpm: float) -> list[Note]:
        """Finalise to onset-ordered Notes in beats, with ``t0`` as the timeline origin."""
        sec_per_beat = 60.0 / bpm
        out = [
            Note(pitch, (start - t0) / sec_per_beat, dur / sec_per_beat, velocity)
            for (pitch, start, dur, velocity) in self._raw
        ]
        return sorted(out, key=lambda n: (n.start, n.pitch))


def record_session(
    human_in: mido.ports.BaseInput,
    machine_in: mido.ports.BaseInput,
    *,
    bpm: float = 130.0,
    seconds: float | None = None,
) -> tuple[list[Note], list[Note]]:
    """Record both streams until ``seconds`` elapses (or Ctrl-C); return ``(human, machine)``."""
    human, machine = StreamRecorder(), StreamRecorder()
    stop = threading.Event()
    t0 = time.perf_counter()

    def pump(port, recorder: StreamRecorder) -> None:
        for msg in port:
            recorder.message(msg, time.perf_counter())
            if stop.is_set():
                break

    threading.Thread(target=pump, args=(human_in, human), daemon=True).start()
    threading.Thread(target=pump, args=(machine_in, machine), daemon=True).start()

    end = (t0 + seconds) if seconds is not None else None
    try:
        while end is None or time.perf_counter() < end:
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return human.notes(t0, bpm), machine.notes(t0, bpm)


def save_session(
    human: list[Note], machine: list[Note], bpm: float, path: str | Path
) -> None:
    """Persist a paired session as JSON: ``{bpm, human: [...], machine: [...]}``."""
    data = {
        "bpm": bpm,
        "human": [list(n) for n in human],
        "machine": [list(n) for n in machine],
    }
    Path(path).write_text(json.dumps(data))


def load_session(path: str | Path) -> tuple[float, list[Note], list[Note]]:
    """Inverse of :func:`save_session`; returns ``(bpm, human, machine)``."""
    data = json.loads(Path(path).read_text())
    return (
        data["bpm"],
        [Note(*n) for n in data["human"]],
        [Note(*n) for n in data["machine"]],
    )


def main() -> None:
    args = build_parser(REGISTRY["record"]).parse_args()

    with (
        mido.open_input(args.human_port) as human_in,
        mido.open_input(args.machine_port) as machine_in,
    ):
        print(
            f"recording: human={args.human_port!r}, machine={args.machine_port!r}. "
            + (f"{args.seconds}s." if args.seconds else "Ctrl-C to stop.")
        )
        human, machine = record_session(
            human_in, machine_in, bpm=args.bpm, seconds=args.seconds
        )
    save_session(human, machine, args.bpm, args.out)
    print(f"recorded {len(human)} human + {len(machine)} machine note(s) -> {args.out}")


if __name__ == "__main__":
    main()
