"""``audio-track``: capture audio, YIN-track it to notes, print them live, optionally save a ``.mid``.

A calibration / verification tool — the audio ingestion path *without* the model. Use it to confirm a
loopback device is wired up and to tune the thresholds (``--confidence``, ``--noise-floor``,
``--yin-threshold``) until single notes read out correctly, before running the duet with
``--audio-in``. **Monophonic only** (YIN tracks one fundamental per frame).
"""

from __future__ import annotations

import threading
import time

from osc_genai.audio.capture import AudioCapture
from osc_genai.audio.segment import NoteSegmenter
from osc_genai.audio.stream import PitchTracker
from osc_genai.audio.yin import Yin
from osc_genai.cli_spec import REGISTRY, build_parser
from osc_genai.data.midi import save_notes_midi
from osc_genai.realtime.clock import WallClock
from osc_genai.realtime.partner import HumanStream

_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _note_name(pitch: int) -> str:
    return f"{_NAMES[pitch % 12]}{pitch // 12 - 1}"


def main() -> None:
    args = build_parser(REGISTRY["audio-track"]).parse_args()

    clock = WallClock(args.bpm)
    human = HumanStream(lambda: clock.beat)

    def on_note_on(pitch: int, velocity: int) -> None:
        human.note_on(pitch, velocity)
        print(
            f"  {clock.beat:7.2f}b  ON  {_note_name(pitch):<4} ({pitch:>3})  vel {velocity}"
        )

    def on_note_off(pitch: int) -> None:
        human.note_off(pitch)

    segmenter = NoteSegmenter(
        note_on=on_note_on,
        note_off=on_note_off,
        confidence=args.confidence,
        noise_floor=args.noise_floor,
    )
    tracker = PitchTracker(
        Yin(args.samplerate, args.frame_size, args.yin_threshold),
        segmenter,
        frame_size=args.frame_size,
        hop=args.hop,
    )
    capture = AudioCapture(
        tracker.feed,
        device=args.device,
        samplerate=args.samplerate,
        blocksize=args.blocksize,
    ).start()

    print(
        f"audio-track: listening on {args.device!r} ({args.samplerate} Hz). Play single notes. "
        f"{f'Stops after {args.seconds:.1f}s.' if args.seconds else 'Ctrl-C to stop.'}"
    )
    stop = threading.Event()
    try:
        if args.seconds:
            stop.wait(args.seconds)
        else:
            while not stop.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        capture.stop()
        segmenter.flush()

    notes, count = human.window(0.0)
    print(f"tracked {count} note(s).")
    if args.save:
        save_notes_midi(notes, args.save)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
