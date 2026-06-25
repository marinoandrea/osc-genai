"""``audio-track``: capture audio, YIN-track it to notes, print them live, optionally save a ``.mid``.

A calibration / verification tool — the audio ingestion path *without* the model. Use it to confirm a
loopback device is wired up and to tune the thresholds (``--confidence``, ``--noise-floor``,
``--yin-threshold``) until single notes read out correctly, before running the duet with
``--audio-in``. **Monophonic only** (YIN tracks one fundamental per frame).
"""

from __future__ import annotations

import argparse
import threading
import time

from osc_genai.audio.capture import AudioCapture, DEFAULT_BLOCKSIZE, DEFAULT_DEVICE, DEFAULT_SAMPLERATE
from osc_genai.audio.segment import NoteSegmenter
from osc_genai.audio.stream import PitchTracker
from osc_genai.audio.yin import Yin
from osc_genai.data.midi import save_notes_midi
from osc_genai.realtime.clock import WallClock
from osc_genai.realtime.partner import HumanStream

_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _note_name(pitch: int) -> str:
    return f"{_NAMES[pitch % 12]}{pitch // 12 - 1}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pitch-track a monophonic audio input to MIDI notes (calibration/verify tool)."
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="capture device name (a loopback)")
    parser.add_argument("--samplerate", type=int, default=DEFAULT_SAMPLERATE)
    parser.add_argument("--blocksize", type=int, default=DEFAULT_BLOCKSIZE, help="capture block size")
    parser.add_argument("--frame-size", type=int, default=4096, help="YIN analysis window in samples (4096 covers a bass's low E ~41Hz; smaller = lower latency but higher pitch floor)")
    parser.add_argument("--hop", type=int, default=512, help="frames advance by this many samples")
    parser.add_argument("--yin-threshold", type=float, default=0.15, help="YIN aperiodicity threshold")
    parser.add_argument("--confidence", type=float, default=0.5, help="min YIN probability to voice")
    parser.add_argument("--noise-floor", type=float, default=0.01, help="min RMS to voice (gate silence)")
    parser.add_argument("--bpm", type=float, default=120.0, help="tempo used to timestamp saved notes")
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds (else Ctrl-C)")
    parser.add_argument("--save", default=None, help="write the tracked notes to this .mid path")
    args = parser.parse_args()

    clock = WallClock(args.bpm)
    human = HumanStream(lambda: clock.beat)

    def on_note_on(pitch: int, velocity: int) -> None:
        human.note_on(pitch, velocity)
        print(f"  {clock.beat:7.2f}b  ON  {_note_name(pitch):<4} ({pitch:>3})  vel {velocity}")

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
        tracker.feed, device=args.device, samplerate=args.samplerate, blocksize=args.blocksize
    ).start()

    print(f"audio-track: listening on {args.device!r} ({args.samplerate} Hz). Play single notes. "
          f"{'Stops after %.1fs.' % args.seconds if args.seconds else 'Ctrl-C to stop.'}")
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
