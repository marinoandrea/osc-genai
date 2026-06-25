"""Stream a trained model's output as a real-time MIDI note stream — *live* generation.

Unlike the clip path (write a whole clip, then play it), this emits notes one at a time over a
virtual MIDI port at a chosen tempo: the model keeps generating phrases and they play continuously,
seamlessly joined. Route a Live MIDI track's input at the port (``osc-genai out``) with an
instrument to hear it. This is the model-driven *output* half of the live duet; M3 adds the
human-conditioned *input* half (anticipation).

Timing uses a small real-time scheduler: a heap of pending note-offs is flushed as their moment
arrives while we wait for each next onset.
"""

from __future__ import annotations

import heapq
import os
import threading
import time

# python-rtmidi is mido's real-time backend; select it before mido resolves a default.
os.environ.setdefault("MIDO_BACKEND", "mido.backends.rtmidi")
import mido  # noqa: E402

from osc_genai.cli_spec import REGISTRY, build_parser  # noqa: E402
from osc_genai.core.event import DEFAULT_STEPS_PER_BEAT  # noqa: E402
from osc_genai.core.vocab import EventCodec  # noqa: E402
from osc_genai.model.checkpoint import load_model  # noqa: E402
from osc_genai.realtime.clock import make_clock  # noqa: E402

DEFAULT_OUT_PORT = "osc-genai out"


def stream(
    model,
    out: mido.ports.BaseOutput,
    clock,
    *,
    steps_per_beat: int = DEFAULT_STEPS_PER_BEAT,
    temperature: float = 0.95,
    channel: int = 0,
    seconds: float | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Generate forever (or for ``seconds``) and play each note at its moment on ``clock``'s grid.

    ``clock`` (see :mod:`osc_genai.realtime.clock`) supplies the beat position: a ``WallClock`` for free-running
    local timing, or a ``LinkClock`` riding Ableton's grid/tempo/transport. The cumulative onset is
    pinned to the clock's grid at start (and re-pinned whenever the transport resumes), so notes land
    on the beat; while ``clock.playing`` is false the stream goes silent.

    ``stop`` (a :class:`threading.Event`) lets a caller end the stream promptly — the loop and its
    inner waits treat it like the ``seconds`` deadline, so a UI can stop playback without Ctrl-C.
    """
    codec = EventCodec(model.vocab)
    start = (
        time.perf_counter()
    )  # wall-clock origin, used only for the ``seconds`` time limit
    onset_step = 0.0  # cumulative step position of the next onset, relative to the run's first event
    origin_step: float | None = (
        None  # grid step the run is pinned to; None until (re)anchored
    )
    pending_off: list[
        tuple[float, int, int]
    ] = []  # min-heap of (off_step, pitch, channel)

    def flush_due(playhead: float) -> None:
        while pending_off and pending_off[0][0] <= playhead:
            _, pitch, ch = heapq.heappop(pending_off)
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=ch))

    def silence() -> None:
        while pending_off:
            _, pitch, ch = heapq.heappop(pending_off)
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=ch))

    def expired() -> bool:
        if stop is not None and stop.is_set():
            return True
        return seconds is not None and (time.perf_counter() - start) >= seconds

    try:
        while not expired():
            events = model.generate(temperature=temperature, max_events=64)
            if not events:
                time.sleep(0.05)
                continue
            for fields in events:
                event = codec.decode(fields)
                onset_step += event.dt
                while True:  # wait for the onset on the shared grid, releasing finished notes meanwhile
                    if expired():
                        break
                    if (
                        not clock.playing
                    ):  # transport stopped: silence and re-anchor on resume
                        silence()
                        origin_step = None
                        time.sleep(0.01)
                        continue
                    if (
                        origin_step is None
                    ):  # pin this event to the nearest grid step "now"
                        origin_step = round(clock.beat * steps_per_beat) - onset_step
                    playhead = clock.beat * steps_per_beat
                    flush_due(playhead)
                    target = origin_step + onset_step
                    if playhead >= target:
                        break
                    sec_per_step = 60.0 / clock.tempo / steps_per_beat
                    wake = (target - playhead) * sec_per_step
                    if pending_off:
                        wake = min(wake, (pending_off[0][0] - playhead) * sec_per_step)
                    time.sleep(
                        max(0.0, min(wake, 0.02))
                    )  # cap so we notice stop/tempo changes
                if expired():
                    break
                out.send(
                    mido.Message(
                        "note_on",
                        note=event.pitch,
                        velocity=event.velocity,
                        channel=event.channel,
                    )
                )
                heapq.heappush(
                    pending_off,
                    (origin_step + onset_step + event.dur, event.pitch, event.channel),
                )
    finally:
        silence()
        for ch in range(16):  # all-notes-off on every channel
            out.send(mido.Message("control_change", control=123, value=0, channel=ch))


def main() -> None:
    args = build_parser(REGISTRY["play"]).parse_args()

    model = load_model(args.checkpoint, device=args.device)
    clock = make_clock(
        args.link,
        bpm=args.bpm,
        quantum=args.quantum,
        start_stop_sync=args.start_stop_sync,
    )
    tempo = "Ableton Link" if args.link else f"{args.bpm} BPM"
    with mido.open_output(args.out_port, virtual=not args.no_virtual) as out:
        print(
            f"streaming live generation to MIDI port {args.out_port!r} ({tempo}, "
            f"temp {args.temperature}). Ctrl-C to stop."
        )
        try:
            stream(
                model,
                out,
                clock,
                steps_per_beat=args.steps_per_beat,
                temperature=args.temperature,
                seconds=args.seconds,
            )
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
