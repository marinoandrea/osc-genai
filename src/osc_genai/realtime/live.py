"""Real-time MIDI duet over virtual MIDI ports.

The clip-based path (``main.py`` + ``ableton.py``) writes whole clips into Live via AbletonOSC —
great offline, but clip-granular and one-directional. A *live* duet needs the opposite: capture the
musician's notes as they play and answer with low latency. That runs over **virtual MIDI ports**
(CoreMIDI on macOS / ALSA on Linux, via ``python-rtmidi``), not OSC — AbletonOSC drives the Live
Object Model, not a real-time note stream.

Data path::

    Live MIDI track  --(IAC / virtual port)-->  this process   (capture human)
                                                      |
                                                 Responder      (rule-based now, model later)
                                                      |
    Live MIDI track  <--(IAC / virtual port)--  this process   (play machine)

The :class:`Responder` seam is deliberately model-agnostic: M1 ships an :class:`IntervalHarmonizer`
stand-in; later milestones drop a trained model in its place without touching the I/O.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# python-rtmidi is mido's real-time backend; select it before mido resolves a default.
os.environ.setdefault("MIDO_BACKEND", "mido.backends.rtmidi")
import mido  # noqa: E402  (import must follow the backend selection above)

from osc_genai.cli_spec import REGISTRY, build_parser  # noqa: E402

DEFAULT_IN_PORT = "osc-genai in"
DEFAULT_OUT_PORT = "osc-genai out"


# -- events -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteEvent:
    """A real-time note-on/off — the live counterpart to a clip ``Note``.

    ``on`` distinguishes note-on from note-off; ``pitch`` and ``velocity`` are 0-127. Unlike a clip
    :class:`~osc_genai.core.note.Note` there is no start/duration: the timing of a live event *is*
    when it happens, carried by the transport rather than the value.
    """

    pitch: int
    velocity: int
    on: bool

    @classmethod
    def from_message(cls, msg: mido.Message) -> NoteEvent | None:
        """Convert a mido message to a NoteEvent, or ``None`` if it isn't a note.

        A ``note_on`` with velocity 0 is the conventional note-off and is normalised here.
        """
        if msg.type == "note_on" and msg.velocity > 0:
            return cls(pitch=msg.note, velocity=msg.velocity, on=True)
        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            return cls(pitch=msg.note, velocity=0, on=False)
        return None

    def to_message(self, channel: int = 0) -> mido.Message:
        """Convert back to a mido message (note-off is emitted as an explicit ``note_off``)."""
        if self.on:
            return mido.Message(
                "note_on", note=self.pitch, velocity=self.velocity, channel=channel
            )
        return mido.Message("note_off", note=self.pitch, velocity=0, channel=channel)


# -- responder seam ---------------------------------------------------------------------------


@runtime_checkable
class Responder(Protocol):
    """Turns an incoming human note event into zero or more machine note events.

    Implementations may be stateless or stateful. The engine plays whatever is returned
    immediately; later milestones extend this with scheduling/anticipation for events placed in the
    future (see ``scheduler.py``, M3).
    """

    def respond(self, event: NoteEvent) -> list[NoteEvent]: ...


@dataclass
class IntervalHarmonizer:
    """M1 stand-in for the model: harmonise each note at fixed intervals.

    Stateless by construction — a note-off for pitch *p* maps to note-offs for *p + interval*, so no
    per-note bookkeeping is needed. Intervals landing outside 0-127 are dropped. This is just enough
    to prove the real-time round-trip; it is *parallel* harmony, not the complementary counterpoint
    the trained model will produce.
    """

    intervals: tuple[int, ...] = (4, 7)  # major third + perfect fifth above

    def respond(self, event: NoteEvent) -> list[NoteEvent]:
        out: list[NoteEvent] = []
        for interval in self.intervals:
            pitch = event.pitch + interval
            if 0 <= pitch <= 127:
                out.append(NoteEvent(pitch=pitch, velocity=event.velocity, on=event.on))
        return out


# -- engine -----------------------------------------------------------------------------------


@dataclass
class DuetEngine:
    """Drives a :class:`Responder` over real-time MIDI ports.

    :meth:`handle` is the pure core (events in -> events out) that tests exercise without any I/O;
    :meth:`run` wraps it in port plumbing and per-event latency accounting.
    """

    responder: Responder
    channel: int = 0
    # Internal processing latency (event received -> replies sent), in milliseconds. This excludes
    # CoreMIDI + Ableton's audio buffer, which only the live test in Ableton can measure.
    latencies_ms: list[float] = field(default_factory=list)

    def handle(self, event: NoteEvent) -> list[NoteEvent]:
        """Pure: map one incoming event to outgoing events (no I/O, no timing side effects)."""
        return self.responder.respond(event)

    def run(
        self,
        in_port: str = DEFAULT_IN_PORT,
        out_port: str = DEFAULT_OUT_PORT,
        virtual: bool = True,
    ) -> None:
        """Open ports and pump messages until interrupted (Ctrl-C).

        With ``virtual=True`` we create our own ports for the musician to route Live to/from;
        otherwise we connect to existing ports by name.
        """
        with (
            mido.open_input(in_port, virtual=virtual) as inp,
            mido.open_output(out_port, virtual=virtual) as out,
        ):
            print(
                f"Duet live: listening on {in_port!r}, responding on {out_port!r}. Ctrl-C to stop."
            )
            try:
                for msg in inp:
                    t0 = time.perf_counter()
                    event = NoteEvent.from_message(msg)
                    if event is None:
                        continue
                    for reply in self.handle(event):
                        out.send(reply.to_message(self.channel))
                    self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            except KeyboardInterrupt:
                pass
            finally:
                _all_notes_off(out, self.channel)
                print()
                self.report()

    def report(self) -> None:
        """Print a summary of internal processing latency."""
        n = len(self.latencies_ms)
        if not n:
            print("No note events processed.")
            return
        ordered = sorted(self.latencies_ms)
        p95 = ordered[min(n - 1, int(round(0.95 * (n - 1))))]
        print(
            f"Processed {n} note event(s). Processing latency (ms): "
            f"mean={statistics.fmean(ordered):.3f} median={statistics.median(ordered):.3f} "
            f"p95={p95:.3f} max={ordered[-1]:.3f}  "
            "(excludes CoreMIDI + Ableton buffer)"
        )


def _all_notes_off(out: mido.ports.BaseOutput, channel: int) -> None:
    """Panic: silence every pitch so nothing is left hanging when we stop."""
    for pitch in range(128):
        out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))


# -- CLI ----------------------------------------------------------------------------------------


def _parse_intervals(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def main() -> None:
    args = build_parser(REGISTRY["live"]).parse_args()

    if args.list_ports:
        print("Inputs: ", mido.get_input_names())
        print("Outputs:", mido.get_output_names())
        return

    engine = DuetEngine(
        responder=IntervalHarmonizer(intervals=_parse_intervals(args.intervals))
    )
    engine.run(
        in_port=args.in_port, out_port=args.out_port, virtual=not args.no_virtual
    )


if __name__ == "__main__":
    main()
