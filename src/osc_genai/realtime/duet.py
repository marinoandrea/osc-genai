"""Real-time anticipatory duet: the model plays a complementary line *with* the human, live.

The human plays into a virtual MIDI input; their notes are folded — **time-aligned, on the shared
clock** — into the model's interleaved event stream as ``PARTNER`` events, and the model generates
its own ``SELF`` line ahead into a buffer and plays it out a virtual MIDI output. This is the duet
the ``source``-field model (trained by ``osc-genai-train-paired``) was built for: unlike the older
pitch-only priming, the model conditions on *what the human plays and when*.

How it stays live and adaptive:

* The authoritative history is the onset-ordered merge of the human's observed ``PARTNER`` notes and
  the ``SELF`` notes we've already committed. Each generation cycle re-primes a fresh GRU state on a
  trailing window of that history (:func:`pairs.interleave` does the merge/tagging), then rolls the
  streaming API forward to fill a lookahead window — ``SELF`` events are scheduled, predicted
  ``PARTNER`` events just advance the state (the model anticipating the human).
* Scheduling/commit-horizon protection and reconciliation reuse :class:`AnticipatoryBuffer`: notes
  within a short horizon ahead of the playhead are locked; when the human moves, the revisable tail
  is dropped and regenerated from the corrected history.
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
import threading
import time
from collections.abc import Callable

# python-rtmidi is mido's real-time backend; select it before mido resolves a default.
os.environ.setdefault("MIDO_BACKEND", "mido.backends.rtmidi")
import mido  # noqa: E402

from osc_genai.core.note import Note  # noqa: E402
from osc_genai.realtime.clock import make_clock  # noqa: E402
from osc_genai.data.pairs import interleave  # noqa: E402
from osc_genai.core.event import DEFAULT_STEPS_PER_BEAT  # noqa: E402
from osc_genai.realtime.scheduler import AnticipatoryBuffer, Scheduled  # noqa: E402
from osc_genai.data.snapshot import save_snapshot  # noqa: E402
from osc_genai.osc.listen import OSCTrigger  # noqa: E402
from osc_genai.model.checkpoint import load_model  # noqa: E402
from osc_genai.core.vocab import EventCodec  # noqa: E402

DEFAULT_IN_PORT = "osc-genai in"
DEFAULT_OUT_PORT = "osc-genai out"


class HumanStream:
    """Thread-safe record of the human's notes as ``PARTNER`` ``Note``s on the shared beat clock.

    Notes are timestamped at ``note_on`` against a shared origin, so they interleave with the model's
    line on the same timeline. Duration is provisional (finalised on ``note_off``) — for conditioning
    the onset/pitch matter most.
    """

    def __init__(self, beat_now: Callable[[], float], channel: int = 0) -> None:
        self._beat_now = beat_now  # current position on the shared clock, in beats
        self._channel = channel
        self._notes: list[Note] = []  # onset-ordered, start in beats
        self._active: dict[int, int] = {}  # pitch -> index in _notes awaiting note_off
        self._lock = threading.Lock()
        self.note_count = 0

    def on_message(self, msg: "mido.Message") -> None:
        now_beats = self._beat_now()
        with self._lock:
            if msg.type == "note_on" and msg.velocity > 0:
                self._active[msg.note] = len(self._notes)
                self._notes.append(Note(msg.note, now_beats, 0.25, msg.velocity, False, self._channel))
                self.note_count += 1
            elif msg.type in ("note_off", "note_on"):  # note_on vel 0 is a note_off
                idx = self._active.pop(msg.note, None)
                if idx is not None:
                    n = self._notes[idx]
                    self._notes[idx] = n._replace(duration=max(0.0625, now_beats - n.start))

    def window(self, since_beats: float) -> tuple[list[Note], int]:
        """Notes with onset >= ``since_beats`` (a trailing window), plus the running note count."""
        with self._lock:
            return [n for n in self._notes if n.start >= since_beats], self.note_count


def duet(
    model,
    inp: "mido.ports.BaseInput",
    out: "mido.ports.BaseOutput",
    clock,
    *,
    steps_per_beat: int = DEFAULT_STEPS_PER_BEAT,
    temperature: float = 0.95,
    chunk_events: int = 8,
    lookahead_steps: float = 8.0,
    commit_horizon: float = 2.0,
    window_beats: float = 16.0,
    pitch_bias: dict[int, float] | None = None,
    human_channel: int = 0,
    out_channel: int | None = 9,
    echo_channel: int | None = 0,
    regular_pitches: tuple[int, ...] = (36, 38),
    regular_temperature: float = 0.15,
    snap_steps: int = 2,
    seconds: float | None = None,
    beats_per_bar: float = 4.0,
    snapshot_root: str | None = None,
    snapshot_bars: int = 4,
    snapshot_human_inst: str = "Bass",
    snapshot_machine_inst: str = "Drums",
    snapshot_artist: str = "personal",
    snapshot_key: str = "s",
    snapshot_osc_port: int | None = None,
    snapshot_osc_addr: str = "/snapshot",
) -> None:
    """Play an anticipatory duet: fold the human's timed notes into the model's interleaved stream and
    generate the complementary ``SELF`` line ahead, reconciling when the human moves.

    ``clock`` is the shared beat clock (see :mod:`osc_genai.realtime.clock`): a ``WallClock`` for free-running
    local timing, or a ``LinkClock`` to ride Ableton Live's grid/tempo and start/stop with its
    transport. The playhead is read as ``clock.beat * steps_per_beat`` so generated onsets land on the
    clock's grid; when ``clock.playing`` is false the model goes silent until the transport resumes.

    ``out_channel`` pins every played note to one MIDI channel (default 9, the GM drum lane) so a
    single Ableton track sees a steady stream; pass ``None`` to keep the model's predicted channel.
    ``echo_channel`` re-sends the human's incoming notes out the *same* port on that channel (default
    0), so the bass you hear shares the duet's clock with the drums — without it, a separately-clocked
    bass source drifts in phase and sounds syncopated. Pass ``None`` to disable the echo.

    ``regular_pitches`` (default kick=36/snare=38) are the foundation lanes kept *deliberately
    regular*: their timing is sampled at ``regular_temperature`` (near-greedy, so they don't wander)
    and their onsets are snapped to a ``snap_steps`` grid (2 = 8th notes, 4 = quarters; 0 = off).
    Everything else (hats, percussion, effects) stays sampled at ``temperature`` for variety.
    """
    import torch

    codec = EventCodec(model.vocab)
    source_field = len(model.vocab.field_sizes) - 1
    regular = set(regular_pitches)
    bias = None
    if pitch_bias:
        tensor = torch.zeros(model.vocab.pitch_vocab)
        for pitch, value in pitch_bias.items():
            tensor[pitch] += value
        bias = {0: tensor}

    start = time.perf_counter()  # wall-clock origin, used only for the ``seconds`` time limit
    human = HumanStream(lambda: clock.beat, channel=human_channel)
    stop = threading.Event()
    out_lock = threading.Lock()  # the pump thread (echo) and firing loop both write ``out``
    gen_lock = threading.Lock()  # the generation thread and firing loop share buffer/committed_self

    def send(msg: "mido.Message") -> None:
        with out_lock:
            out.send(msg)

    def pump() -> None:
        for msg in inp:  # blocking; ends when the port closes
            human.on_message(msg)
            if echo_channel is not None and msg.type in ("note_on", "note_off"):
                send(msg.copy(channel=echo_channel))  # let the player hear the bass on the duet's clock
            if stop.is_set():
                break

    threading.Thread(target=pump, daemon=True).start()

    buffer = AnticipatoryBuffer(commit_horizon=commit_horizon)
    committed_self: list[Note] = []  # SELF notes already played, re-fed as history (start in beats)
    pending_off: list[tuple[float, int, int]] = []  # min-heap of (off_step, pitch, channel)
    was_playing = False

    def release_all() -> None:
        """Silence everything currently sounding (model notes + a belt-and-braces all-notes-off)."""
        while pending_off:
            _, pitch, ch = heapq.heappop(pending_off)
            send(mido.Message("note_off", note=pitch, velocity=0, channel=ch))
        for ch in range(16):
            send(mido.Message("control_change", control=123, value=0, channel=ch))

    # Generation (model inference, ~ms) runs on a background thread so it never stalls note-firing;
    # the two share ``buffer``/``committed_self`` under ``gen_lock``, held only for cheap snapshot/add
    # (never across model compute). This is what keeps the timing tight.
    def snapshot(playhead: float):
        """Trailing-window history (partner + committed + buffered SELF) + the foundation steps taken."""
        since = playhead / steps_per_beat - window_beats
        partner, _ = human.window(since)
        with gen_lock:
            mine = [n for n in committed_self if n.start >= since]
            upcoming = buffer.upcoming()
        mine += [Note(s.pitch, s.onset / steps_per_beat, s.dur / steps_per_beat, s.velocity, False, s.channel)
                 for s in upcoming if s.onset / steps_per_beat >= since]
        taken = {round(s.onset) for s in upcoming if s.pitch in regular}
        return partner, mine, taken

    def prime(partner, mine, playhead):
        """Build a primed state from a history snapshot; onsets shifted to a local origin (small dt),
        the state's running onset seeded to that origin's absolute step (phase aligned to the grid)."""
        if not (partner or mine):
            return model.fresh_state(onset0=round(playhead))
        origin = min(n.start for n in (partner + mine))
        history = interleave(
            [n._replace(start=n.start - origin) for n in partner],
            [n._replace(start=n.start - origin) for n in mine],
            steps_per_beat,
        )
        state = model.fresh_state(onset0=round(origin * steps_per_beat))
        for fields in codec.encode_sequence(history, add_eos=False):
            state = model.observe(state, fields)
        return state

    def generate_from(playhead: float) -> int:
        """Roll forward from the primed frontier and schedule SELF notes; scheduled position equals
        the model's own running onset, so it agrees with the phase feature."""
        partner, mine, taken = snapshot(playhead)
        state = prime(partner, mine, playhead)  # model compute — no lock held
        scheduled: list[Scheduled] = []
        for _ in range(max(chunk_events, 4 * int(lookahead_steps) + 8)):  # one prime fills the lookahead
            fields, state = model.sample_next(
                state, temperature, bias=bias,
                regular_pitches=regular, regular_temperature=regular_temperature,
            )
            if fields[0] == model.vocab.eos_pitch:
                break
            onset = state[1]  # absolute grid step of the just-sampled event (phase-consistent)
            if onset > playhead + lookahead_steps:
                break
            if onset <= playhead:  # already past (frontier behind the playhead); skip, keep rolling
                continue
            if fields[source_field] != 0:  # SELF (0 == PARTNER) — play it
                event = codec.decode(fields)
                if snap_steps and event.pitch in regular:  # force the foundation onto a clean grid
                    onset = round(onset / snap_steps) * snap_steps
                    if onset <= playhead or onset in taken:  # don't double-hit a snapped step
                        continue
                    taken.add(onset)
                channel = out_channel if out_channel is not None else event.channel
                scheduled.append(Scheduled(float(onset), event.pitch, event.velocity, float(event.dur), channel))
        with gen_lock:
            buffer.add(scheduled)
        return len(scheduled)

    def gen_loop() -> None:
        """Background: reconcile when the human moves and keep the lookahead window filled."""
        last_fingerprint = -1
        while not stop.is_set():
            if not clock.playing:
                time.sleep(0.01)
                continue
            playhead = clock.beat * steps_per_beat
            _, fingerprint = human.window(0.0)
            if fingerprint != last_fingerprint:  # the human played something new
                last_fingerprint = fingerprint
                with gen_lock:
                    buffer.reconcile(playhead)
                generate_from(playhead)
            else:
                with gen_lock:
                    frontier = buffer.last_onset(default=playhead)
                if frontier < playhead + lookahead_steps:
                    generate_from(playhead)
            time.sleep(0.004)

    def expired() -> bool:
        return seconds is not None and (time.perf_counter() - start) >= seconds

    # Snapshot: on a keypress or an OSC bang, grab the last N bars of both parts and save them as
    # a training pair (see osc_genai.data.snapshot). Triggers run on their own threads so they
    # never touch the firing loop's timing; state is copied under the existing locks (cheap, no
    # model compute). snapshot_lock serialises concurrent keyboard/OSC triggers.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot_count = [0]
    snapshot_lock = threading.Lock()
    osc_trigger: OSCTrigger | None = None

    def do_snapshot() -> None:
        end_beat = clock.beat
        human_notes, _ = human.window(0.0)
        with gen_lock:
            machine_notes = list(committed_self)
        with snapshot_lock:
            song_id = f"{stamp}-{snapshot_count[0]:03d}"
            result = save_snapshot(
                human_notes, machine_notes, snapshot_root,
                end_beat=end_beat, bars=snapshot_bars, beats_per_bar=beats_per_bar,
                human_inst=snapshot_human_inst, machine_inst=snapshot_machine_inst,
                artist=snapshot_artist, song_id=song_id,
            )
            if result is None:
                print(f"snapshot: not enough played yet (need {snapshot_bars} bars).")
                return
            snapshot_count[0] += 1
        human_path, machine_path = result
        print(f"snapshot: saved last {snapshot_bars} bars -> {human_path} + {machine_path}")

    def snapshot_loop() -> None:
        for line in sys.stdin:  # blocking; ends when stdin closes
            if stop.is_set():
                break
            if line.strip()[:1] == snapshot_key:
                do_snapshot()

    if snapshot_root is not None:
        threading.Thread(target=snapshot_loop, daemon=True).start()
        if snapshot_osc_port is not None and snapshot_osc_port > 0:
            osc_trigger = OSCTrigger({snapshot_osc_addr: do_snapshot}, port=snapshot_osc_port).start()

    threading.Thread(target=gen_loop, daemon=True).start()

    try:
        while not expired():  # tight firing loop — only cheap MIDI I/O, never blocked by the model
            if not clock.playing:  # transport stopped: silence and drop the plan until it resumes
                if was_playing:
                    with gen_lock:
                        release_all()
                        buffer.clear()
                        committed_self.clear()
                    was_playing = False
                time.sleep(0.01)
                continue
            was_playing = True
            playhead = clock.beat * steps_per_beat  # position on the shared grid, in steps

            while pending_off and pending_off[0][0] <= playhead:  # release finished notes
                _, pitch, ch = heapq.heappop(pending_off)
                send(mido.Message("note_off", note=pitch, velocity=0, channel=ch))

            with gen_lock:
                due = buffer.pop_due(playhead)
            for sched in due:  # fire notes whose moment has arrived
                send(mido.Message("note_on", note=sched.pitch, velocity=sched.velocity, channel=sched.channel))
                heapq.heappush(pending_off, (sched.onset + sched.dur, sched.pitch, sched.channel))
                with gen_lock:
                    committed_self.append(  # this SELF note is now history for future conditioning
                        Note(sched.pitch, sched.onset / steps_per_beat, sched.dur / steps_per_beat,
                             sched.velocity, False, sched.channel)
                    )

            time.sleep(0.0007)
    finally:
        stop.set()
        if osc_trigger is not None:
            osc_trigger.close()
        release_all()


def _open_input(name: str):
    """Connect to an existing input port by name (e.g. an IAC bus), else create it virtually."""
    return mido.open_input(name, virtual=name not in mido.get_input_names())


def _open_output(name: str):
    """Connect to an existing output port by name, else create it virtually."""
    return mido.open_output(name, virtual=name not in mido.get_output_names())


def _parse_pitch_bias(text: str | None) -> dict[int, float] | None:
    """Parse a ``"36:2,47:-3"`` pitch-bias spec into ``{pitch: bias}``."""
    if not text:
        return None
    out: dict[int, float] = {}
    for part in text.split(","):
        pitch, value = part.split(":")
        out[int(pitch)] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time anticipatory duet (model plays with you).")
    parser.add_argument("--checkpoint", required=True, help="trained duet model (.pt)")
    parser.add_argument("--in-port", default=DEFAULT_IN_PORT, help="port you play into")
    parser.add_argument("--out-port", default=DEFAULT_OUT_PORT, help="port the model plays out")
    parser.add_argument("--bpm", type=float, default=130.0, help="tempo; with --link this is only the fallback/seed")
    parser.add_argument("--link", action="store_true", help="ride Ableton Link's grid/tempo/transport")
    parser.add_argument("--quantum", type=int, default=4, help="Link bar length in beats (phase alignment)")
    parser.add_argument(
        "--start-stop-sync", action=argparse.BooleanOptionalAction, default=True,
        help="with --link, only play while Ableton's transport runs",
    )
    parser.add_argument("--steps-per-beat", type=int, default=DEFAULT_STEPS_PER_BEAT)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--chunk-events", type=int, default=8, help="events generated per chunk")
    parser.add_argument("--lookahead", type=float, default=8.0, help="grid steps kept generated ahead")
    parser.add_argument("--commit-horizon", type=float, default=2.0, help="steps ahead locked from revision")
    parser.add_argument("--window-beats", type=float, default=16.0, help="trailing history fed back as context")
    parser.add_argument("--out-channel", type=int, default=9, help="pin drums to this MIDI channel (-1 = model's own)")
    parser.add_argument("--echo-channel", type=int, default=0, help="echo the human's bass out on this channel so it shares the duet clock (-1 = off)")
    parser.add_argument("--pitch-bias", default=None, help='per-pitch logit bias, e.g. "47:-3,48:-3,38:1"')
    parser.add_argument("--regular-pitches", default="36,38", help="foundation lanes kept regular (default kick,snare)")
    parser.add_argument("--regular-temperature", type=float, default=0.15, help="near-greedy timing temp for the foundation")
    parser.add_argument("--snap-steps", type=int, default=2, help="snap foundation onsets to this grid (2=8th, 4=1/4, 0=off)")
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    parser.add_argument("--snapshot-dir", default="data/MIDI", help="dataset root snapshots are saved into")
    parser.add_argument("--no-snapshots", action="store_true", help="disable the keypress snapshot trigger")
    parser.add_argument("--snapshot-bars", type=int, default=4, help="bars per snapshot (match the training chunk size)")
    parser.add_argument("--snapshot-human-inst", default="Bass", help="instrument folder for your part")
    parser.add_argument("--snapshot-machine-inst", default="Drums", help="instrument folder for the model's part")
    parser.add_argument("--snapshot-artist", default="personal", help="artist folder snapshots land under")
    parser.add_argument("--snapshot-key", default="s", help="key to press (then Enter) to save a snapshot")
    parser.add_argument("--snapshot-osc-port", type=int, default=11002, help="UDP port to listen on for an OSC snapshot trigger (<=0 = off)")
    parser.add_argument("--snapshot-osc-addr", default="/snapshot", help="OSC address that triggers a snapshot")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    clock = make_clock(args.link, bpm=args.bpm, quantum=args.quantum, start_stop_sync=args.start_stop_sync)
    tempo = "Ableton Link" if args.link else f"{args.bpm} BPM"
    with _open_input(args.in_port) as inp, _open_output(args.out_port) as out:
        snap = "off" if args.no_snapshots else (
            f"press {args.snapshot_key!r}+Enter"
            + (f" or send OSC {args.snapshot_osc_addr} on UDP {args.snapshot_osc_port}"
               if args.snapshot_osc_port > 0 else "")
            + f" to save the last {args.snapshot_bars} bars to {args.snapshot_dir}"
        )
        print(
            f"duet: listening to YOU on {args.in_port!r}, responding on {args.out_port!r} "
            f"({tempo}). Play something. Snapshots: {snap}. Ctrl-C to stop."
        )
        try:
            duet(
                model,
                inp,
                out,
                clock,
                steps_per_beat=args.steps_per_beat,
                temperature=args.temperature,
                chunk_events=args.chunk_events,
                lookahead_steps=args.lookahead,
                commit_horizon=args.commit_horizon,
                window_beats=args.window_beats,
                pitch_bias=_parse_pitch_bias(args.pitch_bias),
                out_channel=None if args.out_channel < 0 else args.out_channel,
                echo_channel=None if args.echo_channel < 0 else args.echo_channel,
                regular_pitches=tuple(int(p) for p in args.regular_pitches.split(",") if p.strip()),
                regular_temperature=args.regular_temperature,
                snap_steps=args.snap_steps,
                seconds=args.seconds,
                beats_per_bar=args.quantum,
                snapshot_root=None if args.no_snapshots else args.snapshot_dir,
                snapshot_bars=args.snapshot_bars,
                snapshot_human_inst=args.snapshot_human_inst,
                snapshot_machine_inst=args.snapshot_machine_inst,
                snapshot_artist=args.snapshot_artist,
                snapshot_key=args.snapshot_key,
                snapshot_osc_port=None if args.no_snapshots else args.snapshot_osc_port,
                snapshot_osc_addr=args.snapshot_osc_addr,
            )
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
