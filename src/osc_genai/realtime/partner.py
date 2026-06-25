"""The duet's *partner* input — where the human's notes come from — behind one small interface.

The duet folds a partner's timed notes into the model's stream. Originally that partner was always a
MIDI port. This module factors the partner out so it can equally be a **real instrument captured as
audio** and pitch-tracked to notes — without the duet loop knowing the difference.

* :class:`HumanStream` — the shared, thread-safe record of partner notes on the beat clock. Both
  inputs feed it the same way (``note_on`` / ``note_off``), so generation and snapshot are agnostic
  to the source. (Lives here, not in :mod:`duet`, so the audio input can import it without a cycle.)
* :class:`MidiPartnerInput` — pump a mido input port into the stream (the original behavior, with the
  optional echo).
* :class:`AudioPartnerInput` — capture audio, run YIN + segmentation, and feed the stream. **Mono-
  phonic only**: a chord collapses to its strongest/lowest note.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import mido

from osc_genai.core.note import Note


class HumanStream:
    """Thread-safe record of the partner's notes as ``Note``s on the shared beat clock.

    Notes are timestamped at ``note_on`` against the shared origin, so they interleave with the
    model's line on one timeline. Duration is provisional until ``note_off`` (for conditioning the
    onset/pitch matter most). The source-agnostic ``note_on``/``note_off`` are what both the MIDI
    pump and the audio segmenter call; :meth:`on_message` adapts raw mido messages onto them.
    """

    def __init__(self, beat_now: Callable[[], float], channel: int = 0) -> None:
        self._beat_now = beat_now  # current position on the shared clock, in beats
        self._channel = channel
        self._notes: list[Note] = []  # onset-ordered, start in beats
        self._active: dict[int, int] = {}  # pitch -> index in _notes awaiting note_off
        self._lock = threading.Lock()
        self.note_count = 0

    def note_on(self, pitch: int, velocity: int) -> None:
        now_beats = self._beat_now()
        with self._lock:
            self._active[pitch] = len(self._notes)
            self._notes.append(
                Note(pitch, now_beats, 0.25, velocity, False, self._channel)
            )
            self.note_count += 1

    def note_off(self, pitch: int) -> None:
        now_beats = self._beat_now()
        with self._lock:
            idx = self._active.pop(pitch, None)
            if idx is not None:
                n = self._notes[idx]
                self._notes[idx] = n._replace(duration=max(0.0625, now_beats - n.start))

    def on_message(self, msg: mido.Message) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            self.note_on(msg.note, msg.velocity)
        elif msg.type in ("note_off", "note_on"):  # note_on vel 0 is a note_off
            self.note_off(msg.note)

    def window(self, since_beats: float) -> tuple[list[Note], int]:
        """Notes with onset >= ``since_beats`` (a trailing window), plus the running note count."""
        with self._lock:
            return [n for n in self._notes if n.start >= since_beats], self.note_count


class MidiPartnerInput:
    """Pump a mido input port into a :class:`HumanStream`, optionally echoing notes back out.

    ``echo_channel`` re-sends the human's incoming notes out the duet's *output* on that channel so
    they share the duet's clock (default 0); pass ``None`` to disable.
    """

    def __init__(self, inp: mido.ports.BaseInput, echo_channel: int | None = 0) -> None:
        self._inp = inp
        self._echo_channel = echo_channel

    def start(
        self,
        human: HumanStream,
        send: Callable[[mido.Message], None],
        stop: threading.Event,
    ) -> None:
        def pump() -> None:
            for msg in self._inp:  # blocking; ends when the port closes
                human.on_message(msg)
                if self._echo_channel is not None and msg.type in (
                    "note_on",
                    "note_off",
                ):
                    send(msg.copy(channel=self._echo_channel))
                if stop.is_set():
                    break

        threading.Thread(target=pump, daemon=True).start()

    def stop(self) -> None:
        pass  # the input port is owned/closed by the caller (a context manager)


class AudioPartnerInput:
    """Capture a monophonic instrument as audio, pitch-track it to notes, and feed the stream.

    Audio is captured from ``device`` (a virtual loopback), YIN-tracked over ``frame_size``/``hop``
    windows and segmented into notes. **YIN is monophonic** — only one fundamental per frame, so a
    bass chord is reduced to a single note. ``echo_channel`` optionally re-emits the *quantized*
    tracked notes out the duet port (for monitoring/recording in Live); ``None`` (the default) leaves
    the player to hear their own acoustic instrument.
    """

    def __init__(
        self,
        *,
        device: str,
        samplerate: int,
        blocksize: int,
        frame_size: int,
        hop: int,
        yin_threshold: float,
        confidence: float,
        noise_floor: float,
        smoothing: int = 3,
        release_frames: int = 3,
        echo_channel: int | None = None,
    ) -> None:
        self._device = device
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._frame_size = frame_size
        self._hop = hop
        self._yin_threshold = yin_threshold
        self._confidence = confidence
        self._noise_floor = noise_floor
        self._smoothing = smoothing
        self._release_frames = release_frames
        self._echo_channel = echo_channel
        self._capture = None

    def start(
        self,
        human: HumanStream,
        send: Callable[[mido.Message], None],
        stop: threading.Event,
    ) -> None:
        # Imported here so the MIDI path (and the package import) never needs audio deps.
        from osc_genai.audio.capture import AudioCapture
        from osc_genai.audio.segment import NoteSegmenter
        from osc_genai.audio.stream import PitchTracker
        from osc_genai.audio.yin import Yin

        def on_note_on(pitch: int, velocity: int) -> None:
            human.note_on(pitch, velocity)
            if self._echo_channel is not None:
                send(
                    mido.Message(
                        "note_on",
                        note=pitch,
                        velocity=velocity,
                        channel=self._echo_channel,
                    )
                )

        def on_note_off(pitch: int) -> None:
            human.note_off(pitch)
            if self._echo_channel is not None:
                send(
                    mido.Message(
                        "note_off", note=pitch, velocity=0, channel=self._echo_channel
                    )
                )

        segmenter = NoteSegmenter(
            note_on=on_note_on,
            note_off=on_note_off,
            confidence=self._confidence,
            noise_floor=self._noise_floor,
            smoothing=self._smoothing,
            release_frames=self._release_frames,
        )
        tracker = PitchTracker(
            Yin(self._samplerate, self._frame_size, self._yin_threshold),
            segmenter,
            frame_size=self._frame_size,
            hop=self._hop,
        )
        self._capture = AudioCapture(
            tracker.feed,
            device=self._device,
            samplerate=self._samplerate,
            blocksize=self._blocksize,
        ).start()

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
