"""Encoding between musical Events and the integer field indices the model predicts.

The model works in a fixed vocabulary per field: pitch as one-of-128 plus an EOS class, ``dt``/
``dur`` clamped to a maximum step count, velocity binned, **channel** (0-15), and **source** (the
duet role — PARTNER/SELF). One head per field; end-of-sequence is an extra pitch class
(``eos_pitch``), its other fields ignored. The channel field lets generation address multiple
instruments at once; the source field lets one interleaved stream carry both lines of a duet. This
module is the only place lossy clamping/binning lives, so the rest of the stack stays in musical
units.
"""

from __future__ import annotations

from dataclasses import dataclass

from osc_genai.core.event import Event

MIDI_RANGE = 128  # MIDI pitch and velocity are both 0-127

# A note as model field indices: (pitch, dt, dur, velocity, channel, source).
Fields = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class VocabConfig:
    max_dt: int = 32  # clamp onset gap to this many grid steps
    max_dur: int = 32  # clamp note length to this many grid steps
    velocity_bins: int = 16
    num_channels: int = 16  # MIDI channels 0-15
    num_sources: int = 2  # duet roles: PARTNER, SELF
    use_phase: bool = False  # feed bar-relative grid position as an input-only conditioning feature
    steps_per_bar: int = 16  # grid steps per bar (4 beats * 4 steps), the period of the phase feature

    @property
    def eos_pitch(self) -> int:
        return MIDI_RANGE  # the index just past the real pitches

    @property
    def pitch_vocab(self) -> int:
        return MIDI_RANGE + 1  # 128 pitches + EOS

    @property
    def dt_vocab(self) -> int:
        return self.max_dt + 1  # 0 .. max_dt

    @property
    def dur_vocab(self) -> int:
        return self.max_dur  # 1 .. max_dur, stored as index 0 .. max_dur-1

    @property
    def velocity_vocab(self) -> int:
        return self.velocity_bins

    @property
    def channel_vocab(self) -> int:
        return self.num_channels

    @property
    def source_vocab(self) -> int:
        return self.num_sources

    @property
    def field_sizes(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.pitch_vocab,
            self.dt_vocab,
            self.dur_vocab,
            self.velocity_vocab,
            self.channel_vocab,
            self.source_vocab,
        )


class EventCodec:
    """Encode Events <-> per-field index tuples for a given :class:`VocabConfig`."""

    def __init__(self, config: VocabConfig | None = None) -> None:
        self.config = config or VocabConfig()

    # -- single event ---------------------------------------------------------------------------
    def encode(self, event: Event) -> Fields:
        cfg = self.config
        pitch = _clamp(event.pitch, 0, MIDI_RANGE - 1)
        dt = _clamp(event.dt, 0, cfg.max_dt)
        dur = _clamp(event.dur, 1, cfg.max_dur)
        channel = _clamp(event.channel, 0, cfg.num_channels - 1)
        source = _clamp(event.source, 0, cfg.num_sources - 1)
        return (pitch, dt, dur - 1, self._encode_velocity(event.velocity), channel, source)

    def decode(self, fields: Fields) -> Event:
        pitch, dt, dur_idx, vel_idx, channel, source = fields
        return Event(
            pitch=pitch,
            dt=dt,
            dur=dur_idx + 1,
            velocity=self._decode_velocity(vel_idx),
            channel=channel,
            source=source,
        )

    # -- velocity binning -----------------------------------------------------------------------
    def _encode_velocity(self, velocity: int) -> int:
        bins = self.config.velocity_bins
        v = _clamp(velocity, 0, MIDI_RANGE - 1)
        return min(bins - 1, v * bins // MIDI_RANGE)

    def _decode_velocity(self, index: int) -> int:
        bins = self.config.velocity_bins
        return min(MIDI_RANGE - 1, int((index + 0.5) * MIDI_RANGE / bins))  # bin centre

    # -- end of sequence ------------------------------------------------------------------------
    @property
    def eos(self) -> Fields:
        return (self.config.eos_pitch, 0, 0, 0, 0, 0)

    def is_eos(self, fields: Fields) -> bool:
        return fields[0] == self.config.eos_pitch

    # -- sequences ------------------------------------------------------------------------------
    def encode_sequence(self, events: list[Event], add_eos: bool = True) -> list[Fields]:
        encoded = [self.encode(e) for e in events]
        if add_eos:
            encoded.append(self.eos)
        return encoded

    def decode_sequence(self, seq: list[Fields]) -> list[Event]:
        events: list[Event] = []
        for fields in seq:
            if self.is_eos(fields):
                break
            events.append(self.decode(fields))
        return events


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
