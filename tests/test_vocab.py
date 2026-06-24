"""Tests for Event <-> model-field-index encoding."""

from __future__ import annotations

from osc_genai.core.event import Event
from osc_genai.core.vocab import EventCodec, VocabConfig


def test_roundtrip_with_bin_centre_velocity_is_exact():
    codec = EventCodec(VocabConfig(velocity_bins=16))
    ev = Event(pitch=60, dt=4, dur=2, velocity=100)  # 100 is a 16-bin centre
    assert codec.decode(codec.encode(ev)) == ev


def test_pitch_dt_dur_are_exact():
    codec = EventCodec()
    out = codec.decode(codec.encode(Event(72, 7, 5, 64)))
    assert (out.pitch, out.dt, out.dur) == (72, 7, 5)


def test_velocity_binning_within_one_bin():
    codec = EventCodec(VocabConfig(velocity_bins=16))  # bin width 8
    for v in range(128):
        out = codec.decode(codec.encode(Event(60, 1, 1, v)))
        assert abs(out.velocity - v) <= 8


def test_clamping():
    codec = EventCodec(VocabConfig(max_dt=32, max_dur=32, velocity_bins=16, num_channels=16))
    pitch, dt, dur_idx, vel, channel, source = codec.encode(
        Event(pitch=200, dt=999, dur=999, velocity=300, channel=99, source=5)
    )
    assert pitch == 127
    assert dt == 32
    assert dur_idx == 31  # dur clamped to 32 -> index 31
    assert vel == 15
    assert channel == 15  # clamped to num_channels - 1
    assert source == 1  # clamped to num_sources - 1


def test_dur_floor_is_one_step():
    codec = EventCodec()
    assert codec.decode(codec.encode(Event(60, 0, 0, 100))).dur == 1


def test_eos_and_sequence_roundtrip():
    codec = EventCodec()
    events = [Event(60, 0, 4, 100), Event(64, 4, 4, 100)]
    seq = codec.encode_sequence(events)
    assert codec.is_eos(seq[-1])
    assert codec.decode_sequence(seq) == events


def test_field_sizes():
    codec = EventCodec(VocabConfig(max_dt=32, max_dur=32, velocity_bins=16, num_channels=16))
    assert codec.config.field_sizes == (129, 33, 32, 16, 16, 2)


def test_channel_roundtrips():
    codec = EventCodec(VocabConfig(velocity_bins=16))
    ev = Event(pitch=60, dt=2, dur=2, velocity=100, channel=9)
    assert codec.decode(codec.encode(ev)).channel == 9


def test_source_roundtrips():
    codec = EventCodec(VocabConfig(velocity_bins=16))
    ev = Event(pitch=60, dt=2, dur=2, velocity=100, channel=9, source=1)
    assert codec.decode(codec.encode(ev)).source == 1
