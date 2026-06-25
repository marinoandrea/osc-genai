"""End-to-end of the audio path's pure pipeline: audio blocks -> YIN -> segmenter -> HumanStream.

The live ``AudioPartnerInput`` only adds the capture device on top of this pipeline; capture needs
hardware, so here we drive the same PitchTracker/NoteSegmenter/HumanStream chain with synthetic
blocks (no PortAudio). This also exercises the generalized ``HumanStream.note_on/note_off``.
"""

from __future__ import annotations

import numpy as np

from osc_genai.audio.segment import NoteSegmenter
from osc_genai.audio.stream import PitchTracker
from osc_genai.audio.yin import Yin
from osc_genai.realtime.partner import AudioPartnerInput, HumanStream

SR = 44100
FRAME = 2048
HOP = 512
BLOCK = 512


class FakeClock:
    def __init__(self) -> None:
        self.beat = 0.0


def _midi_to_hz(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def _feed_tone(tracker: PitchTracker, clk: FakeClock, pitch: int, seconds: float) -> None:
    hz = _midi_to_hz(pitch)
    for i in range(int(seconds * SR / BLOCK)):
        clk.beat += BLOCK / SR * 2.0  # advance beats (120 BPM) so onsets carry a sensible time
        t = (np.arange(BLOCK) + i * BLOCK) / SR  # continuous phase across blocks
        tracker.feed(0.5 * np.sin(2 * np.pi * hz * t))


def _feed_silence(tracker: PitchTracker, clk: FakeClock, seconds: float) -> None:
    for _ in range(int(seconds * SR / BLOCK)):
        clk.beat += BLOCK / SR * 2.0
        tracker.feed(np.zeros(BLOCK))


def _pipeline(human: HumanStream) -> tuple[PitchTracker, NoteSegmenter]:
    seg = NoteSegmenter(
        note_on=human.note_on, note_off=human.note_off, confidence=0.4, noise_floor=0.001
    )
    tracker = PitchTracker(Yin(SR, FRAME, 0.15), seg, frame_size=FRAME, hop=HOP)
    return tracker, seg


def test_sustained_tone_becomes_a_single_note():
    clk = FakeClock()
    human = HumanStream(lambda: clk.beat)
    tracker, seg = _pipeline(human)
    _feed_tone(tracker, clk, pitch=45, seconds=0.5)  # A2 = 110 Hz
    _feed_silence(tracker, clk, seconds=0.3)
    seg.flush()

    notes, count = human.window(0.0)
    assert count >= 1
    assert all(n.pitch == 45 for n in notes), [n.pitch for n in notes]
    assert notes[0].start >= 0.0 and notes[0].duration > 0.0


def test_two_tones_become_two_notes_in_order():
    clk = FakeClock()
    human = HumanStream(lambda: clk.beat)
    tracker, seg = _pipeline(human)
    _feed_tone(tracker, clk, pitch=45, seconds=0.4)
    _feed_tone(tracker, clk, pitch=52, seconds=0.4)  # E3
    seg.flush()

    notes, _ = human.window(0.0)
    pitches = [n.pitch for n in notes]
    assert pitches == [45, 52], pitches
    assert notes[1].start > notes[0].start  # monotonic onsets on the shared clock


def test_audio_partner_stop_is_safe_before_start():
    partner = AudioPartnerInput(
        device="nonexistent", samplerate=SR, blocksize=BLOCK, frame_size=FRAME, hop=HOP,
        yin_threshold=0.15, confidence=0.5, noise_floor=0.01,
    )
    partner.stop()  # no capture started yet — must not raise
