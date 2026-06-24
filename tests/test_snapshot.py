"""Tests for the live-duet snapshot save (last N bars -> personal training pair)."""

from __future__ import annotations

import threading

from pythonosc.udp_client import SimpleUDPClient

from osc_genai.core.note import Note
from osc_genai.data.midi import load_midi_file
from osc_genai.data.pairs import build_aligned_pairs
from osc_genai.data.snapshot import save_snapshot
from osc_genai.osc.listen import OSCTrigger

BPB = 4.0  # beats per bar
BARS = 4


def _human(beats: list[float]) -> list[Note]:
    return [Note(40 + i, b, 0.5, 100, False, 0) for i, b in enumerate(beats)]


def _machine(beats: list[float]) -> list[Note]:
    return [Note(36, b, 0.25, 100, False, 9) for b in beats]


def test_save_snapshot_writes_aligned_pair(tmp_path):
    # 8 bars (0..32 beats) played; snapshot the last 4 bars ending at the bar boundary <= 33.
    human = _human([1.0, 5.0, 17.0, 20.0, 28.0])  # last three fall in bars 4-7 (>=16)
    machine = _machine([16.0, 18.0, 24.0, 30.5])

    result = save_snapshot(
        human, machine, tmp_path,
        end_beat=33.0, bars=BARS, beats_per_bar=BPB,  # floors to bar boundary 32 -> window [16, 32)
        human_inst="Bass", machine_inst="Drums", song_id="sess-000",
    )
    assert result is not None
    human_path, machine_path = result

    # Two files, expected personal/ layout, shared song-id prefix.
    assert human_path == tmp_path / "Bass" / "personal" / "sess-000__human.mid"
    assert machine_path == tmp_path / "Drums" / "personal" / "sess-000__machine.mid"
    assert human_path.name.split("__")[0] == machine_path.name.split("__")[0]

    # Window is [16, 32): re-origined so both clips start at ~0 and stay within N bars.
    h = load_midi_file(human_path)
    m = load_midi_file(machine_path)
    assert [round(n.start, 3) for n in h] == [1.0, 4.0, 12.0]  # 17->1, 20->4, 28->12
    assert [round(n.start, 3) for n in m] == [0.0, 2.0, 8.0, 14.5]
    assert all(0.0 <= n.start < BARS * BPB for n in h + m)


def test_save_snapshot_returns_none_when_too_little_played(tmp_path):
    result = save_snapshot(
        _human([1.0]), _machine([2.0]), tmp_path,
        end_beat=8.0, bars=BARS, beats_per_bar=BPB,  # window would start at 8 - 16 < 0
        human_inst="Bass", machine_inst="Drums", song_id="sess-000",
    )
    assert result is None
    assert not (tmp_path / "Bass").exists()
    assert not (tmp_path / "Drums").exists()


def test_snapshot_becomes_training_pair(tmp_path):
    # A snapshot written into a temp data root is paired by the existing pipeline.
    human = _human([16.0, 18.0, 20.0, 28.0])
    machine = _machine([16.0, 17.0, 24.0, 30.0])
    save_snapshot(
        human, machine, tmp_path,
        end_beat=32.0, bars=BARS, beats_per_bar=BPB,
        human_inst="Bass", machine_inst="Drums", song_id="sess-000",
    )

    pairs = build_aligned_pairs(tmp_path, "Bass", "Drums", chunk_bars=BARS, normalize_drums=False)
    assert pairs, "snapshot should reconstruct into at least one aligned pair"
    pair = pairs[0]
    assert pair.context and pair.target  # both stems present in the window


def test_osc_trigger_invokes_callback():
    fired = threading.Event()
    port = 11099
    with OSCTrigger({"/snapshot": fired.set}, port=port):
        SimpleUDPClient("127.0.0.1", port).send_message("/snapshot", [])
        assert fired.wait(2.0), "OSC bang should invoke the mapped handler"
