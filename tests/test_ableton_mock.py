"""Isolated M0 regression: mock AbletonOSC on private ports + the real client round-trip.

Uses ports 11900/11901 — never AbletonOSC's 11000/11001 — so the tests can never reach a live
Ableton. The client is constructed (binding the reply port) *before* the mock subprocess starts, so
the mock's real-Ableton probe finds the reply port busy and skips, then serves normally.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path

from osc_genai.core.note import Note, generate_notes, total_beats
from osc_genai.osc.ableton import AbletonOSC

MOCK = Path(__file__).resolve().parent.parent / "scripts" / "mock_ableton.py"
RECV_PORT = 11900
REPLY_PORT = 11901


def _wait_num_tracks(live: AbletonOSC, attempts: int = 50, timeout: float = 0.2) -> int:
    """Poll the mock with short-timeout queries until it answers (covers subprocess startup)."""
    for _ in range(attempts):
        try:
            return int(live.query("/live/song/get/num_tracks", timeout=timeout)[0])
        except TimeoutError:
            continue
    raise AssertionError("mock never responded to num_tracks")


@contextlib.contextmanager
def running_mock():
    """Yield a client connected to a freshly started mock on the isolated ports."""
    live = AbletonOSC(
        send_port=RECV_PORT, recv_port=REPLY_PORT
    )  # binds REPLY_PORT first
    proc = subprocess.Popen(
        [
            sys.executable,
            str(MOCK),
            "--recv-port",
            str(RECV_PORT),
            "--reply-port",
            str(REPLY_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # The mock answers 3 tracks; a real Ableton would answer otherwise. Blocking here until it
        # responds also confirms the test stayed isolated from any live set.
        assert _wait_num_tracks(live) == 3, (
            "expected the MOCK (3 tracks), not a real Ableton"
        )
        yield live
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        live.close()


def test_mock_write_path():
    with running_mock() as live:
        assert live.get_track_name(0) == "Mock Track 0"
        notes = generate_notes()
        # Fire-and-forget; just exercises that the send path encodes cleanly.
        live.create_clip(0, 0, total_beats(notes))
        live.add_notes(0, 0, notes)


def test_mock_read_path():
    with running_mock() as live:
        assert live.has_clip(0, 0) is True
        assert live.get_clip_notes(0, 0) == [Note(60, 0.0, 1.0, 100, False)]


def test_capture_from_ableton_scans_tracks_and_slots():
    from osc_genai.data.midi import capture_from_ableton

    with running_mock() as live:  # mock reports 3 tracks, every slot has the same clip
        sequences = capture_from_ableton(live, slots=2)
        assert len(sequences) == 3 * 2
        assert all(seq == [Note(60, 0.0, 1.0, 100, False)] for seq in sequences)
