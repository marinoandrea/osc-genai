"""Tests for the anticipatory lookahead/reconciliation buffer."""

from __future__ import annotations

from osc_genai.realtime.scheduler import AnticipatoryBuffer, Scheduled


def s(onset: float, pitch: int = 60, dur: float = 1.0) -> Scheduled:
    return Scheduled(onset=onset, pitch=pitch, velocity=100, dur=dur)


def test_add_keeps_buffer_sorted():
    buf = AnticipatoryBuffer()
    buf.add([s(3), s(1), s(2)])
    assert [x.onset for x in buf._notes] == [1, 2, 3]


def test_pop_due_returns_and_removes_arrived_notes():
    buf = AnticipatoryBuffer()
    buf.add([s(1), s(2), s(3)])
    due = buf.pop_due(2.0)
    assert [x.onset for x in due] == [1, 2]
    assert [x.onset for x in buf._notes] == [3]
    assert buf.pop_due(0.5) == []  # nothing new due


def test_reconcile_protects_commit_horizon_and_drops_tail():
    buf = AnticipatoryBuffer(commit_horizon=2.0)
    buf.add(
        [s(1, dur=1.0), s(3), s(5)]
    )  # playhead 0 -> horizon 2: only onset 1 is committed
    resume, dropped = buf.reconcile(playhead=0.0)
    assert dropped == 2
    assert [x.onset for x in buf._notes] == [1]
    assert resume == 2.0  # committed note ends at onset(1) + dur(1)


def test_reconcile_resumes_at_playhead_when_nothing_committed():
    buf = AnticipatoryBuffer(commit_horizon=1.0)
    buf.add([s(5)])  # horizon at playhead 2 is 3; onset 5 is revisable
    resume, dropped = buf.reconcile(playhead=2.0)
    assert dropped == 1 and resume == 2.0 and len(buf) == 0


def test_last_onset_tracks_lookahead_depth():
    buf = AnticipatoryBuffer()
    assert buf.last_onset(default=4.0) == 4.0
    buf.add([s(1), s(7)])
    assert buf.last_onset() == 7
