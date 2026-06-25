"""A shared beat clock for the real-time commands — local wall-clock or Ableton Link.

The duet/play/fake-human loops used to compute time from ``time.perf_counter()`` and a fixed
``--bpm``, so they ran on a clock of their own: generated MIDI drifted against Ableton Live and
landed on an arbitrary phase. This module factors timing behind one tiny interface so the loops can
instead read a beat position that is *shared* with Live.

Two implementations, same surface (``beat``/``tempo``/``playing``):

* :class:`WallClock` — the original behavior. ``beat`` advances from a local origin at a fixed BPM
  and ``playing`` is always true. ``WallClock(bpm).beat * steps_per_beat`` equals the old
  ``(now - start) / sec_per_step``, so the non-Link path is unchanged.
* :class:`LinkClock` — backed by `Ableton Link <https://github.com/artfwo/aalink>`_. ``beat`` is the
  session beat (phase-aligned to ``quantum``), ``tempo`` follows the session (Live is master), and
  ``playing`` reflects Live's transport when start/stop sync is on. ``aalink`` is an optional extra
  (``pip install osc-genai[link]``); it is imported lazily so this module loads without it.
"""

from __future__ import annotations

import asyncio
import time
import warnings


class WallClock:
    """Free-running local clock at a fixed tempo — the original (non-Link) timing."""

    def __init__(self, bpm: float) -> None:
        self._bpm = bpm
        self._start = time.perf_counter()

    @property
    def beat(self) -> float:
        return (time.perf_counter() - self._start) * self._bpm / 60.0

    @property
    def tempo(self) -> float:
        return self._bpm

    @property
    def playing(self) -> bool:
        return True

    @property
    def peers(self) -> int:
        return 0


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """aalink's ``Link`` is built around an asyncio loop; we only read properties, so it need not
    run — but a loop must exist for construction. Reuse the current one or make a fresh one."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class LinkClock:
    """Ableton Link session clock: shared beat timeline, tempo, and transport with Live."""

    def __init__(
        self, bpm: float, quantum: int = 4, start_stop_sync: bool = True
    ) -> None:
        from aalink import (
            Link,  # lazy: optional dependency, imported only when --link is used
        )

        # aalink schedules sync() callbacks on an asyncio loop. We only read properties (beat/tempo/
        # playing), so the loop never has to run — but aalink 0.2.x grabs the *running* loop at
        # construction and raises off the main thread, so we hand it an explicit loop. Newer aalink
        # self-manages its loop and deprecates the parameter, so try the bare form first.
        try:
            self._link = Link(bpm)
        except (RuntimeError, TypeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._link = Link(bpm, _ensure_event_loop())
        self._link.quantum = quantum
        self._link.start_stop_sync_enabled = start_stop_sync
        self._link.enabled = True
        self._start_stop_sync = start_stop_sync

    @property
    def beat(self) -> float:
        return self._link.beat

    @property
    def tempo(self) -> float:
        return self._link.tempo

    @property
    def playing(self) -> bool:
        # When not following Live's transport, play continuously on the shared tempo/beat grid —
        # otherwise the transport gate would keep us silent until someone broadcasts a "playing" state.
        return self._link.playing if self._start_stop_sync else True

    @property
    def peers(self) -> int:
        return self._link.num_peers


def make_clock(
    use_link: bool, *, bpm: float, quantum: int = 4, start_stop_sync: bool = True
):
    """Build the clock the realtime loops read from: :class:`LinkClock` when ``use_link``, else
    :class:`WallClock`. Raises a helpful error if Link is requested but ``aalink`` isn't installed."""
    if not use_link:
        return WallClock(bpm)
    try:
        clock = LinkClock(bpm, quantum=quantum, start_stop_sync=start_stop_sync)
    except ImportError as exc:
        raise SystemExit(
            "--link needs the 'aalink' package (Ableton Link). Install it with:\n"
            "    uv sync --extra link      # or: pip install osc-genai[link]"
        ) from exc
    if clock.peers == 0:
        print(
            "link: enabled, but no Link peers found yet. Enable Link in Ableton Live "
            "(and press Play if start/stop sync is on) — output stays silent until the transport runs."
        )
    return clock
