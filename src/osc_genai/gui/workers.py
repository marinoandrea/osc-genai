"""Run a command as a child process and surface its output/lifecycle as Qt signals.

A :class:`QProcess` keeps the work off the GUI thread without us managing threads: its
``readyReadStandardOutput``/``finished`` signals are delivered on the Qt event loop. Stop sends
``SIGINT`` first (the realtime/training commands catch ``KeyboardInterrupt`` and release MIDI notes /
save state cleanly), escalating to ``kill`` only if the child ignores it.
"""

from __future__ import annotations

import os
import signal

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from osc_genai.gui.invoke import build_invocation

_GRACE_MS = 3000  # wait this long after SIGINT before force-killing


class CommandProcess(QObject):
    """Owns at most one running child process for a command, emitting its output and exit."""

    output = Signal(str)  # a chunk of merged stdout/stderr
    started = Signal()
    finished = Signal(int)  # process exit code

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: QProcess | None = None

    @property
    def running(self) -> bool:
        return (
            self._proc is not None
            and self._proc.state() != QProcess.ProcessState.NotRunning
        )

    def start(self, command_name: str, values: dict) -> None:
        if self.running:
            return
        program, args = build_invocation(command_name, values)
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain)
        proc.finished.connect(self._on_finished)
        proc.started.connect(self.started)
        self._proc = proc
        proc.start(program, args)

    def _drain(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        if data:
            self.output.emit(data)

    def _on_finished(self, code: int, _status: object) -> None:
        self._drain()  # flush any buffered tail before we drop the process
        self._proc = None
        self.finished.emit(int(code))

    def stop(self) -> None:
        """Ask the child to stop gracefully (SIGINT), then force-kill if it lingers."""
        if not self.running or self._proc is None:
            return
        pid = int(self._proc.processId())
        if pid > 0:
            try:
                os.kill(pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        QTimer.singleShot(_GRACE_MS, self._force_kill)

    def _force_kill(self) -> None:
        if self.running and self._proc is not None:
            self._proc.kill()
