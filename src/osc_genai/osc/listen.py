"""Listen for inbound OSC and invoke a callback — the trigger side of the OSC bridge.

:class:`AbletonOSC` is the *client* (we send commands to Live and await replies). This is the
mirror image: a small UDP server that lets external software — a Max for Live device, a
TouchOSC layout, any OSC sender — fire actions inside a running session. The duet uses it so a
snapshot save can be triggered over OSC, not just from the keyboard.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11002  # osc-genai inbound triggers (AbletonOSC uses 11000/11001)


class OSCTrigger:
    """Run a UDP OSC server that calls a handler when a mapped address arrives.

    ``handlers`` maps an OSC address (e.g. ``"/snapshot"``) to a zero-argument
    callback. Any OSC arguments sent with the message are ignored, so a bare bang triggers it.
    Runs on a daemon thread; use as a context manager or call :meth:`start` / :meth:`close`.
    """

    def __init__(
        self,
        handlers: dict[str, Callable[[], None]],
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        dispatcher = Dispatcher()
        for address, callback in handlers.items():
            dispatcher.map(address, lambda _addr, *_args, _cb=callback: _cb())
        self._server = ThreadingOSCUDPServer((host, port), dispatcher)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> "OSCTrigger":
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "OSCTrigger":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
