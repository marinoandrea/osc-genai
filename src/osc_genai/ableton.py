"""Direct OSC client for talking to AbletonOSC.

AbletonOSC (https://github.com/ideoforms/AbletonOSC) is a MIDI Remote Script that
exposes Ableton Live's Object Model over OSC. It listens for messages on UDP 11000 and
sends replies on UDP 11001. This module is a thin wrapper over ``python-osc`` that knows
how to send commands and block on query replies — nothing Ableton-specific lives outside
of here.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Iterable

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from .generate import Note

# AbletonOSC's default wiring. It receives commands on SEND_PORT and replies on RECV_PORT.
DEFAULT_HOST = "127.0.0.1"
SEND_PORT = 11000
RECV_PORT = 11001


class AbletonOSC:
    """Send commands to AbletonOSC and read back query replies.

    A ``get`` query in AbletonOSC replies to the *same* address it was called on. There is
    no per-request id in the protocol — some getters discard extra arguments entirely
    (``num_tracks`` is ``lambda _: (len(tracks),)``) and others strict-unpack their params,
    so an appended id token can't be relied on as a correlation key.

    Instead we correlate by keeping a FIFO queue of waiters per address: the n-th reply on
    an address fulfils the n-th outstanding query on it. This makes concurrent queries
    safe (no more last-writer-wins) without touching the wire format. It assumes replies
    on a given address come back in send order, which holds for AbletonOSC's single
    in-order message pump over loopback UDP.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        send_port: int = SEND_PORT,
        recv_port: int = RECV_PORT,
    ) -> None:
        self._client = SimpleUDPClient(host, send_port)

        self._dispatcher = Dispatcher()
        self._dispatcher.set_default_handler(self._on_message)
        self._server = ThreadingOSCUDPServer((host, recv_port), self._dispatcher)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()

        # Awaited address -> FIFO queue of (Event, list-to-fill-with-reply-args) waiters.
        self._pending: dict[str, deque[tuple[threading.Event, list]]] = {}
        self._lock = threading.Lock()

    # -- low-level send / receive -------------------------------------------------

    def _on_message(self, address: str, *args) -> None:
        with self._lock:
            queue = self._pending.get(address)
            waiter = queue.popleft() if queue else None
            if queue is not None and not queue:
                del self._pending[address]
        if waiter is not None:
            event, sink = waiter
            sink.extend(args)
            event.set()

    def send(self, address: str, *args) -> None:
        """Fire-and-forget an OSC command."""
        self._client.send_message(address, list(args))

    def query(self, address: str, *args, timeout: float = 2.0) -> list:
        """Send a query and block until AbletonOSC replies on the same address.

        Returns the reply arguments (request args echoed back, then the value(s)).
        Raises ``TimeoutError`` if no reply arrives within ``timeout`` seconds.
        """
        event = threading.Event()
        sink: list = []
        waiter = (event, sink)
        with self._lock:
            self._pending.setdefault(address, deque()).append(waiter)

        self._client.send_message(address, list(args))

        if not event.wait(timeout):
            with self._lock:
                queue = self._pending.get(address)
                if queue is not None and waiter in queue:
                    queue.remove(waiter)
                    if not queue:
                        del self._pending[address]
            raise TimeoutError(
                f"No reply from AbletonOSC for {address} {list(args)} within {timeout}s. "
                "Is Ableton running with AbletonOSC enabled as a Control Surface?"
            )
        return sink

    # -- Live Object Model convenience methods ------------------------------------

    def get_num_tracks(self) -> int:
        """Number of tracks in the current Live set."""
        reply = self.query("/live/song/get/num_tracks")
        return int(reply[0])

    def get_track_name(self, track: int) -> str:
        """Name of ``track`` (reply echoes the index, then the name)."""
        reply = self.query("/live/track/get/name", track)
        # Reply is [track_index, name]; the name is the last argument.
        return str(reply[-1])

    def create_clip(self, track: int, slot: int, length: float) -> None:
        """Create an empty MIDI clip ``length`` beats long in ``track``'s ``slot``."""
        self.send("/live/clip_slot/create_clip", track, slot, float(length))

    def add_notes(self, track: int, slot: int, notes: Iterable[Note]) -> None:
        """Write MIDI notes into the clip at ``track``/``slot``.

        Each note is flattened to ``pitch, start, duration, velocity, mute`` and the
        sequence is appended after the track/slot indices, per AbletonOSC's
        ``/live/clip/add/notes`` contract.
        """
        args: list = [track, slot]
        for note in notes:
            args.extend(
                [
                    int(note.pitch),
                    float(note.start),
                    float(note.duration),
                    int(note.velocity),
                    bool(note.mute),
                ]
            )
        self.send("/live/clip/add/notes", *args)

    def fire_clip(self, track: int, slot: int) -> None:
        """Start playback of the clip at ``track``/``slot``."""
        self.send("/live/clip_slot/fire", track, slot)

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "AbletonOSC":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
