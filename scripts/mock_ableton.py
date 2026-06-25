"""A tiny stand-in for AbletonOSC, for testing without Ableton running.

Listens like AbletonOSC (default UDP 11000 for commands, replies on 11001), logs every message it
receives, and answers the ``get`` queries our client makes. Run it in one terminal, then
``uv run osc-genai`` in another to exercise the full send + blocking-receive path::

    uv run python scripts/mock_ableton.py

Ports are configurable (``--recv-port`` / ``--reply-port``) so a test can run the mock in isolation
on private ports. As a safety net, the mock **refuses to start if a real AbletonOSC is already
answering** on the command port: otherwise an open Ableton set silently shadows the mock and
receives the test's note writes (which is exactly how a stray clip can end up in a live set).
"""

from __future__ import annotations

import argparse
import threading

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer, ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

DEFAULT_HOST = "127.0.0.1"
DEFAULT_RECV_PORT = 11000  # where we receive commands (AbletonOSC's listen port)
DEFAULT_REPLY_PORT = 11001  # where we send replies (AbletonOSC's send port)

NUM_TRACKS = 3  # distinct from a typical real set, so tests can tell mock from Ableton


def make_dispatcher(reply_client: SimpleUDPClient) -> Dispatcher:
    """Build a dispatcher that logs commands and answers the queries our client makes."""

    def handle(address: str, *args) -> None:
        print(f"<- {address} {list(args)}")
        if address == "/live/song/get/num_tracks":
            reply_client.send_message(address, [NUM_TRACKS])
            print(f"-> {address} [{NUM_TRACKS}]")
        elif address == "/live/track/get/name":
            track = args[0] if args else 0
            reply_client.send_message(address, [track, f"Mock Track {track}"])
            print(f"-> {address} [{track}, 'Mock Track {track}']")
        elif address == "/live/clip_slot/get/has_clip":
            track = args[0] if args else 0
            slot = args[1] if len(args) > 1 else 0
            reply_client.send_message(address, [track, slot, True])
            print(f"-> {address} [{track}, {slot}, True]")
        elif address == "/live/clip/get/notes":
            track = args[0] if args else 0
            slot = args[1] if len(args) > 1 else 0
            # one fixed note so the read path has something deterministic to assert.
            reply_client.send_message(address, [track, slot, 60, 0.0, 1.0, 100, False])
            print(f"-> {address} [{track}, {slot}, 60, 0.0, 1.0, 100, False]")

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handle)
    return dispatcher


def real_ableton_present(
    host: str, recv_port: int, reply_port: int, timeout: float = 0.25
) -> bool:
    """Best-effort probe: is a real AbletonOSC already answering on ``recv_port``?

    Sends a harmless ``num_tracks`` query and listens briefly on ``reply_port`` for any reply. If
    something answers, a live Ableton (or another mock) owns the ports and we should not start.
    Returns ``False`` if the reply port can't be bound (e.g. our own client holds it) — then we
    can't probe, so we let the caller proceed.
    """
    answered = threading.Event()
    probe = Dispatcher()
    probe.set_default_handler(lambda *_: answered.set())
    try:
        server = ThreadingOSCUDPServer((host, reply_port), probe)
    except OSError:
        return False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        SimpleUDPClient(host, recv_port).send_message("/live/song/get/num_tracks", [])
        answered.wait(timeout)
    finally:
        server.shutdown()
        server.server_close()
    return answered.is_set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock AbletonOSC for offline testing.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--recv-port", type=int, default=DEFAULT_RECV_PORT)
    parser.add_argument("--reply-port", type=int, default=DEFAULT_REPLY_PORT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="start even if a real AbletonOSC seems to be answering on the command port",
    )
    args = parser.parse_args()

    if not args.force and real_ableton_present(
        args.host, args.recv_port, args.reply_port
    ):
        raise SystemExit(
            f"A real AbletonOSC is answering on {args.host}:{args.recv_port}. Refusing to start so "
            "the mock doesn't shadow your live set (note writes would hit Ableton). Close Ableton, "
            "use --recv-port/--reply-port for an isolated run, or pass --force to override."
        )

    reply_client = SimpleUDPClient(args.host, args.reply_port)
    server = BlockingOSCUDPServer(
        (args.host, args.recv_port), make_dispatcher(reply_client)
    )
    print(
        f"Mock AbletonOSC listening on {args.host}:{args.recv_port}, replying on {args.reply_port}."
    )
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
