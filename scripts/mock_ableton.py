"""A tiny stand-in for AbletonOSC, for testing without Ableton running.

Listens on UDP 11000 like AbletonOSC, logs every message it receives, and echoes a fake
reply on 11001 for the ``get`` queries our client makes. Run this in one terminal, then
``uv run osc-genai`` in another to exercise the full send + blocking-receive path.

    uv run python scripts/mock_ableton.py
"""

from __future__ import annotations

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

HOST = "127.0.0.1"
RECV_PORT = 11000  # where we receive commands (AbletonOSC's listen port)
REPLY_PORT = 11001  # where we send replies (AbletonOSC's send port)

reply_client = SimpleUDPClient(HOST, REPLY_PORT)


def handle(address: str, *args) -> None:
    print(f"<- {address} {list(args)}")
    if address == "/live/song/get/num_tracks":
        reply_client.send_message(address, [3])
        print(f"-> {address} [3]")
    elif address == "/live/track/get/name":
        track = args[0] if args else 0
        reply_client.send_message(address, [track, f"Mock Track {track}"])
        print(f"-> {address} [{track}, 'Mock Track {track}']")


def main() -> None:
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handle)
    server = BlockingOSCUDPServer((HOST, RECV_PORT), dispatcher)
    print(f"Mock AbletonOSC listening on {HOST}:{RECV_PORT}, replying on {REPLY_PORT}.")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
