"""Capture audio from a (virtual) input device via PortAudio/``sounddevice``.

The Live API exposes no PCM, so we don't ask Ableton for samples — we tap the bass at signal level
through a **virtual loopback device** and read it here. The source is selected **by name** and is
device-agnostic: BlackHole (the documented default, ``brew install blackhole-2ch``), Loopback, or a
CoreAudio Aggregate Device all work — route the bass track to it in Live (an Aggregate also lets you
keep monitoring). ``sounddevice`` is imported lazily so the package (and its tests) load without
PortAudio present.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

DEFAULT_DEVICE = "BlackHole 2ch"
DEFAULT_SAMPLERATE = 44100
DEFAULT_BLOCKSIZE = 1024


def _sd():
    """Import ``sounddevice`` lazily with a helpful message if PortAudio isn't installed."""
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio shared lib missing
        raise SystemExit(
            "audio capture needs PortAudio. Install it (macOS: 'brew install portaudio') "
            "and the 'sounddevice' package (it ships with this project)."
        ) from exc
    return sd


def list_input_devices() -> list[dict]:
    """Every device with at least one input channel (each dict carries name/channels/samplerate)."""
    sd = _sd()
    return [dict(d, index=i) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def find_device(name: str) -> dict | None:
    """Resolve an input device by exact name, then case-insensitive substring; ``None`` if absent."""
    devices = list_input_devices()
    for d in devices:
        if d["name"] == name:
            return d
    lowered = name.lower()
    for d in devices:
        if lowered in d["name"].lower():
            return d
    return None


class AudioCapture:
    """Stream mono float samples from an input device to ``callback`` on PortAudio's audio thread.

    Multi-channel devices are down-mixed to mono (channel average). ``callback`` runs on the
    real-time audio thread, so keep it light — YIN on a hop is fine; never block or allocate wildly.
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        *,
        device: str = DEFAULT_DEVICE,
        samplerate: int = DEFAULT_SAMPLERATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
    ) -> None:
        self._callback = callback
        self._device = device
        self._samplerate = int(samplerate)
        self._blocksize = int(blocksize)
        self._stream = None

    def start(self) -> "AudioCapture":
        sd = _sd()
        resolved = find_device(self._device)
        if resolved is None:
            raise SystemExit(
                f"audio device {self._device!r} not found. Run 'uv run audio-devices' to list inputs."
            )

        def _cb(indata, _frames, _time, status) -> None:
            mono = indata.mean(axis=1) if indata.ndim > 1 else indata
            self._callback(np.asarray(mono, dtype=np.float64))

        self._stream = sd.InputStream(
            samplerate=self._samplerate,
            blocksize=self._blocksize,
            device=resolved["index"],
            channels=1,
            dtype="float32",
            callback=_cb,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def main() -> None:
    """``audio-devices``: list input devices and verify the chosen capture device is present."""
    parser = argparse.ArgumentParser(description="List audio input devices and verify capture setup.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="device name to verify is present")
    args = parser.parse_args()

    print("Audio input devices:")
    for d in list_input_devices():
        print(f"  [{d['index']:>2}] {d['name']}  ({d['max_input_channels']} ch, "
              f"{int(d['default_samplerate'])} Hz)")
    found = find_device(args.device)
    if found is None:
        print(f"\n'{args.device}' is NOT available. Install a loopback device (e.g. "
              "'brew install blackhole-2ch') and route the bass track to it in Ableton.")
        raise SystemExit(1)
    print(f"\nOK: capture device {found['name']!r} is available (index {found['index']}).")


if __name__ == "__main__":
    main()
