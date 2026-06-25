"""Save/load named config presets — a command's form values persisted as JSON. Qt-free, testable.

This is the "snapshot configuration" pillar: a preset is just ``{command, name, values}`` so the
control center can repopulate a form from a saved set of flags. Stored under a per-user config dir
(override with ``OSC_GENAI_CONFIG_DIR``, which the tests use for isolation).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def presets_dir() -> Path:
    base = os.environ.get("OSC_GENAI_CONFIG_DIR")
    root = Path(base) if base else Path.home() / ".config" / "osc-genai"
    return root / "presets"


def _safe(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    return cleaned or "preset"


def _path(command: str, name: str) -> Path:
    return presets_dir() / command / f"{_safe(name)}.json"


def save_preset(command: str, name: str, values: dict[str, Any]) -> Path:
    """Persist ``values`` under ``command``/``name`` and return the file path."""
    path = _path(command, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"command": command, "name": name, "values": values}, indent=2)
    )
    return path


def list_presets(command: str) -> list[str]:
    """Sorted preset names saved for ``command`` (empty if none)."""
    directory = presets_dir() / command
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def load_preset(command: str, name: str) -> dict[str, Any]:
    """The saved values for ``command``/``name`` (raises if it does not exist)."""
    return json.loads(_path(command, name).read_text())["values"]


def delete_preset(command: str, name: str) -> None:
    _path(command, name).unlink(missing_ok=True)
