"""Qt-free control-center helpers: invocation building and preset persistence.

These run on the base install (no ``gui`` extra). The Qt UI smoke lives in ``test_gui_qt.py``,
which is skipped unless PySide6 is installed.
"""

from __future__ import annotations

import sys

from osc_genai.cli_spec import REGISTRY, defaults
from osc_genai.gui import presets
from osc_genai.gui.invoke import build_invocation

# -- Qt-free: invocation building -------------------------------------------------------------


def test_build_invocation_targets_entry_and_passes_flags():
    values = defaults(REGISTRY["train"])
    values["data_dir"] = "/tmp/midi"
    program, args = build_invocation("train", values)
    assert program == sys.executable
    assert args[0] == "-c"
    assert "from osc_genai.training.train import main" in args[1]
    # the flag values follow the -c <code> preamble
    tail = args[2:]
    assert "--data-dir" in tail and "/tmp/midi" in tail


def test_build_invocation_boolean_optional_and_store_true():
    values = defaults(REGISTRY["train-paired"])
    values["data_dir"] = "data/MIDI"
    values["interleaved"] = False  # BooleanOptionalAction -> --no-interleaved
    values["also_8"] = True  # store_true -> --also-8
    _, args = build_invocation("train-paired", values)
    tail = args[2:]
    assert "--no-interleaved" in tail
    assert "--also-8" in tail


# -- Qt-free: presets round-trip --------------------------------------------------------------


def test_preset_save_load_list_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_GENAI_CONFIG_DIR", str(tmp_path))
    values = {"epochs": 7, "balance_pitch": True, "data_dir": "x"}
    assert presets.list_presets("train") == []
    presets.save_preset("train", "my run", values)
    assert presets.list_presets("train") == ["my run"]
    assert presets.load_preset("train", "my run") == values
    presets.delete_preset("train", "my run")
    assert presets.list_presets("train") == []
