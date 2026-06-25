"""Headless Qt UI smoke for the control center. Skipped unless the ``gui`` extra is installed.

Runs against Qt's ``offscreen`` platform so it needs no display. Exercises the registry-driven form
(value round-trip, required detection) and that the main window builds a page for every command.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6", reason="gui extra not installed")

from osc_genai.cli_spec import REGISTRY  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_form_values_round_trip(qt_app):
    from osc_genai.gui.forms import CommandForm

    form = CommandForm(REGISTRY["duet"])
    form.set_values(
        {"temperature": 0.5, "link": True, "device": "mps", "checkpoint": "m.pt"}
    )
    values = form.values()
    assert values["temperature"] == 0.5
    assert values["link"] is True
    assert values["device"] == "mps"
    assert values["checkpoint"] == "m.pt"


def test_form_missing_required(qt_app):
    from osc_genai.gui.forms import CommandForm

    form = CommandForm(
        REGISTRY["duet"]
    )  # --checkpoint is required and empty by default
    assert "--checkpoint" in form.missing_required()


def test_main_window_lists_every_command(qt_app):
    from osc_genai.gui.app import MainWindow

    window = MainWindow()
    assert window.stack.count() == len(REGISTRY)
    assert set(window._forms) == set(REGISTRY)
    assert window.current_command is not None  # a command is selected on launch
