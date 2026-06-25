"""Desktop control center for osc-genai (the ``control-center`` command).

A native Qt (PySide6) app that renders every command's flags — declared once in
:mod:`osc_genai.cli_spec` — as a form, runs the command, streams its output, and saves/loads config
presets. Kept out of the base install: import only under the ``gui`` extra.

This package's ``__init__`` is intentionally import-light (no Qt) so that the Qt-free helpers
(:mod:`osc_genai.gui.invoke`, :mod:`osc_genai.gui.presets`) can be imported and tested headlessly.
The Qt UI lives in :mod:`osc_genai.gui.app`, :mod:`osc_genai.gui.forms`, and
:mod:`osc_genai.gui.workers`.
"""
