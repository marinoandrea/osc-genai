"""Turn a command + form values into a subprocess invocation. Qt-free, so it is unit-testable.

The control center runs each command as a child process of its own entry point — the same function
the installed console script calls — built from the registry. Running out-of-process keeps a crash
or a torch/MIDI hang from taking the GUI down and gives a clean stop (a signal to the child), while
the in-process hooks added to ``train``/``duet``/``stream`` (``on_epoch``/``stop``) remain available
for a later phase that embeds them directly for live telemetry.
"""

from __future__ import annotations

import sys

from osc_genai.cli_spec import REGISTRY, values_to_argv


def build_invocation(command_name: str, values: dict) -> tuple[str, list[str]]:
    """Return ``(program, args)`` to launch ``command_name`` with ``values`` via the Python entry.

    Uses ``python -c "from <module> import <func>; func()"`` rather than the console-script name so
    it works regardless of PATH/venv activation. The flag values become the child's ``sys.argv[1:]``,
    which its argparse (built from the same registry entry) parses back.
    """
    cmd = REGISTRY[command_name]
    module, func = cmd.entry.split(":")
    code = f"import sys; from {module} import {func} as _m; raise SystemExit(_m())"
    args = ["-c", code, *values_to_argv(cmd, values)]
    return sys.executable, args
