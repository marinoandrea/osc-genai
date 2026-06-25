"""The declarative command registry: parser parity, value round-tripping, and pyproject coupling.

These guard the single source of truth in :mod:`osc_genai.cli_spec` that both the CLI parsers and
the desktop control center consume. If a command's flags drift from its registry entry, or the
registry stops matching the installed console scripts, these fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osc_genai.cli_spec import REGISTRY, build_parser, defaults, values_to_argv

COMMANDS_WITH_FLAGS = [name for name, cmd in REGISTRY.items() if cmd.params]


def _required_filled(cmd):
    """Defaults plus a placeholder value for every required param (so argv is parseable)."""
    values = defaults(cmd)
    for p in cmd.params:
        if p.required:
            values[p.name] = ["a", "b"] if p.nargs == "+" else "REQ"
    return values


@pytest.mark.parametrize("name", COMMANDS_WITH_FLAGS)
def test_parser_builds_and_defaults_round_trip(name):
    """values_to_argv -> parse_args reproduces the original values (defaults case)."""
    cmd = REGISTRY[name]
    values = _required_filled(cmd)
    parsed = vars(build_parser(cmd).parse_args(values_to_argv(cmd, values)))
    for p in cmd.params:
        assert parsed[p.name] == values[p.name], f"{name}.{p.name}"


@pytest.mark.parametrize("name", COMMANDS_WITH_FLAGS)
def test_non_default_values_round_trip(name):
    """Flipping booleans and bumping numbers also survives the argv round-trip."""
    cmd = REGISTRY[name]
    values = _required_filled(cmd)
    for p in cmd.params:
        if p.required:
            continue
        if p.kind == "bool":
            values[p.name] = not values[p.name]
        elif p.kind == "int" and values[p.name] is not None:
            values[p.name] = int(values[p.name]) + 1
        elif p.kind == "float" and values[p.name] is not None:
            values[p.name] = float(values[p.name]) + 0.5
    parsed = vars(build_parser(cmd).parse_args(values_to_argv(cmd, values)))
    for p in cmd.params:
        assert parsed[p.name] == values[p.name], f"{name}.{p.name}"


@pytest.mark.parametrize("name", COMMANDS_WITH_FLAGS)
def test_argparse_defaults_match_registry_defaults(name):
    """Parsing an empty (required-stripped) arg list yields the registry's declared defaults."""
    cmd = REGISTRY[name]
    parser = build_parser(cmd)
    argv: list[str] = []
    for p in cmd.params:  # satisfy required flags with a placeholder
        if p.required:
            argv += [p.flag, "x"] if p.nargs != "+" else [p.flag, "x"]
    parsed = vars(parser.parse_args(argv))
    for p in cmd.params:
        if p.required:
            continue
        assert parsed[p.name] == p.effective_default, f"{name}.{p.name}"


def _pyproject_scripts() -> dict[str, str]:
    """Parse the ``[project.scripts]`` table (kept tomllib-free so it runs on 3.10)."""
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    lines = text.splitlines()
    start = lines.index("[project.scripts]")
    scripts: dict[str, str] = {}
    for line in lines[start + 1 :]:
        if line.startswith("["):  # next table
            break
        if "=" in line:
            name, _, value = line.partition("=")
            scripts[name.strip()] = value.strip().strip('"')
    return scripts


def test_registry_matches_pyproject_scripts():
    """Every console script in pyproject has a registry entry with a matching entry point.

    ``control-center`` is the GUI launcher, not a registry command, so it is excluded.
    """
    scripts = {k: v for k, v in _pyproject_scripts().items() if k != "control-center"}
    assert set(scripts) == set(REGISTRY), "registry and [project.scripts] disagree"
    for name, entry in scripts.items():
        assert REGISTRY[name].entry == entry, (
            f"{name}: {REGISTRY[name].entry} != {entry}"
        )
