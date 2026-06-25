"""Single source of truth for every command's flags.

Each console script's command-line surface is declared here once, as a :class:`Command` of
:class:`Param`s, and consumed by three callers:

- the CLI ``main()`` functions, via :func:`build_parser` (so flags are declared once, not inline);
- the desktop control center, which renders each :class:`Command` as a form (the ``Param`` carries
  enough metadata — type, default, help, widget hint — to pick the right widget);
- the GUI's subprocess fallback, via :func:`values_to_argv`, which turns a form's values back into a
  command line so any command can be launched as its installed console script.

This module is deliberately dependency-light (stdlib only): importing the registry must not pull in
torch / mido / sounddevice, so the GUI can show every form without loading the heavy realtime/model
code. The heavy module is imported only when a command is actually run in-process.

Defaults mirror the values in each command module. Note that ``argparse`` does not print defaults in
``--help`` (no command uses ``ArgumentDefaultsHelpFormatter``), so the default values here affect
behavior and the GUI but not ``--help`` parity, which depends only on flags, help text, and actions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

# Mirrors osc_genai.core.event.DEFAULT_STEPS_PER_BEAT (kept literal to avoid a heavy import here).
DEFAULT_STEPS_PER_BEAT = 4
DEFAULT_IN_PORT = "osc-genai in"
DEFAULT_OUT_PORT = "osc-genai out"
DEFAULT_AUDIO_DEVICE = "BlackHole 2ch"
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


@dataclass(frozen=True)
class Param:
    """One command-line flag, described once for the CLI parser and the GUI form generator."""

    name: str  # argparse dest, snake_case (flag derived as --kebab-case)
    kind: str = "str"  # "int" | "float" | "str" | "bool"
    default: Any = None
    required: bool = False
    help: str = ""
    nargs: str | None = None  # "+" for multi-value flags (e.g. --context-dir)
    boolean_optional: bool = (
        False  # bool rendered as --x / --no-x (argparse.BooleanOptionalAction)
    )
    widget: str | None = (
        None  # GUI hint: file | dir | checkpoint | midi_port | midi_in |
    )
    #                                  audio_device | device
    ui_choices: tuple[str, ...] | None = (
        None  # GUI dropdown values (NOT passed to argparse)
    )
    min: float | None = None  # GUI numeric bound hint
    max: float | None = None

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")

    @property
    def neg_flag(self) -> str:
        """The ``--no-...`` form for a BooleanOptionalAction flag."""
        return "--no-" + self.name.replace("_", "-")

    @property
    def effective_default(self) -> Any:
        """Default as argparse would yield it — a plain ``store_true`` flag defaults to ``False``."""
        if self.kind == "bool" and not self.boolean_optional:
            return bool(self.default)
        return self.default


@dataclass(frozen=True)
class Command:
    """A console script: its identity, how to launch it, and its full flag surface."""

    name: str  # console-script name, e.g. "train-paired"
    entry: str  # "module.path:func" (matches pyproject [project.scripts])
    description: str  # argparse description, shown in --help
    summary: str  # short label for the GUI
    kind: str  # "realtime" | "training" | "oneshot"
    params: tuple[Param, ...] = ()
    exclusive: bool = (
        False  # realtime command that grabs MIDI/audio ports (only one at a time)
    )

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)


# -- parser / argv adapters -------------------------------------------------------------------


def _add_param(parser: argparse.ArgumentParser, p: Param) -> None:
    if p.kind == "bool":
        if p.boolean_optional:
            parser.add_argument(
                p.flag,
                action=argparse.BooleanOptionalAction,
                default=p.default,
                help=p.help or None,
            )
        else:
            parser.add_argument(p.flag, action="store_true", help=p.help or None)
        return
    kwargs: dict[str, Any] = {"default": p.default, "help": p.help or None}
    if p.kind == "int":
        kwargs["type"] = int
    elif p.kind == "float":
        kwargs["type"] = float
    if p.required:
        kwargs["required"] = True
    if p.nargs:
        kwargs["nargs"] = p.nargs
    parser.add_argument(p.flag, **kwargs)


def build_parser(cmd: Command) -> argparse.ArgumentParser:
    """Reconstruct a command's argparse parser from its declarative spec."""
    parser = argparse.ArgumentParser(prog=cmd.name, description=cmd.description)
    for p in cmd.params:
        _add_param(parser, p)
    return parser


def defaults(cmd: Command) -> dict[str, Any]:
    """The command's default values, keyed by param name — the GUI form's initial state."""
    return {p.name: p.effective_default for p in cmd.params}


def values_to_argv(cmd: Command, values: dict[str, Any]) -> list[str]:
    """Turn a values dict into a command line, the inverse of parsing.

    Used by the GUI subprocess fallback and by the round-trip parity test. Omits ``None``/empty
    optionals so the parser falls back to their defaults.
    """
    argv: list[str] = []
    for p in cmd.params:
        v = values.get(p.name, p.default)
        if p.kind == "bool":
            if p.boolean_optional:
                argv.append(p.flag if v else p.neg_flag)
            elif v:
                argv.append(p.flag)
        elif p.nargs == "+":
            if v:
                argv.append(p.flag)
                argv.extend(str(x) for x in v)
        else:
            if v is None or v == "":
                continue
            argv.extend([p.flag, str(v)])
    return argv


# -- shared param fragments -------------------------------------------------------------------


def _device(default: str, help: str) -> Param:
    return Param(
        "device", "str", default, help=help, widget="device", ui_choices=DEVICE_CHOICES
    )


def _link_params() -> tuple[Param, ...]:
    return (
        Param(
            "bpm",
            "float",
            130.0,
            help="tempo; with --link this is only the fallback/seed",
        ),
        Param("link", "bool", help="ride Ableton Link's grid/tempo/transport"),
        Param("quantum", "int", 4, help="Link bar length in beats (phase alignment)"),
    )


# -- the registry -----------------------------------------------------------------------------

_DUET = Command(
    name="duet",
    entry="osc_genai.realtime.duet:main",
    description="Real-time anticipatory duet (model plays with you).",
    summary="Real-time duet: the model accompanies you live.",
    kind="realtime",
    exclusive=True,
    params=(
        Param(
            "checkpoint",
            "str",
            required=True,
            help="trained duet model (.pt)",
            widget="checkpoint",
        ),
        Param(
            "in_port",
            "str",
            DEFAULT_IN_PORT,
            help="port you play into",
            widget="midi_port",
        ),
        Param(
            "out_port",
            "str",
            DEFAULT_OUT_PORT,
            help="port the model plays out",
            widget="midi_port",
        ),
        *_link_params(),
        Param(
            "start_stop_sync",
            "bool",
            True,
            boolean_optional=True,
            help="with --link, only play while Ableton's transport runs",
        ),
        Param("steps_per_beat", "int", DEFAULT_STEPS_PER_BEAT),
        Param("temperature", "float", 0.95),
        Param("chunk_events", "int", 8, help="events generated per chunk"),
        Param("lookahead", "float", 8.0, help="grid steps kept generated ahead"),
        Param("commit_horizon", "float", 2.0, help="steps ahead locked from revision"),
        Param(
            "window_beats", "float", 16.0, help="trailing history fed back as context"
        ),
        Param(
            "out_channel",
            "int",
            9,
            help="pin drums to this MIDI channel (-1 = model's own)",
        ),
        Param(
            "echo_channel",
            "int",
            0,
            help="echo the human's bass out on this channel so it shares the duet clock (-1 = off)",
        ),
        Param(
            "audio_in",
            "bool",
            help="capture a real instrument as audio and pitch-track it (monophonic, instead of MIDI in)",
        ),
        Param(
            "audio_device",
            "str",
            DEFAULT_AUDIO_DEVICE,
            help="capture device name (a loopback) for --audio-in",
            widget="audio_device",
        ),
        Param("audio_samplerate", "int", 44100, help="audio capture sample rate"),
        Param("audio_blocksize", "int", 1024, help="audio capture block size"),
        Param(
            "frame_size",
            "int",
            4096,
            help="YIN analysis window in samples (4096 covers a bass's low E ~41Hz; smaller = lower latency but higher pitch floor)",
        ),
        Param("hop", "int", 512, help="YIN frame advance in samples"),
        Param("yin_threshold", "float", 0.15, help="YIN aperiodicity threshold"),
        Param("confidence", "float", 0.5, help="min YIN probability to voice a note"),
        Param(
            "noise_floor", "float", 0.01, help="min RMS to voice a note (gate silence)"
        ),
        Param(
            "audio_echo",
            "bool",
            help="with --audio-in, echo the tracked notes out the duet port for monitoring",
        ),
        Param(
            "pitch_bias",
            "str",
            None,
            help='per-pitch logit bias, e.g. "47:-3,48:-3,38:1"',
        ),
        Param(
            "regular_pitches",
            "str",
            "36,38",
            help="foundation lanes kept regular (default kick,snare)",
        ),
        Param(
            "regular_temperature",
            "float",
            0.15,
            help="near-greedy timing temp for the foundation",
        ),
        Param(
            "snap_steps",
            "int",
            2,
            help="snap foundation onsets to this grid (2=8th, 4=1/4, 0=off)",
        ),
        Param("seconds", "float", None, help="stop after N seconds"),
        _device("cpu", "cpu | cuda | mps | auto (realtime defaults to cpu)"),
        Param(
            "snapshot_dir",
            "str",
            "data/MIDI",
            help="dataset root snapshots are saved into",
            widget="dir",
        ),
        Param("no_snapshots", "bool", help="disable the keypress snapshot trigger"),
        Param(
            "snapshot_bars",
            "int",
            4,
            help="bars per snapshot (match the training chunk size)",
        ),
        Param(
            "snapshot_human_inst", "str", "Bass", help="instrument folder for your part"
        ),
        Param(
            "snapshot_machine_inst",
            "str",
            "Drums",
            help="instrument folder for the model's part",
        ),
        Param(
            "snapshot_artist",
            "str",
            "personal",
            help="artist folder snapshots land under",
        ),
        Param(
            "snapshot_key",
            "str",
            "s",
            help="key to press (then Enter) to save a snapshot",
        ),
        Param(
            "snapshot_osc_port",
            "int",
            11002,
            help="UDP port to listen on for an OSC snapshot trigger (<=0 = off)",
        ),
        Param(
            "snapshot_osc_addr",
            "str",
            "/snapshot",
            help="OSC address that triggers a snapshot",
        ),
    ),
)

_PLAY = Command(
    name="play",
    entry="osc_genai.realtime.play:main",
    description="Stream live model generation to a MIDI port.",
    summary="Stream the model's output continuously to a MIDI port.",
    kind="realtime",
    exclusive=True,
    params=(
        Param(
            "checkpoint",
            "str",
            required=True,
            help="trained model (.pt)",
            widget="checkpoint",
        ),
        Param("out_port", "str", DEFAULT_OUT_PORT, widget="midi_port"),
        *_link_params(),
        Param(
            "start_stop_sync",
            "bool",
            True,
            boolean_optional=True,
            help="with --link, only play while Ableton's transport runs",
        ),
        Param("steps_per_beat", "int", DEFAULT_STEPS_PER_BEAT),
        Param("temperature", "float", 0.95),
        Param("seconds", "float", None, help="stop after N seconds"),
        Param("no_virtual", "bool", help="connect to an existing port by name"),
        _device("cpu", "cpu | cuda | mps | auto (realtime defaults to cpu)"),
    ),
)

_LIVE = Command(
    name="live",
    entry="osc_genai.realtime.live:main",
    description="Real-time MIDI duet over virtual MIDI ports.",
    summary="Rule-based harmonizer duet (M1 stand-in, no model).",
    kind="realtime",
    exclusive=True,
    params=(
        Param(
            "in_port",
            "str",
            DEFAULT_IN_PORT,
            help="input port name",
            widget="midi_port",
        ),
        Param(
            "out_port",
            "str",
            DEFAULT_OUT_PORT,
            help="output port name",
            widget="midi_port",
        ),
        Param(
            "intervals",
            "str",
            "4,7",
            help="comma-separated semitone intervals for the harmonizer stand-in (default 4,7)",
        ),
        Param(
            "no_virtual",
            "bool",
            help="connect to existing ports by name instead of creating virtual ports",
        ),
        Param("list_ports", "bool", help="list available MIDI ports and exit"),
    ),
)

_GENERATE = Command(
    name="generate",
    entry="osc_genai.inference:main",
    description="Generate a phrase with a trained model into Live.",
    summary="Generate a phrase into an Ableton clip (offline).",
    kind="oneshot",
    params=(
        Param(
            "checkpoint",
            "str",
            required=True,
            help="path to a saved model (.pt)",
            widget="checkpoint",
        ),
        Param("track", "int", 0, help="destination track for the response"),
        Param("slot", "int", 0, help="destination clip slot"),
        Param(
            "context_track", "int", None, help="track to read a call/context clip from"
        ),
        Param("context_slot", "int", 0, help="slot of the context clip"),
        Param("temperature", "float", 1.0),
        Param("max_events", "int", 64),
        Param("steps_per_beat", "int", DEFAULT_STEPS_PER_BEAT),
        _device("auto", "cpu | cuda | mps | auto"),
    ),
)

_TRAIN = Command(
    name="train",
    entry="osc_genai.training.train:main",
    description="Train the factored event model on a MIDI corpus.",
    summary="Train the unconditional model on a MIDI corpus.",
    kind="training",
    params=(
        Param(
            "data_dir",
            "str",
            required=True,
            help=".mid folder (searched recursively)",
            widget="dir",
        ),
        Param("out", "str", "model.pt", help="checkpoint output path", widget="file"),
        Param("epochs", "int", 40),
        Param("batch_size", "int", 32),
        Param("lr", "float", 1e-3),
        Param(
            "transpose",
            "int",
            5,
            help="augment by +/- this many semitones (0 disables)",
        ),
        Param("steps_per_beat", "int", DEFAULT_STEPS_PER_BEAT),
        Param("hidden", "int", 256),
        Param("layers", "int", 1),
        _device("auto", "cpu | cuda | mps | auto"),
        Param("balance_pitch", "bool", help="up-weight rare pitches in loss"),
    ),
)

_TRAIN_CONDITIONAL = Command(
    name="train-conditional",
    entry="osc_genai.training.train:conditional_main",
    description="Train a directional context->target snapshot.",
    summary="Train a context->target model (cross-paired clips).",
    kind="training",
    params=(
        Param(
            "context_dir",
            "str",
            nargs="+",
            required=True,
            help=".mid folder(s) for context role",
            widget="dir",
        ),
        Param(
            "target_dir",
            "str",
            nargs="+",
            required=True,
            help=".mid folder(s) for response role",
            widget="dir",
        ),
        Param("out", "str", "model.pt", widget="file"),
        Param("pairs_per_context", "int", 4),
        Param("epochs", "int", 30),
        Param("batch_size", "int", 32),
        Param("lr", "float", 1e-3),
        Param("hidden", "int", 256),
        Param("layers", "int", 1),
        Param("steps_per_beat", "int", DEFAULT_STEPS_PER_BEAT),
        _device("auto", "cpu | cuda | mps | auto"),
        Param("seed", "int", 0),
        Param(
            "balance_pitch", "bool", help="up-weight rare pitches (kick/snare) in loss"
        ),
        Param(
            "target_drums",
            "bool",
            help="targets are drums: keep full kits, normalize to GM",
        ),
    ),
)

_TRAIN_PAIRED = Command(
    name="train-paired",
    entry="osc_genai.training.train:paired_main",
    description="Train a directional model on same-song, time-aligned instrument pairs.",
    summary="Train on time-aligned instrument pairs (bass->drums).",
    kind="training",
    params=(
        Param(
            "data_dir",
            "str",
            "data/MIDI",
            help="<Instrument>/<Artist>/ clip store",
            widget="dir",
        ),
        Param("context_inst", "str", "Bass"),
        Param("target_inst", "str", "Drums"),
        Param("chunk_bars", "int", 4),
        Param("also_8", "bool", help="also emit 8-bar chunks"),
        Param("hop_bars", "int", None, help="window hop (default = chunk size)"),
        Param("no_normalize_drums", "bool"),
        Param(
            "regular_drums",
            "bool",
            help="snap kick/snare onsets to a coarse grid so they train regular (hats stay free)",
        ),
        Param(
            "regular_grid",
            "float",
            0.5,
            help="grid in beats for --regular-drums (0.5 = 8th notes, 1.0 = quarters)",
        ),
        Param(
            "phase",
            "bool",
            help="feed bar-relative grid position as an input feature (anchors kick/snare to the grid)",
        ),
        Param("beats_per_bar", "int", 4, help="bar length for the --phase feature"),
        Param("out", "str", "models/bass2drums.pt", widget="file"),
        Param(
            "transpose", "int", 5, help="augment context by +/- semitones (0 disables)"
        ),
        Param("epochs", "int", 200),
        Param("batch_size", "int", 32),
        Param("lr", "float", 1e-3),
        Param("hidden", "int", 256),
        Param("layers", "int", 1),
        Param("steps_per_beat", "int", DEFAULT_STEPS_PER_BEAT),
        _device("auto", "cpu | cuda | mps | auto"),
        Param("balance_pitch", "bool", help="up-weight rare pitches (kick/snare)"),
        Param(
            "interleaved",
            "bool",
            True,
            boolean_optional=True,
            help="interleave both lines on a shared clock with a source tag (the live-duet model); "
            "--no-interleaved falls back to legacy prefix conditioning (call-and-response)",
        ),
    ),
)

_BUILD_PAIRS = Command(
    name="build-pairs",
    entry="osc_genai.data.pairs:main",
    description="Build same-song, time-aligned instrument-pair chunks; print stats / materialize.",
    summary="Inspect / materialize time-aligned instrument-pair chunks.",
    kind="oneshot",
    params=(
        Param("data_dir", "str", "data/MIDI", widget="dir"),
        Param("context_inst", "str", "Bass"),
        Param("target_inst", "str", "Drums"),
        Param("chunk_bars", "int", 4),
        Param("also_8", "bool", help="also emit 8-bar chunks"),
        Param("hop_bars", "int", None, help="window hop (default = chunk size)"),
        Param("no_require_both", "bool", help="keep windows with one stem empty"),
        Param("no_normalize_drums", "bool"),
        Param(
            "materialize",
            "str",
            None,
            help="write a <Ctx>_to_<Tgt>/ tree here",
            widget="dir",
        ),
    ),
)

_FAKE_HUMAN = Command(
    name="fake-human",
    entry="osc_genai.realtime.fake_human:main",
    description="Loop a MIDI line into the duet's input port.",
    summary="Loop a MIDI line into the duet's input (for testing).",
    kind="realtime",
    params=(
        Param(
            "to_port",
            "str",
            DEFAULT_IN_PORT,
            help="port to send into (the duet's input)",
            widget="midi_port",
        ),
        Param(
            "from_data",
            "str",
            None,
            help="folder of .mid files; loops one clip",
            widget="dir",
        ),
        Param("midi", "str", None, help="a single .mid file to loop", widget="file"),
        *_link_params(),
        Param(
            "start_stop_sync",
            "bool",
            True,
            boolean_optional=True,
            help="with --link, only loop while Ableton's transport runs",
        ),
        Param("seconds", "float", None, help="stop after N seconds"),
        Param("seed", "int", 0, help="which clip to pick from --from-data"),
        Param("virtual", "bool", help="create the port instead of connecting"),
    ),
)

_RECORD = Command(
    name="record",
    entry="osc_genai.data.record:main",
    description="Record a duet session (human + machine streams).",
    summary="Record a duet session to paired-session JSON.",
    kind="oneshot",
    params=(
        Param(
            "human_port",
            "str",
            "IAC Driver Bus 1",
            help="port the human plays on",
            widget="midi_in",
        ),
        Param(
            "machine_port",
            "str",
            DEFAULT_OUT_PORT,
            help="the model's output port",
            widget="midi_in",
        ),
        Param("bpm", "float", 130.0),
        Param("seconds", "float", None, help="record length (else Ctrl-C)"),
        Param(
            "out",
            "str",
            "session.json",
            help="output paired-session JSON",
            widget="file",
        ),
    ),
)

_AUDIO_DEVICES = Command(
    name="audio-devices",
    entry="osc_genai.audio.capture:main",
    description="List audio input devices and verify capture setup.",
    summary="List audio input devices; verify the capture device.",
    kind="oneshot",
    params=(
        Param(
            "device",
            "str",
            DEFAULT_AUDIO_DEVICE,
            help="device name to verify is present",
            widget="audio_device",
        ),
    ),
)

_AUDIO_TRACK = Command(
    name="audio-track",
    entry="osc_genai.audio.track:main",
    description="Pitch-track a monophonic audio input to MIDI notes (calibration/verify tool).",
    summary="Pitch-track audio to MIDI notes (calibration tool).",
    kind="oneshot",
    params=(
        Param(
            "device",
            "str",
            DEFAULT_AUDIO_DEVICE,
            help="capture device name (a loopback)",
            widget="audio_device",
        ),
        Param("samplerate", "int", 44100),
        Param("blocksize", "int", 1024, help="capture block size"),
        Param(
            "frame_size",
            "int",
            4096,
            help="YIN analysis window in samples (4096 covers a bass's low E ~41Hz; smaller = lower latency but higher pitch floor)",
        ),
        Param("hop", "int", 512, help="frames advance by this many samples"),
        Param("yin_threshold", "float", 0.15, help="YIN aperiodicity threshold"),
        Param("confidence", "float", 0.5, help="min YIN probability to voice"),
        Param("noise_floor", "float", 0.01, help="min RMS to voice (gate silence)"),
        Param("bpm", "float", 120.0, help="tempo used to timestamp saved notes"),
        Param("seconds", "float", None, help="stop after N seconds (else Ctrl-C)"),
        Param(
            "save",
            "str",
            None,
            help="write the tracked notes to this .mid path",
            widget="file",
        ),
    ),
)

_DEMO = Command(
    name="demo",
    entry="osc_genai:main",
    description="Read a track from Live, generate notes, and write a clip (hardcoded track/slot).",
    summary="Minimal AbletonOSC demo (no flags).",
    kind="oneshot",
    params=(),
)

REGISTRY: dict[str, Command] = {
    cmd.name: cmd
    for cmd in (
        _DUET,
        _PLAY,
        _LIVE,
        _GENERATE,
        _TRAIN,
        _TRAIN_CONDITIONAL,
        _TRAIN_PAIRED,
        _BUILD_PAIRS,
        _FAKE_HUMAN,
        _RECORD,
        _AUDIO_DEVICES,
        _AUDIO_TRACK,
        _DEMO,
    )
}
