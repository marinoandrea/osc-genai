# osc-genai

A from-scratch generative-MIDI system for Ableton Live: a small per-event neural model learns
from a MIDI corpus, writes clips into Live over OSC, and plays a **real-time anticipatory duet**
with you over virtual MIDI ports.

No pretrained weights and no `pylive` wrapper — clip control talks **directly** to
[AbletonOSC](https://github.com/ideoforms/AbletonOSC) via
[`python-osc`](https://pypi.org/project/python-osc/), and the live duet streams raw MIDI for
low latency. (`pylive` is a convenience layer over AbletonOSC and, per its own docs, isn't meant
for writing notes — note-writing goes through AbletonOSC's endpoints regardless, so we use them
directly.)

## Status

| Stage | What works |
|------|------------|
| **M1 — rule-based** | `live`: a real-time harmonizer (no model). |
| **M2 — solo model** | A factored per-event GRU trains on `.mid` and generates clips into Live. |
| **M3 — real-time duet** | `duet`: anticipatory scheduling — generates a complementary line *ahead*, plays from a lookahead buffer, and **reconciles** when your recent notes change. Rides **Ableton Link** for shared tempo/grid/transport. |
| **M4 — paired data** | Capture genuine `(human, machine)` material live: session recording, and **snapshot save** (grab the last N bars straight into the dataset). The paired-training pipeline (`build-pairs` + `train-paired`) is in place; *training on captured pairs is not yet validated*. |

The scheduling is already anticipatory; the shipped model is still **solo-trained**, so it
follows rather than jointly predicts you. Closing that gap is what the M4 paired data is for.

## How it fits together

The code is organised into layered packages under `src/osc_genai/`:

```
core/      note + event representation, vocab/codec (Note <-> factored Event <-> model indices)
model/     factored per-event GRU + per-field heads, streaming sample API, checkpoints
data/      MIDI I/O, augmentation, drums normalize, aligned instrument pairs, record, snapshot
training/  teacher-forced training loops (solo / conditional / paired) + metrics
inference.py   model -> Note phrases, generate-into-Live CLI
realtime/  clock (WallClock / Ableton-Link), play, duet, scheduler, fake-human, live (M1), partner inputs
audio/     YIN pitch tracker + note segmentation + device capture (real instrument -> notes, monophonic)
osc/       AbletonOSC client (send/query clips) + OSC trigger listener
cli.py     the minimal plumbing demo (read track, generate, create clip, add notes)
```

AbletonOSC listens on UDP **11000** (commands) and replies on UDP **11001**.

## Setup

```bash
uv sync                 # core install
uv sync --extra link    # + Ableton Link support (native aalink build) for the duet's --link
```

### Install AbletonOSC into Ableton Live (required for the clip path)

1. Ableton Live 11 or 12.
2. Clone AbletonOSC into the Remote Scripts directory — on macOS:
   ```bash
   git clone https://github.com/ideoforms/AbletonOSC.git \
     ~/Music/Ableton/User\ Library/Remote\ Scripts/AbletonOSC
   ```
3. In Live: *Preferences → Link/Tempo/MIDI*, set a **Control Surface** to `AbletonOSC`.

## Quick start (plumbing demo)

```bash
uv run demo
```

Against a live set this prints the track count and track 0's name, then a new clip appears in
slot 0 of track 0 with the generated notes. `demo` is the minimal seam — a hardcoded
C-major scale via `core.note.generate_notes()`; swap in a model for real output (below).

### Verify without Ableton

A mock mimics AbletonOSC's ports so you can exercise the full send/receive path with no Live:

```bash
uv run python scripts/mock_ableton.py   # terminal 1
uv run demo                              # terminal 2
```

The mock **refuses to start if a real AbletonOSC is answering on 11000** (so an open Live set
can't silently receive test writes). Use `--recv-port`/`--reply-port` for isolated ports, or
`--force`. The automated tests (`uv run pytest`) use private ports and never touch 11000/11001.

## Dataset layout

Training data lives as single-instrument clips:

```
data/MIDI/<Instrument>/<Artist>/<source_song>__<label>.mid
```

The prefix before `__` is the song id; two instruments of the *same song* (e.g. `Bass/` and
`Drums/`) are what the paired pipeline aligns into `(context, target)` chunks. `data/` is
gitignored (only the folder is kept). `scripts/fetch_midm_database.py` reproducibly fetches and
organises a corpus into this layout.

## Model pipeline (train, generate into Live)

A note is four factored fields — pitch, onset-Δ, duration, velocity (plus a duet `source` tag) —
and a GRU predicts the next note one event at a time (O(1) per event, which the live duet needs).

**Solo** — train on a folder of `.mid` (recursive, with transposition augmentation):

```bash
uv run train --data-dir data/MIDI --out models/acid_v1.pt --epochs 40 --transpose 5
```

**Conditional / paired** — learn to respond in one role given another (e.g. bass → drums):

```bash
# inspect how many aligned chunks the corpus yields
uv run build-pairs --data-dir data/MIDI --context-inst Bass --target-inst Drums

# train directly off the same-song aligned pairs
uv run train-paired --data-dir data/MIDI \
  --context-inst Bass --target-inst Drums --chunk-bars 4 --out models/bass2drums.pt
```

**Generate** a phrase into a Live clip (optionally primed on a context clip to "respond"):

```bash
uv run generate --checkpoint models/acid_v1.pt --track 0 --slot 0 --temperature 0.95
# respond to track 2/slot 0, writing the answer to track 0/slot 1:
uv run generate --checkpoint models/acid_v1.pt \
  --context-track 2 --context-slot 0 --track 0 --slot 1
```

Training data can also be captured straight from Live via `data.midi.capture_from_ableton(...)`.

## Live duet (real-time)

These run over **virtual MIDI ports** (not OSC — that's for clip/LOM control, not low-latency
note streams):

```bash
uv run play --checkpoint models/acid_v1.pt   # one-way: model plays continuously
uv run duet --checkpoint models/acid_v1.pt   # model plays *with* you
uv run duet --checkpoint models/acid_v1.pt --link   # ride Ableton Link tempo/transport
```

Route in Ableton: enable **`osc-genai out`** as a synth track's *MIDI From*, and for the duet
**`osc-genai in`** as your controller's *MIDI To*. The macOS **IAC Driver** works too. Tune the
anticipation with `--lookahead`, `--commit-horizon`, `--chunk-events`.

### Snapshot save — keep the last N bars as training data

During a duet, capture a passage you like straight into the dataset. A snapshot grabs the last
**N bars of both parts** and writes them as two aligned `.mid` clips sharing a song id under a
`personal` artist (`Bass/personal/` + `Drums/personal/`) — exactly the shape the paired pipeline
reconstructs pairs from, so it becomes training data with no extra steps.

Trigger it two ways while the duet runs:

* **Keyboard** — press `s` then Enter.
* **OSC** — send a bang to `/snapshot` on UDP **11002** (e.g. from a Max for Live device or
  TouchOSC), so a controller/footswitch can capture hands-free.

```bash
uv run duet --checkpoint models/bass2drums.pt --snapshot-bars 4
# flags: --snapshot-dir --snapshot-bars --snapshot-human-inst --snapshot-machine-inst
#        --snapshot-artist --snapshot-key --snapshot-osc-port --snapshot-osc-addr --no-snapshots
```

Snapshot length should match your training chunk size (`--snapshot-bars`, default 4).

### Audio ingestion — play a real (non-MIDI) instrument

Instead of a MIDI controller, the duet partner can be a **real instrument captured as audio**: a
bass goes into a virtual loopback device, we run our own **YIN** pitch tracker on it on the fly
(`audio/yin.py`), segment the pitch into notes, and feed those into the *same* duet — so generation
and snapshot-to-dataset work unchanged.

> **Monophonic only.** YIN estimates one fundamental per frame, so this tracks a single-note line.
> If the bass plays a **chord**, only its strongest/lowest note is captured.

The Live API exposes no PCM, so we don't query Ableton for samples — we tap the track's output at
signal level. One-time setup (macOS):

```bash
brew install blackhole-2ch     # a virtual loopback audio device (or use Loopback / an Aggregate Device)
sudo killall coreaudiod        # reload CoreAudio so the device appears
uv run audio-devices           # verify it's present (defaults to "BlackHole 2ch")
```

In Ableton, route the bass track's **Audio To → BlackHole 2ch** (an *Aggregate Device* combining
your interface + BlackHole lets you keep monitoring). Then calibrate the tracker on its own —
no model needed — until single notes read out correctly:

```bash
uv run audio-track --device "BlackHole 2ch"      # prints detected notes live
uv run audio-track --device "BlackHole 2ch" --save take.mid   # also dump to MIDI
# tune: --frame-size (4096 covers a bass low E; smaller = lower latency, higher pitch floor),
#       --confidence, --noise-floor, --yin-threshold, --hop
```

Then run the duet straight off the audio:

```bash
uv run duet --checkpoint models/bass2drums.pt --audio-in --link
# --audio-device, --audio-samplerate, --frame-size, --hop, --confidence, --noise-floor,
# --yin-threshold, --audio-echo (monitor the tracked MIDI). Snapshot flags work as usual.
```

### Record a full session

`record` captures the human and machine streams of a session to a paired JSON file
(both parts on a shared timeline) for offline paired training.

### Mock the duet locally (no controller)

`fake-human` loops a MIDI line into the duet's input:

```bash
uv run duet --checkpoint models/acid_v1.pt   # terminal 1: creates ports, listens
uv run fake-human --from-data data/MIDI      # terminal 2: loops one of your clips in
```

Route `osc-genai out` to a synth to hear the response (`--midi FILE`, a built-in acid pattern
when no data is given, and `--bpm` are also available).

## Tests

```bash
uv run pytest
```
