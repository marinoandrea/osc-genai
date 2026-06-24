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
| **M1 — rule-based** | `osc-genai-live`: a real-time harmonizer (no model). |
| **M2 — solo model** | A factored per-event GRU trains on `.mid` and generates clips into Live. |
| **M3 — real-time duet** | `osc-genai-duet`: anticipatory scheduling — generates a complementary line *ahead*, plays from a lookahead buffer, and **reconciles** when your recent notes change. Rides **Ableton Link** for shared tempo/grid/transport. |
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
realtime/  clock (WallClock / Ableton-Link), play, duet, scheduler, fake-human, live (M1)
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
uv run osc-genai
```

Against a live set this prints the track count and track 0's name, then a new clip appears in
slot 0 of track 0 with the generated notes. `osc-genai` is the minimal seam — a hardcoded
C-major scale via `core.note.generate_notes()`; swap in a model for real output (below).

### Verify without Ableton

A mock mimics AbletonOSC's ports so you can exercise the full send/receive path with no Live:

```bash
uv run python scripts/mock_ableton.py   # terminal 1
uv run osc-genai                         # terminal 2
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
uv run osc-genai-train --data-dir data/MIDI --out models/acid_v1.pt --epochs 40 --transpose 5
```

**Conditional / paired** — learn to respond in one role given another (e.g. bass → drums):

```bash
# inspect how many aligned chunks the corpus yields
uv run osc-genai-build-pairs --data-dir data/MIDI --context-inst Bass --target-inst Drums

# train directly off the same-song aligned pairs
uv run osc-genai-train-paired --data-dir data/MIDI \
  --context-inst Bass --target-inst Drums --chunk-bars 4 --out models/bass2drums.pt
```

**Generate** a phrase into a Live clip (optionally primed on a context clip to "respond"):

```bash
uv run osc-genai-generate --checkpoint models/acid_v1.pt --track 0 --slot 0 --temperature 0.95
# respond to track 2/slot 0, writing the answer to track 0/slot 1:
uv run osc-genai-generate --checkpoint models/acid_v1.pt \
  --context-track 2 --context-slot 0 --track 0 --slot 1
```

Training data can also be captured straight from Live via `data.midi.capture_from_ableton(...)`.

## Live duet (real-time)

These run over **virtual MIDI ports** (not OSC — that's for clip/LOM control, not low-latency
note streams):

```bash
uv run osc-genai-play --checkpoint models/acid_v1.pt   # one-way: model plays continuously
uv run osc-genai-duet --checkpoint models/acid_v1.pt   # model plays *with* you
uv run osc-genai-duet --checkpoint models/acid_v1.pt --link   # ride Ableton Link tempo/transport
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
uv run osc-genai-duet --checkpoint models/bass2drums.pt --snapshot-bars 4
# flags: --snapshot-dir --snapshot-bars --snapshot-human-inst --snapshot-machine-inst
#        --snapshot-artist --snapshot-key --snapshot-osc-port --snapshot-osc-addr --no-snapshots
```

Snapshot length should match your training chunk size (`--snapshot-bars`, default 4).

### Record a full session

`osc-genai-record` captures the human and machine streams of a session to a paired JSON file
(both parts on a shared timeline) for offline paired training.

### Mock the duet locally (no controller)

`osc-genai-fake-human` loops a MIDI line into the duet's input:

```bash
uv run osc-genai-duet --checkpoint models/acid_v1.pt   # terminal 1: creates ports, listens
uv run osc-genai-fake-human --from-data data/MIDI      # terminal 2: loops one of your clips in
```

Route `osc-genai out` to a synth to hear the response (`--midi FILE`, a built-in acid pattern
when no data is given, and `--bpm` are also available).

## Tests

```bash
uv run pytest
```
