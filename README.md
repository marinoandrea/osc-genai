# osc-genai

A Python foundation for feeding (eventually ML-generated) MIDI into Ableton Live over OSC.

It talks **directly** to [AbletonOSC](https://github.com/ideoforms/AbletonOSC) using
[`python-osc`](https://pypi.org/project/python-osc/) — no `pylive` wrapper. (`pylive` is a
convenience layer over AbletonOSC and, per its own docs, isn't meant for writing MIDI
notes; note-writing goes through AbletonOSC's OSC endpoints regardless, so we use them
directly.)

This first iteration is plumbing: read track 0 from Live, generate notes (currently a
hardcoded C-major scale standing in for a model), create a clip, and write the notes into
it.

## How it fits together

```
generate.py   ->  generate_notes() returns list[Note]   (the "ML magic" seam)
ableton.py    ->  AbletonOSC: send commands + block on query replies over OSC
main.py       ->  read track 0, generate, create clip, add notes
```

AbletonOSC listens on UDP **11000** (commands) and replies on UDP **11001**.

## Setup

```bash
uv sync
```

### Install AbletonOSC into Ableton Live (required for the real run)

1. Ableton Live 11 or 12.
2. Clone AbletonOSC into the Remote Scripts directory — on macOS:
   `~/Music/Ableton/User Library/Remote Scripts/AbletonOSC`
   ```bash
   git clone https://github.com/ideoforms/AbletonOSC.git \
     ~/Music/Ableton/User\ Library/Remote\ Scripts/AbletonOSC
   ```
3. In Live: *Preferences → Link/Tempo/MIDI*, set a **Control Surface** to `AbletonOSC`.

## Run

```bash
uv run osc-genai
```

Expected against a live set: prints the track count and track 0's name, then a new clip
appears in slot 0 of track 0 containing the generated notes. Set `FIRE_AFTER_WRITE = True`
in `main.py` to auto-play it.

## Verify without Ableton

A mock that mimics AbletonOSC's ports lets you exercise the full send/receive path with no
Ableton:

```bash
# terminal 1
uv run python scripts/mock_ableton.py
# terminal 2
uv run osc-genai
```

The mock logs the `create_clip` / `add/notes` messages it receives and answers the
track-name and track-count queries.

## Tweaking

- `TRACK_INDEX` / `CLIP_SLOT` in `main.py` — which track/slot to write (hardcoded to 0/0).
- `generate_notes()` in `generate.py` — replace the hardcoded melody with a model.
