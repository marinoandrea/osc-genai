"""Orchestration entry point.

Wires the pieces together: read the track of interest from Live, ask the (stubbed) model
for notes, then create a clip and write the notes into it via OSC.
"""

from __future__ import annotations

from osc_genai.core.note import generate_notes, total_beats
from osc_genai.osc.ableton import AbletonOSC

# Hardcoded for now — the track and clip slot we operate on.
TRACK_INDEX = 0
CLIP_SLOT = 0

# Whether to start playback of the clip once it's written.
FIRE_AFTER_WRITE = False


def main() -> None:
    with AbletonOSC() as live:
        # 1. Retrieve the track of interest.
        num_tracks = live.get_num_tracks()
        print(f"Live set has {num_tracks} track(s).")
        if TRACK_INDEX >= num_tracks:
            raise SystemExit(
                f"Track {TRACK_INDEX} does not exist (only {num_tracks} track(s))."
            )
        track_name = live.get_track_name(TRACK_INDEX)
        print(f"Target track {TRACK_INDEX}: {track_name!r}")

        # 2. 'ML magic' — produce the notes.
        notes = generate_notes()
        length = total_beats(notes)
        print(f"Generated {len(notes)} note(s) spanning {length} beat(s).")

        # 3. Create a clip and write the notes into it.
        live.create_clip(TRACK_INDEX, CLIP_SLOT, length)
        live.add_notes(TRACK_INDEX, CLIP_SLOT, notes)
        print(f"Wrote clip to track {TRACK_INDEX}, slot {CLIP_SLOT}.")

        if FIRE_AFTER_WRITE:
            live.fire_clip(TRACK_INDEX, CLIP_SLOT)
            print("Fired clip.")


if __name__ == "__main__":
    main()
