"""Turn a trained model into ``Note`` phrases, and respond to a context phrase.

:func:`generate_phrase` is the bridge from model field-indices back to clip ``Note``s: optionally
primed on a *context* phrase (the musician's call read from a clip), it samples a response. The CLI
(``generate``) loads a checkpoint, optionally reads a context clip out of Ableton, and
writes the generated response into a clip — the offline / clip-based duet. The *real-time* model
responder (turn-taking / simultaneous, with scheduling) lands in M3 alongside the scheduler.
"""

from __future__ import annotations

from osc_genai.cli_spec import REGISTRY, build_parser
from osc_genai.core.event import (
    DEFAULT_STEPS_PER_BEAT,
    events_to_notes,
    notes_to_events,
)
from osc_genai.core.note import Note, total_beats
from osc_genai.core.vocab import EventCodec
from osc_genai.model.checkpoint import load_model
from osc_genai.model.factored import FactoredEventModel
from osc_genai.osc.ableton import AbletonOSC


def generate_phrase(
    model: FactoredEventModel,
    *,
    context: list[Note] | None = None,
    codec: EventCodec | None = None,
    steps_per_beat: int = DEFAULT_STEPS_PER_BEAT,
    max_events: int = 64,
    temperature: float = 1.0,
    pitch_bias: dict[int, float] | None = None,
) -> list[Note]:
    """Sample a phrase as clip ``Note``s, optionally primed on a ``context`` phrase."""
    codec = codec or EventCodec(model.vocab)
    context_fields = None
    if context:
        context_events = notes_to_events(context, steps_per_beat=steps_per_beat)
        context_fields = codec.encode_sequence(context_events, add_eos=False)
    generated = model.generate(
        context=context_fields,
        max_events=max_events,
        temperature=temperature,
        pitch_bias=pitch_bias,
    )
    events = codec.decode_sequence(generated)
    return events_to_notes(events, steps_per_beat=steps_per_beat)


def main() -> None:
    args = build_parser(REGISTRY["generate"]).parse_args()

    model = load_model(args.checkpoint, device=args.device)
    with AbletonOSC() as live:
        context = None
        if args.context_track is not None and live.has_clip(
            args.context_track, args.context_slot
        ):
            context = live.get_clip_notes(args.context_track, args.context_slot)
            print(f"primed on {len(context)} context note(s)")
        notes = generate_phrase(
            model,
            context=context,
            steps_per_beat=args.steps_per_beat,
            max_events=args.max_events,
            temperature=args.temperature,
        )
        if not notes:
            print("model produced no notes (immediate EOS) — try a higher temperature.")
            return
        length = total_beats(notes)
        live.create_clip(args.track, args.slot, length)
        live.add_notes(args.track, args.slot, notes)
        print(f"wrote {len(notes)} note(s) to track {args.track}, slot {args.slot}.")


if __name__ == "__main__":
    main()
