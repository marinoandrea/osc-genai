"""Training the factored event model: batching, the teacher-forced loop, and checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from osc_genai.cli_spec import REGISTRY, build_parser
from osc_genai.core.device import resolve_device
from osc_genai.core.event import Event, notes_to_events
from osc_genai.core.vocab import EventCodec, Fields, VocabConfig
from osc_genai.data.midi import augment, cross_pairs, load_midi_dir
from osc_genai.model.checkpoint import (
    load_model as load_model,  # noqa: F401  (re-exported for callers/tests)
)
from osc_genai.model.checkpoint import save_model
from osc_genai.model.factored import FactoredEventModel, ModelConfig


def collate(
    sequences: list[list[Fields]], eos: Fields
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad EOS-terminated encoded sequences into a batch.

    Returns ``targets`` ``(B, L, 4)`` long and ``mask`` ``(B, L)`` bool (True for real positions,
    including each sequence's terminal EOS). Padding is the EOS tuple so embeddings stay valid; the
    mask keeps padding out of the loss.
    """
    max_len = max(len(s) for s in sequences)
    targets = torch.tensor(eos, dtype=torch.long).repeat(len(sequences), max_len, 1)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        targets[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return targets, mask


def pitch_class_weights(
    encoded: list[list[Fields]], pitch_vocab: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Balanced per-pitch class weights (rare classes up-weighted) from encoded sequences.

    ``weight[p] = total / (n_present_classes * count[p])`` — the count-weighted average is 1, so
    frequent pitches (hi-hat) get <1 and rare ones (kick/snare) get >1.
    """
    counts = torch.zeros(pitch_vocab)
    for seq in encoded:
        for fields in seq:
            counts[fields[0]] += 1
    weights = torch.ones(pitch_vocab)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights.to(device)


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    grad_clip: float = 1.0
    device: str = "auto"
    balance_pitch: bool = False


def train(
    model: FactoredEventModel,
    event_sequences: list[list[Event]],
    codec: EventCodec | None = None,
    config: TrainConfig | None = None,
    log_every: int = 50,
    on_epoch: Callable[[int, float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[float]:
    """Teacher-forced next-event training; returns the per-epoch mean loss history.

    ``on_epoch(epoch, loss)`` is called after each epoch and ``should_stop()`` is polled before each
    one (both optional) so a UI can plot live progress and abort cooperatively mid-run.
    """
    codec = codec or EventCodec(model.vocab)
    config = config or TrainConfig()
    device = resolve_device(config.device)
    model.to(device)
    model.train()

    encoded = [
        codec.encode_sequence(seq, add_eos=True) for seq in event_sequences if seq
    ]
    if not encoded:
        raise ValueError("no (non-empty) training sequences")
    pitch_weights = (
        pitch_class_weights(encoded, model.vocab.pitch_vocab, device)
        if config.balance_pitch
        else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: list[float] = []
    for epoch in range(config.epochs):
        if should_stop is not None and should_stop():
            break
        order = torch.randperm(len(encoded)).tolist()
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = [encoded[i] for i in order[start : start + config.batch_size]]
            targets, mask = collate(batch, codec.eos)
            targets, mask = targets.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = model.loss(targets, mask, pitch_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        history.append(epoch_loss / max(1, batches))
        if on_epoch is not None:
            on_epoch(epoch, history[-1])
        if log_every and (epoch % log_every == 0 or epoch == config.epochs - 1):
            print(f"epoch {epoch:4d}  loss {history[-1]:.4f}")
    return history


def _conditional_collate(
    batch: list[tuple[list[Fields], list[bool]]], eos: Fields
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad (combined-sequence, region-mask) pairs; the mask marks the machine region to train on."""
    max_len = max(len(target) for target, _ in batch)
    targets = torch.tensor(eos, dtype=torch.long).repeat(len(batch), max_len, 1)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (target, region) in enumerate(batch):
        targets[i, : len(target)] = torch.tensor(target, dtype=torch.long)
        mask[i, : len(region)] = torch.tensor(region, dtype=torch.bool)
    return targets, mask


def train_conditional(
    model: FactoredEventModel,
    pairs: list[tuple[list[Event], list[Event]]],
    codec: EventCodec | None = None,
    config: TrainConfig | None = None,
    log_every: int = 50,
    on_epoch: Callable[[int, float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[float]:
    """Train the model to generate the *machine* part given the *human* part.

    Each pair ``(human, machine)`` is encoded as one sequence ``human + machine + EOS``; loss is
    taken only over the machine region (+EOS), so the human part is conditioning, not a target. At
    inference ``generate(context=human)`` then produces a complementary response.
    """
    codec = codec or EventCodec(model.vocab)
    config = config or TrainConfig()
    device = resolve_device(config.device)
    model.to(device)
    model.train()

    encoded: list[tuple[list[Fields], list[bool]]] = []
    for human, machine in pairs:
        human_fields = codec.encode_sequence(human, add_eos=False)
        machine_fields = codec.encode_sequence(machine, add_eos=True)
        if machine_fields:
            encoded.append(
                (
                    human_fields + machine_fields,
                    [False] * len(human_fields) + [True] * len(machine_fields),
                )
            )
    if not encoded:
        raise ValueError("no (non-empty) training pairs")

    pitch_weights = (
        pitch_class_weights(
            [
                [
                    fields
                    for fields, is_target in zip(tgt, reg, strict=True)
                    if is_target
                ]
                for tgt, reg in encoded
            ],
            model.vocab.pitch_vocab,
            device,
        )
        if config.balance_pitch
        else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[float] = []
    for epoch in range(config.epochs):
        if should_stop is not None and should_stop():
            break
        order = torch.randperm(len(encoded)).tolist()
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = [encoded[i] for i in order[start : start + config.batch_size]]
            targets, mask = _conditional_collate(batch, codec.eos)
            targets, mask = targets.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = model.loss(targets, mask, pitch_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        history.append(epoch_loss / max(1, batches))
        if on_epoch is not None:
            on_epoch(epoch, history[-1])
        if log_every and (epoch % log_every == 0 or epoch == config.epochs - 1):
            print(f"epoch {epoch:4d}  loss {history[-1]:.4f}")
    return history


def main() -> None:
    args = build_parser(REGISTRY["train"]).parse_args()

    sequences = [s for s in load_midi_dir(args.data_dir) if s]
    print(f"loaded {len(sequences)} non-empty sequence(s) from {args.data_dir}")
    if args.transpose:
        sequences = augment(
            sequences, semitones=range(-args.transpose, args.transpose + 1)
        )
        print(
            f"after +/-{args.transpose} semitone transposition: {len(sequences)} sequence(s)"
        )

    event_sequences = [
        notes_to_events(s, steps_per_beat=args.steps_per_beat) for s in sequences
    ]
    vocab = VocabConfig()
    model = FactoredEventModel(
        vocab, ModelConfig(hidden_size=args.hidden, num_layers=args.layers)
    )
    train(
        model,
        event_sequences,
        codec=EventCodec(vocab),
        config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            balance_pitch=args.balance_pitch,
        ),
        log_every=1,
    )
    save_model(model, args.out)
    print(f"saved checkpoint to {args.out}")


def conditional_main() -> None:
    args = build_parser(REGISTRY["train-conditional"]).parse_args()

    context = [s for d in args.context_dir for s in load_midi_dir(d) if s]
    if args.target_drums:
        from osc_genai.data.drums import load_drum_kits

        target = load_drum_kits(args.target_dir)
    else:
        target = [s for d in args.target_dir for s in load_midi_dir(d) if s]
    print(f"context: {len(context)} clips from {args.context_dir}")
    print(
        f"target:  {len(target)} clips ({'drum kits, GM-normalized' if args.target_drums else 'raw'})"
    )
    note_pairs = cross_pairs(context, target, k=args.pairs_per_context, seed=args.seed)
    event_pairs = [
        (
            notes_to_events(c, steps_per_beat=args.steps_per_beat),
            notes_to_events(t, steps_per_beat=args.steps_per_beat),
        )
        for c, t in note_pairs
    ]
    print(f"training pairs: {len(event_pairs)}")
    vocab = VocabConfig()
    model = FactoredEventModel(
        vocab, ModelConfig(hidden_size=args.hidden, num_layers=args.layers)
    )
    train_conditional(
        model,
        event_pairs,
        codec=EventCodec(vocab),
        config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            balance_pitch=args.balance_pitch,
        ),
        log_every=1,
    )
    save_model(model, args.out)
    print(f"saved {args.context_dir} -> {args.target_dir} snapshot to {args.out}")


def paired_main() -> None:
    args = build_parser(REGISTRY["train-paired"]).parse_args()

    from osc_genai.data.pairs import (
        augment_pairs,
        build_aligned_pairs,
        interleave_pairs,
        note_pairs,
    )

    sizes = [args.chunk_bars] + ([8] if args.also_8 and args.chunk_bars != 8 else [])
    aligned = build_aligned_pairs(
        args.data_dir,
        args.context_inst,
        args.target_inst,
        chunk_bars=args.chunk_bars,
        hop_bars=args.hop_bars,
        sizes=sizes,
        normalize_drums=not args.no_normalize_drums,
        regularize=args.regular_grid if args.regular_drums else None,
    )
    songs = {p.song for p in aligned}
    print(
        f"{args.context_inst} -> {args.target_inst}: {len(aligned)} aligned chunks "
        f"from {len(songs)} songs (sizes={sizes} bars)"
    )

    pairs = note_pairs(aligned)
    if args.transpose:
        pairs = augment_pairs(
            pairs,
            semitones=range(-args.transpose, args.transpose + 1),
            target_is_drums=args.target_inst.lower() == "drums",
        )
        print(
            f"after +/-{args.transpose} semitone context transposition: {len(pairs)} pairs"
        )

    vocab = VocabConfig(
        use_phase=args.phase, steps_per_bar=args.beats_per_bar * args.steps_per_beat
    )
    model = FactoredEventModel(
        vocab, ModelConfig(hidden_size=args.hidden, num_layers=args.layers)
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        balance_pitch=args.balance_pitch,
    )
    codec = EventCodec(vocab)
    if args.interleaved:
        sequences = interleave_pairs(pairs, args.steps_per_beat)
        print(f"interleaved into {len(sequences)} time-aligned two-stream sequences")
        train(model, sequences, codec=codec, config=config, log_every=1)
    else:
        event_pairs = [
            (
                notes_to_events(c, steps_per_beat=args.steps_per_beat),
                notes_to_events(t, steps_per_beat=args.steps_per_beat),
            )
            for c, t in pairs
        ]
        train_conditional(model, event_pairs, codec=codec, config=config, log_every=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out)
    print(f"saved {args.context_inst} -> {args.target_inst} model to {args.out}")


if __name__ == "__main__":
    main()
