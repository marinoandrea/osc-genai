"""The conditional trainer learns to produce the machine part given the human part."""

from __future__ import annotations

import torch

from osc_genai.core.event import SELF, notes_to_events
from osc_genai.core.note import Note
from osc_genai.core.vocab import EventCodec, VocabConfig
from osc_genai.data.midi import transpose
from osc_genai.data.pairs import interleave, interleave_pairs
from osc_genai.inference import generate_phrase
from osc_genai.model.factored import FactoredEventModel, ModelConfig
from osc_genai.training.train import TrainConfig, train, train_conditional


def test_conditional_learns_a_complement():
    # Synthetic complement: machine = human transposed up a fifth (velocity 100 = 16-bin centre).
    torch.manual_seed(0)
    vocab = VocabConfig(max_dt=8, max_dur=8, velocity_bins=16)
    model = FactoredEventModel(vocab, ModelConfig(embed_dim=16, hidden_size=64))
    human = [Note(50, 0.0, 0.5, 100), Note(52, 0.5, 0.5, 100), Note(54, 1.0, 0.5, 100)]
    machine = transpose(human, 7)

    he, me = notes_to_events(human), notes_to_events(machine)
    history = train_conditional(
        model,
        [(he, me)] * 8,
        config=TrainConfig(epochs=400, batch_size=8, lr=5e-3),
        log_every=0,
    )
    assert history[-1] < history[0]

    # Conditioned on the human, greedy generation reproduces the trained complement.
    assert (
        generate_phrase(model, context=human, temperature=0.0, max_events=10) == machine
    )


def test_interleaved_joint_duet_overfits():
    # Time-aligned duet: bass (PARTNER) on beats 0-1, drums (SELF) on beats 2-3. Interleaved into one
    # source-tagged stream and trained jointly. Priming the streaming state with the observed bass
    # and forcing SELF must recover the drum line — the live-duet inference path.
    torch.manual_seed(0)
    vocab = VocabConfig(max_dt=8, max_dur=8, velocity_bins=16)
    model = FactoredEventModel(vocab, ModelConfig(embed_dim=16, hidden_size=64))
    codec = EventCodec(vocab)

    bass = [Note(40, 0.0, 1.0, 100, False, 0), Note(43, 1.0, 1.0, 100, False, 0)]
    drums = [Note(36, 2.0, 1.0, 100, False, 9), Note(38, 3.0, 1.0, 100, False, 9)]
    sequences = interleave_pairs([(bass, drums)] * 8)

    history = train(
        model,
        sequences,
        codec=codec,
        config=TrainConfig(epochs=500, batch_size=8, lr=5e-3),
        log_every=0,
    )
    assert history[-1] < history[0]

    # Stream the observed bass into the state, then greedily emit SELF events.
    source_field = len(vocab.field_sizes) - 1
    state = model.fresh_state()
    for ev in interleave(bass, []):  # PARTNER events, dt on the shared clock
        state = model.observe(state, codec.encode(ev))
    emitted = []
    for _ in range(6):
        fields, state = model.sample_next(
            state, temperature=0.0, force={source_field: SELF}
        )
        if fields[0] == vocab.eos_pitch:
            break
        emitted.append(codec.decode(fields))
    assert [(e.pitch, e.source) for e in emitted] == [(36, SELF), (38, SELF)]
