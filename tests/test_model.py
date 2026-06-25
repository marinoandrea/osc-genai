"""Tests for the factored per-event model: shapes, loss/backprop, and generation validity."""

from __future__ import annotations

import torch

from osc_genai.core.event import Event
from osc_genai.core.vocab import EventCodec, VocabConfig
from osc_genai.model.factored import FactoredEventModel, ModelConfig


def make_model() -> FactoredEventModel:
    return FactoredEventModel(
        VocabConfig(max_dt=16, max_dur=16, velocity_bins=8),
        ModelConfig(embed_dim=16, hidden_size=32, num_layers=1),
    )


def _random_targets(model: FactoredEventModel, batch: int, length: int) -> torch.Tensor:
    sizes = model.vocab.field_sizes
    return torch.stack([torch.randint(0, s, (batch, length)) for s in sizes], dim=-1)


def test_forward_shapes():
    torch.manual_seed(0)
    model = make_model()
    targets = _random_targets(model, batch=2, length=5)
    logits = model(targets)
    assert len(logits) == len(
        model.vocab.field_sizes
    )  # pitch, dt, dur, velocity, channel, source
    for field_logits, size in zip(logits, model.vocab.field_sizes, strict=True):
        assert field_logits.shape == (2, 5, size)


def test_loss_is_scalar_and_backprops():
    torch.manual_seed(0)
    model = make_model()
    targets = _random_targets(model, batch=2, length=4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss = model.loss(targets, mask)
    assert loss.ndim == 0 and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_generate_returns_valid_fields():
    torch.manual_seed(0)
    model = make_model()
    out = model.generate(max_events=10, temperature=1.0)
    assert isinstance(out, list)
    for fields in out:
        assert len(fields) == len(model.vocab.field_sizes)
        for index, size in zip(fields, model.vocab.field_sizes, strict=True):
            assert 0 <= index < size
        assert (
            fields[0] != model.vocab.eos_pitch
        )  # EOS is the stop signal, never emitted


def test_generate_with_context_runs():
    torch.manual_seed(0)
    model = make_model()
    codec = EventCodec(model.vocab)
    context = codec.encode_sequence(
        [Event(60, 0, 4, 100), Event(62, 4, 4, 100)], add_eos=False
    )
    out = model.generate(context=context, max_events=8)
    assert isinstance(out, list)


def test_streaming_matches_generate():
    """generate() must be exactly the streaming API (fresh_state/observe/sample_next) under the hood."""
    model = make_model()
    codec = EventCodec(model.vocab)
    context = codec.encode_sequence([Event(60, 0, 4, 100)], add_eos=False)

    torch.manual_seed(0)
    expected = model.generate(context=context, max_events=6, temperature=1.0)

    torch.manual_seed(0)
    state = model.fresh_state()
    for fields in context:
        state = model.observe(state, fields)
    streamed = []
    for _ in range(6):
        fields, state = model.sample_next(state, temperature=1.0)
        if fields[0] == model.vocab.eos_pitch:
            break
        streamed.append(fields)
    assert streamed == expected


def test_phase_model_trains_and_generates():
    """With use_phase the input widens by one embedding; heads/loss are unchanged (6 predicted fields)."""
    torch.manual_seed(0)
    vocab = VocabConfig(
        max_dt=16, max_dur=16, velocity_bins=8, use_phase=True, steps_per_bar=16
    )
    model = FactoredEventModel(vocab, ModelConfig(embed_dim=16, hidden_size=32))
    assert model.rnn.input_size == 16 * (
        len(vocab.field_sizes) + 1
    )  # +1 for the phase feature

    targets = _random_targets(model, batch=2, length=5)
    logits = model(targets)
    assert len(logits) == len(vocab.field_sizes)  # phase is not a predicted head

    mask = torch.ones(2, 5, dtype=torch.bool)
    model.loss(targets, mask).backward()
    out = model.generate(max_events=8, temperature=1.0)
    assert all(len(f) == len(vocab.field_sizes) for f in out)


def test_phase_is_dt_cumulative_mod_bar():
    vocab = VocabConfig(use_phase=True, steps_per_bar=16)
    model = FactoredEventModel(vocab, ModelConfig(embed_dim=8, hidden_size=16))
    # dt column [3, 5, 10] -> onsets [3, 8, 18] -> phase [3, 8, 2] (mod 16)
    dt = torch.tensor([[3, 5, 10]])
    onset = torch.cumsum(dt, dim=1)
    assert model._phase_of(onset).tolist() == [[3, 8, 2]]


def test_sample_next_force_pins_a_field():
    """force={source_field: SELF} must always emit that source, whatever the model would sample."""
    from osc_genai.core.event import SELF

    model = make_model()
    source_field = len(model.vocab.field_sizes) - 1
    state = model.fresh_state()
    for _ in range(20):
        fields, state = model.sample_next(
            state, temperature=1.0, force={source_field: SELF}
        )
        assert fields[source_field] == SELF
