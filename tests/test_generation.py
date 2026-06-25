"""Tests for the model -> Note phrase bridge."""

from __future__ import annotations

import torch

from osc_genai.core.event import notes_to_events
from osc_genai.core.note import Note
from osc_genai.core.vocab import VocabConfig
from osc_genai.inference import generate_phrase
from osc_genai.model.factored import FactoredEventModel, ModelConfig
from osc_genai.training.train import TrainConfig, train

# velocity 100 is a 16-bin centre; with steps_per_beat=4 the onsets/durations are on-grid.
NOTES = [Note(60, 0.0, 0.5, 100), Note(62, 0.5, 0.5, 100), Note(64, 1.0, 0.5, 100)]


def test_generate_phrase_reproduces_trained_pattern():
    torch.manual_seed(0)
    vocab = VocabConfig(max_dt=8, max_dur=8, velocity_bins=16)
    model = FactoredEventModel(
        vocab, ModelConfig(embed_dim=16, hidden_size=64, num_layers=1)
    )
    train(
        model,
        [notes_to_events(NOTES)] * 8,
        config=TrainConfig(epochs=400, batch_size=8, lr=5e-3),
        log_every=0,
    )
    assert generate_phrase(model, temperature=0.0, max_events=10) == NOTES


def test_generate_phrase_with_context_runs():
    torch.manual_seed(0)
    model = FactoredEventModel(
        VocabConfig(max_dt=8, max_dur=8, velocity_bins=16),
        ModelConfig(embed_dim=16, hidden_size=32),
    )
    out = generate_phrase(
        model, context=[Note(60, 0.0, 0.5, 100)], max_events=5, temperature=1.0
    )
    assert isinstance(out, list)
