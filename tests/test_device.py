"""Device selection: resolve_device's auto-detection and load_model placing weights on a device."""

from __future__ import annotations

import torch

from osc_genai.core.device import resolve_device
from osc_genai.core.vocab import VocabConfig
from osc_genai.model.checkpoint import load_model, save_model
from osc_genai.model.factored import FactoredEventModel, ModelConfig


def test_explicit_spec_is_honoured():
    assert resolve_device("cpu").type == "cpu"


def test_auto_returns_a_valid_backend():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda", "mps"}


def test_auto_prefers_accelerator_when_present():
    device = resolve_device("auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_load_model_places_weights_on_device(tmp_path):
    torch.manual_seed(0)
    model = FactoredEventModel(
        VocabConfig(), ModelConfig(embed_dim=16, hidden_size=32, num_layers=1)
    )
    path = tmp_path / "model.pt"
    save_model(model, path)

    device = resolve_device("auto")
    reloaded = load_model(path, device=str(device))
    assert next(reloaded.parameters()).device.type == device.type
    # Generation exercises the GRU + sampling on whatever backend auto picked.
    assert isinstance(reloaded.generate(max_events=4), list)
