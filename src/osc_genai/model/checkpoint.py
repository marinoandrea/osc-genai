"""Persist and restore a :class:`FactoredEventModel` — weights plus the configs to rebuild it.

Kept beside the model (not in :mod:`training`) so the realtime/inference commands can load a
checkpoint without importing the trainer.
"""

from __future__ import annotations

from pathlib import Path

import torch

from osc_genai.core.vocab import VocabConfig
from osc_genai.model.factored import FactoredEventModel, ModelConfig


def save_model(model: FactoredEventModel, path: str | Path) -> None:
    """Persist weights + the vocab/model configs needed to rebuild the module."""
    torch.save(
        {
            "vocab": vars(model.vocab),
            "config": vars(model.config),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: str | Path, map_location: str = "cpu") -> FactoredEventModel:
    """Inverse of :func:`save_model`."""
    checkpoint = torch.load(path, map_location=map_location)
    model = FactoredEventModel(
        VocabConfig(**checkpoint["vocab"]), ModelConfig(**checkpoint["config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
