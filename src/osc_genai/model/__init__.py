"""The factored per-event recurrent model and its checkpoint IO.

``from osc_genai.model import FactoredEventModel, load_model``.
"""

from osc_genai.model.factored import FactoredEventModel, ModelConfig
from osc_genai.model.checkpoint import load_model, save_model

__all__ = ["FactoredEventModel", "ModelConfig", "load_model", "save_model"]
