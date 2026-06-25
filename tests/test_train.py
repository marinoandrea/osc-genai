"""Training tests, incl. the end-to-end overfit smoke test (the pipeline really learns)."""

from __future__ import annotations

import torch

from osc_genai.core.event import Event
from osc_genai.core.vocab import EventCodec, VocabConfig
from osc_genai.model.factored import FactoredEventModel, ModelConfig
from osc_genai.training.train import TrainConfig, collate, load_model, save_model, train

# velocity 100 is a 16-bin centre, so it round-trips exactly through the codec.
PATTERN = [
    Event(60, 0, 2, 100),
    Event(62, 2, 2, 100),
    Event(64, 2, 2, 100),
    Event(65, 2, 2, 100),
]


def test_collate_pads_and_masks():
    codec = EventCodec(VocabConfig())
    s1 = codec.encode_sequence([Event(60, 0, 2, 100)])  # 1 event + EOS = len 2
    s2 = codec.encode_sequence([Event(60, 0, 2, 100), Event(62, 2, 2, 100)])  # len 3
    targets, mask = collate([s1, s2], codec.eos)
    assert targets.shape == (2, 3, 6)  # pitch, dt, dur, velocity, channel, source
    assert mask[0].tolist() == [True, True, False]
    assert mask[1].tolist() == [True, True, True]
    assert tuple(targets[0, 2].tolist()) == codec.eos  # padding is EOS


def test_pitch_class_weights_upweight_rare():
    from osc_genai.training.train import pitch_class_weights

    codec = EventCodec(VocabConfig())
    seq = codec.encode_sequence(
        [Event(60, 0, 1, 100)] * 8 + [Event(64, 0, 1, 100)], add_eos=False
    )
    weights = pitch_class_weights([seq], codec.config.pitch_vocab)
    assert (
        weights[64] > weights[60]
    )  # the rare pitch is up-weighted relative to the frequent one


def _overfit_model() -> tuple[FactoredEventModel, VocabConfig]:
    torch.manual_seed(0)
    vocab = VocabConfig(max_dt=8, max_dur=8, velocity_bins=16)
    model = FactoredEventModel(
        vocab, ModelConfig(embed_dim=16, hidden_size=64, num_layers=1)
    )
    history = train(
        model,
        [PATTERN] * 8,
        config=TrainConfig(epochs=400, batch_size=8, lr=5e-3),
        log_every=0,
    )
    assert history[-1] < history[0], "loss did not decrease"
    assert history[-1] < 0.2, f"loss did not collapse: {history[-1]}"
    return model, vocab


def test_overfit_reproduces_pattern():
    model, vocab = _overfit_model()
    codec = EventCodec(vocab)
    generated = model.generate(temperature=0.0, max_events=10)  # greedy
    assert codec.decode_sequence(generated + [codec.eos]) == PATTERN


def test_save_load_roundtrip_preserves_generation(tmp_path):
    model, vocab = _overfit_model()
    path = tmp_path / "model.pt"
    save_model(model, path)
    reloaded = load_model(path)
    assert reloaded.generate(temperature=0.0) == model.generate(temperature=0.0)
