"""Factored per-event recurrent model.

Each note is a handful of independent categorical fields (pitch, dt, dur, velocity, channel, and —
for duet models — source); the model embeds each, concatenates them, runs a GRU over the sequence,
and predicts the *next* note with one softmax head per field. A learned ``start`` vector is the
first input, so the model can generate from nothing; generation stops when the pitch head emits EOS.

This recurrence is deliberately O(1)-per-event at inference (carry the hidden state, feed one note,
emit one note) — the property the live duet needs. The streaming API (:meth:`fresh_state`,
:meth:`observe`, :meth:`sample_next`) exposes that hidden state so the duet can teacher-force the
human's observed events into it and sample only its own; :meth:`generate` is a convenience built on
top for unconditional sampling and ``context``-primed continuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from osc_genai.core.vocab import Fields, VocabConfig


@dataclass
class ModelConfig:
    embed_dim: int = 32  # per-field embedding size
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0


class FactoredEventModel(nn.Module):
    def __init__(
        self, vocab: VocabConfig | None = None, config: ModelConfig | None = None
    ) -> None:
        super().__init__()
        self.vocab = vocab or VocabConfig()
        self.config = config or ModelConfig()
        sizes = self.vocab.field_sizes  # (pitch, dt, dur, velocity, channel[, source])
        embed = self.config.embed_dim

        self.embeddings = nn.ModuleList([nn.Embedding(size, embed) for size in sizes])
        input_dim = embed * len(sizes)
        # Metrical phase (bar-relative grid position) is an *input-only* conditioning feature: embedded
        # into the input but never predicted (no head, absent from the loss). It is derived from the
        # ``dt`` column, so it adds no plumbing to encode/collate/train — only the input widens.
        if self.vocab.use_phase:
            self.phase_embed = nn.Embedding(self.vocab.steps_per_bar, embed)
            input_dim += embed
        self.start = nn.Parameter(torch.zeros(input_dim))
        self.rnn = nn.GRU(
            input_dim,
            self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.heads = nn.ModuleList(
            [nn.Linear(self.config.hidden_size, size) for size in sizes]
        )

    # -- core -----------------------------------------------------------------------------------
    def embed(
        self, fields: torch.Tensor, phase: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(..., num_fields)`` long field indices -> ``(..., input_dim)`` embedding.

        With ``use_phase`` the per-event bar phase ``(...)`` long must be supplied and is embedded
        alongside the predicted fields as an input-only conditioning feature.
        """
        parts = [emb(fields[..., i]) for i, emb in enumerate(self.embeddings)]
        if self.vocab.use_phase:
            parts.append(self.phase_embed(phase))
        return torch.cat(parts, dim=-1)

    def _phase_of(self, onset: torch.Tensor) -> torch.Tensor:
        """Bar-relative grid position from an absolute onset (in grid steps)."""
        return onset % self.vocab.steps_per_bar

    def _logits(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        return [head(hidden) for head in self.heads]

    def forward(self, targets: torch.Tensor) -> list[torch.Tensor]:
        """``targets`` ``(B, L, num_fields)`` long -> per-field logits, each ``(B, L, vocab_i)``.

        Input at step *t* is the embedding of target *t-1* (a learned ``start`` at *t=0*), so logits
        at *t* predict target *t* — standard teacher forcing. Phase, when used, is the running onset
        (cumulative ``dt``) mod bar, so each event carries its own metrical position.
        """
        batch = targets.shape[0]
        phase = (
            self._phase_of(torch.cumsum(targets[..., 1], dim=1))
            if self.vocab.use_phase
            else None
        )
        emb = self.embed(targets, phase)  # (B, L, input_dim)
        start = self.start.view(1, 1, -1).expand(batch, 1, -1)
        inp = torch.cat([start, emb[:, :-1, :]], dim=1)  # (B, L, input_dim)
        out, _ = self.rnn(inp)  # (B, L, hidden)
        return self._logits(out)

    def loss(
        self,
        targets: torch.Tensor,
        mask: torch.Tensor,
        pitch_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum of per-field cross-entropies, averaged over masked positions.

        ``mask`` ``(B, L)`` marks valid positions (real events + the terminal EOS). The dt/dur/
        velocity/channel heads are not trained on EOS positions. ``pitch_weights`` (per-pitch class
        weights) re-balances the pitch head so rare-but-essential classes (kick/snare) aren't drowned
        out by frequent ones (hi-hat).
        """
        logits = self.forward(targets)
        eos = targets[..., 0] == self.vocab.eos_pitch
        other = (
            mask & ~eos
        )  # dt/dur/velocity/channel are meaningless at the EOS position
        masks = [mask] + [other] * (targets.shape[-1] - 1)
        total = torch.zeros((), device=targets.device)
        for i, (field_logits, field_mask) in enumerate(zip(logits, masks, strict=True)):
            vocab = field_logits.shape[-1]
            ce = F.cross_entropy(
                field_logits.reshape(-1, vocab),
                targets[..., i].reshape(-1),
                weight=pitch_weights
                if i == 0
                else None,  # re-balance the pitch head only
                reduction="none",
            ).reshape(targets.shape[:2])
            total = total + (ce * field_mask).sum() / field_mask.sum().clamp(min=1)
        return total

    # -- streaming state ------------------------------------------------------------------------
    # The GRU hidden state IS the model's memory of everything fed so far. Exposing it lets the live
    # duet teacher-force the human's observed events into the state and sample only its own — the
    # interleaved two-stream conditioning a single ``generate`` loop can't express.

    # The streaming ``state`` is ``(hidden, onset)``: the GRU hidden state plus the absolute onset (in
    # grid steps) of the last event fed in, so phase = onset mod bar can be tracked incrementally.

    @torch.no_grad()
    def fresh_state(self, onset0: int = 0) -> tuple[torch.Tensor, int]:
        """State after priming with the learned ``start`` — the point before the first event.

        ``onset0`` seeds the running onset (in grid steps) so the phase feature aligns to an absolute
        clock: pass the bar-grid position of the first event the caller is about to ``observe``.
        """
        self.eval()
        _, hidden = self.rnn(self.start.view(1, 1, -1))  # (num_layers, 1, hidden)
        return hidden, onset0

    @torch.no_grad()
    def observe(
        self, state: tuple[torch.Tensor, int], fields: Fields
    ) -> tuple[torch.Tensor, int]:
        """Advance the state by one teacher-forced event (partner or committed self), emitting nothing."""
        hidden, onset = state
        onset += fields[1]  # dt (in grid steps) advances the running onset
        phase = None
        if self.vocab.use_phase:
            phase = torch.tensor(
                [[onset % self.vocab.steps_per_bar]],
                dtype=torch.long,
                device=self.start.device,
            )
        step = self.embed(
            torch.tensor([[fields]], dtype=torch.long, device=self.start.device), phase
        )
        _, hidden = self.rnn(step, hidden)
        return hidden, onset

    @torch.no_grad()
    def sample_next(
        self,
        state: tuple[torch.Tensor, int],
        temperature: float = 1.0,
        *,
        bias: dict[int, torch.Tensor] | None = None,
        force: dict[int, int] | None = None,
        regular_pitches: set[int] | None = None,
        regular_temperature: float | None = None,
    ) -> tuple[Fields, tuple[torch.Tensor, int]]:
        """Sample the next event from ``state``; return it and the advanced state.

        ``bias`` maps a field index to an additive logit vector (its field's vocab size) — e.g.
        ``{0: pitch_bias}`` to boost a kick. ``force`` pins a field to a fixed value — e.g.
        ``{source_field: SELF}`` when the duet *requires* a self note rather than a predicted partner
        one. ``temperature <= 0`` is greedy. When the sampled pitch is in ``regular_pitches`` (e.g.
        kick/snare), the *timing* fields (dt, dur) are sampled at ``regular_temperature`` instead —
        near-greedy so the foundation stays regular while other lanes keep their variety.
        """
        hidden, _ = state
        fields = self._sample(
            hidden[-1], temperature, bias, force, regular_pitches, regular_temperature
        )
        return fields, self.observe(state, fields)

    # -- generation -----------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        context: list[Fields] | None = None,
        max_events: int = 64,
        temperature: float = 1.0,
        pitch_bias: dict[int, float] | None = None,
    ) -> list[Fields]:
        """Autoregressively sample events. With ``context`` the model is primed on it first.

        ``temperature <= 0`` is greedy (argmax). Sampling halts at EOS or ``max_events``.
        ``pitch_bias`` adds a per-pitch value to the pitch logits before sampling — e.g. boost an
        under-produced kick/snare or damp a dominant hi-hat. A thin wrapper over the streaming API.
        """
        bias = None
        if pitch_bias:
            tensor = torch.zeros(self.vocab.pitch_vocab, device=self.start.device)
            for pitch, value in pitch_bias.items():
                tensor[pitch] += value
            bias = {0: tensor}  # field 0 is pitch

        state = self.fresh_state()
        for fields in context or []:
            state = self.observe(state, fields)

        result: list[Fields] = []
        for _ in range(max_events):
            fields, state = self.sample_next(state, temperature, bias=bias)
            if fields[0] == self.vocab.eos_pitch:
                break
            result.append(fields)
        return result

    def _sample(
        self,
        hidden: torch.Tensor,
        temperature: float,
        bias: dict[int, torch.Tensor] | None = None,
        force: dict[int, int] | None = None,
        regular_pitches: set[int] | None = None,
        regular_temperature: float | None = None,
    ) -> Fields:
        all_logits = self._logits(hidden)  # each (1, vocab)

        def pick(i: int, temp: float) -> int:
            if force is not None and i in force:
                return int(force[i])
            logits = all_logits[i].squeeze(0)
            if bias is not None and i in bias:
                logits = logits + bias[i]
            if temp <= 0:
                return int(torch.argmax(logits))
            return int(torch.multinomial(F.softmax(logits / temp, dim=-1), 1))

        pitch = pick(0, temperature)
        # A kick/snare keeps the foundation tight: sample its timing (dt=1, dur=2) near-greedy.
        on_grid = (
            regular_pitches is not None
            and regular_temperature is not None
            and pitch in regular_pitches
        )
        indices = [pitch] + [
            pick(i, regular_temperature if (on_grid and i in (1, 2)) else temperature)
            for i in range(1, len(all_logits))
        ]
        return tuple(indices)  # type: ignore[return-value]
