"""Trainer-backend bridge for MPK-owned forward values.

MPK is the numerical authority for policy log-probabilities.  A trainer
backend still runs a differentiable replay to supply the vector-Jacobian
product, optimizer, and distributed training machinery.  This module keeps
that boundary explicit and lets HF, TorchTitan, Megatron, or a future native
MPK backward implement the same small contract.
"""

from __future__ import annotations

import importlib
from typing import Mapping, Protocol, Sequence, runtime_checkable

import torch


Sample = Mapping[str, object]


@runtime_checkable
class TrainerBackend(Protocol):
    """Minimal contract consumed by an MPK RL training loop."""

    def selected_token_logprobs(
        self, samples: Sequence[Sample]
    ) -> list[torch.Tensor]:
        """Return differentiable selected-token log-probabilities."""

    def zero_grad(self) -> None:
        """Clear accumulated parameter gradients."""

    def backward_and_step(self, loss: torch.Tensor) -> float:
        """Run backward and the optimizer, returning the pre-clip grad norm."""

    def named_parameters(self):
        """Expose trainer parameters for weight synchronization."""


def bind_forward_values(
    authoritative: torch.Tensor,
    differentiable: torch.Tensor,
) -> torch.Tensor:
    """Use ``authoritative`` values with ``differentiable`` gradients.

    The parenthesized difference is exactly zero in forward, while its
    derivative with respect to ``differentiable`` is one.  Keeping the normal
    trainer graph visible to autograd is important for FSDP and pipeline
    schedules; it also avoids a nested backward call inside autograd.Function.
    """

    if authoritative.shape != differentiable.shape:
        raise ValueError(
            "authoritative and differentiable values must have the same "
            f"shape, got {authoritative.shape} and {differentiable.shape}"
        )
    if authoritative.device != differentiable.device:
        raise ValueError(
            "authoritative and differentiable values must be on the same "
            f"device, got {authoritative.device} and {differentiable.device}"
        )
    return authoritative.detach() + (
        differentiable - differentiable.detach()
    )


class HuggingFaceTrainerBackend:
    """Batched selected-token replay on a HuggingFace causal LM."""

    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        *,
        micro_batch_size: int = 0,
        grad_clip: float = 1.0,
    ) -> None:
        if micro_batch_size < 0:
            raise ValueError("micro_batch_size must be non-negative")
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.micro_batch_size = micro_batch_size
        self.grad_clip = grad_clip

    def selected_token_logprobs(
        self, samples: Sequence[Sample]
    ) -> list[torch.Tensor]:
        if not samples:
            return []

        batch_size = self.micro_batch_size or len(samples)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(samples), batch_size):
            outputs.extend(
                self._selected_token_logprobs_batch(samples[start:start + batch_size])
            )
        return outputs

    def _selected_token_logprobs_batch(
        self, samples: Sequence[Sample]
    ) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device
        lengths = [len(sample["ids"]) for sample in samples]
        counts = [len(sample["pos"]) for sample in samples]
        max_length = max(lengths)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = torch.full(
            (len(samples), max_length),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        batch_rows: list[int] = []
        token_rows: list[int] = []
        targets: list[int] = []
        for batch_idx, sample in enumerate(samples):
            ids = sample["ids"]
            positions = sample["pos"]
            seq = torch.as_tensor(ids, dtype=torch.long, device=device)
            input_ids[batch_idx, :seq.numel()] = seq
            attention_mask[batch_idx, :seq.numel()] = 1
            batch_rows.extend([batch_idx] * len(positions))
            token_rows.extend(int(pos) - 1 for pos in positions)
            targets.extend(int(ids[int(pos)]) for pos in positions)

        if not targets:
            empty = torch.empty(0, dtype=torch.float32, device=device)
            return [empty for _ in samples]

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        batch_index = torch.tensor(batch_rows, dtype=torch.long, device=device)
        row_index = torch.tensor(token_rows, dtype=torch.long, device=device)
        target_index = torch.tensor(targets, dtype=torch.long, device=device)
        selected_logits = logits[batch_index, row_index].float()
        flat = torch.log_softmax(selected_logits, dim=-1).gather(
            -1, target_index.unsqueeze(-1)
        ).squeeze(-1)
        return list(torch.split(flat, counts))

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def backward_and_step(self, loss: torch.Tensor) -> float:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        )
        self.optimizer.step()
        return float(grad_norm)

    def named_parameters(self):
        return self.model.named_parameters()


def create_trainer_backend(
    spec: str,
    *,
    model_name: str,
    tokenizer,
    learning_rate: float,
    micro_batch_size: int = 0,
    grad_clip: float = 1.0,
    device: str = "cuda",
    factory_kwargs: Mapping[str, object] | None = None,
) -> TrainerBackend:
    """Create the selected trainer stack without coupling MPK to it.

    ``hf`` is the built-in, evaluated backend. Any ``module:factory`` entry
    is loaded lazily and receives the same construction arguments. This is
    the integration point for TorchTitan, Megatron, or another distributed
    trainer; those frameworks remain responsible for their model topology,
    replay schedule, backward, optimizer, and normalized parameter names.
    """

    common_kwargs = {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "learning_rate": learning_rate,
        "micro_batch_size": micro_batch_size,
        "grad_clip": grad_clip,
        "device": device,
    }
    if factory_kwargs:
        overlap = common_kwargs.keys() & factory_kwargs.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"factory_kwargs cannot override: {names}")
        common_kwargs.update(factory_kwargs)

    if spec == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(device)
        model.gradient_checkpointing_enable()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        return HuggingFaceTrainerBackend(
            model,
            tokenizer,
            optimizer,
            micro_batch_size=micro_batch_size,
            grad_clip=grad_clip,
        )

    if ":" not in spec:
        raise ValueError(
            "trainer backend must be 'hf' or '<module>:<factory>', "
            f"got {spec!r}"
        )
    module_name, factory_name = spec.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    backend = factory(**common_kwargs)
    if not isinstance(backend, TrainerBackend):
        raise TypeError(
            f"trainer backend factory {spec!r} did not return an object "
            "implementing TrainerBackend"
        )
    return backend
