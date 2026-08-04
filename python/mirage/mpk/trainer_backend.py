"""Trainer-backend bridge for MPK-owned forward values.

MPK is the numerical authority for policy log-probabilities.  A trainer
backend still runs a differentiable replay to supply the vector-Jacobian
product, optimizer, and distributed training machinery.  This module keeps
that boundary explicit and lets HF, TorchTitan, Megatron, or a future native
MPK backward implement the same small contract.
"""

from __future__ import annotations

import copy
import importlib
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import torch


Sample = Mapping[str, object]
NamedTensorSource = Callable[[], Iterable[tuple[str, torch.Tensor]]]


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


@dataclass(frozen=True)
class _ReplayBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    batch_rows: torch.Tensor
    token_rows: torch.Tensor
    targets: torch.Tensor
    counts: list[int]


def _prepare_replay_batch(
    samples: Sequence[Sample], tokenizer, device: torch.device | str
) -> _ReplayBatch:
    lengths = [len(sample["ids"]) for sample in samples]
    counts = [len(sample["pos"]) for sample in samples]
    max_length = max(lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
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

    return _ReplayBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        batch_rows=torch.tensor(batch_rows, dtype=torch.long, device=device),
        token_rows=torch.tensor(token_rows, dtype=torch.long, device=device),
        targets=torch.tensor(targets, dtype=torch.long, device=device),
        counts=counts,
    )


def _selected_token_logprobs_from_logits(
    logits: torch.Tensor,
    batch: _ReplayBatch,
    *,
    sequence_first: bool = False,
) -> list[torch.Tensor]:
    if not batch.targets.numel():
        empty = torch.empty(0, dtype=torch.float32, device=logits.device)
        return [empty for _ in batch.counts]

    batch_size, seq_len = batch.input_ids.shape
    if sequence_first:
        logits = logits.transpose(0, 1)
    if tuple(logits.shape[:2]) != (batch_size, seq_len):
        raise ValueError(
            "trainer logits do not match the replay batch after layout "
            "normalization: expected [batch, sequence, vocabulary], got "
            f"{tuple(logits.shape)}"
        )

    selected_logits = logits[batch.batch_rows, batch.token_rows].float()
    flat = torch.log_softmax(selected_logits, dim=-1).gather(
        -1, batch.targets.unsqueeze(-1)
    ).squeeze(-1)
    return list(torch.split(flat, batch.counts))


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


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
        batch = _prepare_replay_batch(
            samples, self.tokenizer, _model_device(self.model)
        )
        logits = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        ).logits
        return _selected_token_logprobs_from_logits(logits, batch)

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


class TorchTitanTrainerBackend:
    """TorchTitan-native replay, clipping, optimizer, and scheduler adapter.

    TorchTitan pipeline schedules couple forward and backward in one call,
    while the Phase-1 value bridge currently requests differentiable replay
    values before constructing the RL loss. For that reason this adapter
    supports one model part (including FSDP2/TP) and rejects PP for now.
    """

    def __init__(
        self,
        model_parts: Sequence[torch.nn.Module],
        tokenizer,
        optimizers,
        *,
        lr_schedulers=None,
        micro_batch_size: int = 0,
        grad_clip: float = 1.0,
        parallel_dims=None,
        named_tensor_source: NamedTensorSource | None = None,
    ) -> None:
        if len(model_parts) != 1:
            raise ValueError(
                "TorchTitanTrainerBackend currently supports one model part; "
                "pipeline parallelism needs a schedule-owned RL loss callback"
            )
        if micro_batch_size < 0:
            raise ValueError("micro_batch_size must be non-negative")
        if optimizers.__class__.__name__ == "OptimizersInBackwardContainer":
            raise ValueError(
                "TorchTitan optimizer-in-backward is incompatible with "
                "post-backward distributed clipping and grad-norm reporting"
            )
        self.model_parts = list(model_parts)
        self.model = self.model_parts[0]
        self.tokenizer = tokenizer
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.micro_batch_size = micro_batch_size
        self.grad_clip = grad_clip
        self.parallel_dims = parallel_dims
        self._named_tensor_source = named_tensor_source

    def selected_token_logprobs(
        self, samples: Sequence[Sample]
    ) -> list[torch.Tensor]:
        if not samples:
            return []
        batch_size = self.micro_batch_size or len(samples)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(samples), batch_size):
            micro = samples[start:start + batch_size]
            batch = _prepare_replay_batch(
                micro, self.tokenizer, _model_device(self.model)
            )
            logits = self.model(batch.input_ids)
            if hasattr(logits, "logits"):
                logits = logits.logits
            outputs.extend(_selected_token_logprobs_from_logits(logits, batch))
        return outputs

    def zero_grad(self) -> None:
        self.optimizers.zero_grad(set_to_none=True)

    def backward_and_step(self, loss: torch.Tensor) -> float:
        loss.backward()
        try:
            from torchtitan.distributed import utils as dist_utils
        except ImportError as exc:
            raise RuntimeError(
                "TorchTitanTrainerBackend requires the 'torchtitan' package"
            ) from exc

        pp_mesh = None
        ep_enabled = False
        if self.parallel_dims is not None:
            if getattr(self.parallel_dims, "pp_enabled", False):
                pp_mesh = self.parallel_dims.world_mesh["pp"]
            ep_enabled = bool(getattr(self.parallel_dims, "ep_enabled", False))
        grad_norm = dist_utils.clip_grad_norm_(
            [p for part in self.model_parts for p in part.parameters()],
            self.grad_clip,
            foreach=True,
            pp_mesh=pp_mesh,
            ep_enabled=ep_enabled,
        )
        self.optimizers.step()
        if self.lr_schedulers is not None:
            self.lr_schedulers.step()
        return float(grad_norm)

    def named_parameters(self):
        if self._named_tensor_source is not None:
            return self._named_tensor_source()
        return self.model.named_parameters()


class MegatronTrainerBackend:
    """Megatron-Core TP/DDP replay and native optimizer adapter.

    Logits are gathered at runtime so selected-token log-softmax sees the
    complete vocabulary. As with TorchTitan, PP is rejected until the RL
    loss is moved into Megatron's schedule-owned loss callback.
    """

    def __init__(
        self,
        model_chunks: Sequence[torch.nn.Module],
        tokenizer,
        optimizer,
        *,
        lr_scheduler=None,
        micro_batch_size: int = 0,
        grad_clip: float = 1.0,
        sequence_first: bool = False,
        named_tensor_source: NamedTensorSource | None = None,
        finalize_grads: Callable[[Sequence[torch.nn.Module]], None] | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        if len(model_chunks) != 1:
            raise ValueError(
                "MegatronTrainerBackend currently supports PP=1; pipeline "
                "parallelism needs a schedule-owned RL loss callback"
            )
        if micro_batch_size < 0:
            raise ValueError("micro_batch_size must be non-negative")
        self.model_chunks = list(model_chunks)
        self.model = self.model_chunks[0]
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.micro_batch_size = micro_batch_size
        self.grad_clip = grad_clip
        self.sequence_first = sequence_first
        self._named_tensor_source = named_tensor_source
        self._finalize_grads = finalize_grads
        self._cleanup = cleanup
        self._closed = False

    def selected_token_logprobs(
        self, samples: Sequence[Sample]
    ) -> list[torch.Tensor]:
        if not samples:
            return []
        batch_size = self.micro_batch_size or len(samples)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(samples), batch_size):
            micro = samples[start:start + batch_size]
            batch = _prepare_replay_batch(
                micro, self.tokenizer, _model_device(self.model)
            )
            seq_len = batch.input_ids.shape[1]
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=batch.input_ids.device
            ).unsqueeze(0).expand(batch.input_ids.shape[0], -1)
            causal_mask = torch.triu(
                torch.ones(
                    (1, 1, seq_len, seq_len),
                    dtype=torch.bool,
                    device=batch.input_ids.device,
                ),
                diagonal=1,
            )
            logits = self.model(
                batch.input_ids,
                position_ids,
                causal_mask,
                labels=None,
                runtime_gather_output=True,
            )
            if hasattr(logits, "logits"):
                logits = logits.logits
            outputs.extend(
                _selected_token_logprobs_from_logits(
                    logits, batch, sequence_first=self.sequence_first
                )
            )
        return outputs

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def backward_and_step(self, loss: torch.Tensor) -> float:
        loss.backward()
        if self._finalize_grads is not None:
            self._finalize_grads(self.model_chunks)
        is_megatron_optimizer = self.optimizer.__class__.__module__.startswith(
            "megatron."
        )
        if not is_megatron_optimizer:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()
        else:
            result = self.optimizer.step()
            if isinstance(result, tuple):
                success = bool(result[0])
                grad_norm = result[1] if len(result) > 1 else 0.0
                if not success:
                    raise FloatingPointError(
                        "Megatron optimizer skipped the step due to overflow"
                    )
            else:
                grad_norm = 0.0
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return float(grad_norm or 0.0)

    def named_parameters(self):
        if self._named_tensor_source is not None:
            return self._named_tensor_source()
        return self.model.named_parameters()

    def close(self) -> None:
        if not self._closed and self._cleanup is not None:
            self._cleanup()
        self._closed = True


def _create_torchtitan_backend(
    *,
    model_name: str,
    tokenizer,
    learning_rate: float,
    micro_batch_size: int,
    grad_clip: float,
    device: str,
    options: Mapping[str, object],
) -> TorchTitanTrainerBackend:
    options = dict(options)
    options.pop("engine_args", None)
    model_parts = options.pop("model_parts", None)
    optimizers = options.pop("optimizers", None)
    if model_parts is not None or optimizers is not None:
        if model_parts is None or optimizers is None:
            raise ValueError(
                "prebuilt TorchTitan backend needs both model_parts and optimizers"
            )
        lr_schedulers = options.pop("lr_schedulers", None)
        parallel_dims = options.pop("parallel_dims", None)
        named_tensor_source = options.pop("named_tensor_source", None)
        if options:
            names = ", ".join(sorted(options))
            raise ValueError(f"unsupported TorchTitan backend options: {names}")
        return TorchTitanTrainerBackend(
            model_parts,
            tokenizer,
            optimizers,
            lr_schedulers=lr_schedulers,
            micro_batch_size=micro_batch_size,
            grad_clip=grad_clip,
            parallel_dims=parallel_dims,
            named_tensor_source=named_tensor_source,
        )

    try:
        from torchtitan.components.optimizer import OptimizersContainer
        from torchtitan.models.qwen3 import (
            Qwen3Model,
            Qwen3StateDictAdapter,
            qwen3_args,
        )
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "the 'torchtitan' backend requires torchtitan with Qwen3 support"
        ) from exc

    config = AutoConfig.from_pretrained(model_name)
    matches = []
    for variant, candidate in qwen3_args.items():
        if variant.startswith("debug"):
            continue
        dense_match = (
            candidate.dim == config.hidden_size
            and candidate.n_layers == config.num_hidden_layers
            and candidate.n_heads == config.num_attention_heads
            and candidate.n_kv_heads == config.num_key_value_heads
            and candidate.vocab_size == config.vocab_size
        )
        if dense_match:
            matches.append((variant, candidate))
    if len(matches) != 1:
        raise ValueError(
            "TorchTitan Qwen3 registry has no unique architecture match for "
            f"{model_name!r}"
        )
    _, model_args = matches[0]
    model_args = copy.deepcopy(model_args)
    model_args.max_seq_len = min(
        model_args.max_seq_len,
        int(getattr(config, "max_position_embeddings", model_args.max_seq_len)),
    )

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3Model(model_args)
    finally:
        torch.set_default_dtype(old_dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    )
    state_adapter = Qwen3StateDictAdapter(model_args, None)
    titan_state = state_adapter.from_hf(hf_model.state_dict())
    model.load_state_dict(titan_state, strict=True)
    del hf_model, titan_state
    model.to(device)
    model.train()

    optimizer_kwargs = {
        "lr": learning_rate,
        "weight_decay": float(options.pop("weight_decay", 0.0)),
        "betas": options.pop("betas", (0.9, 0.95)),
        "eps": float(options.pop("eps", 1e-8)),
    }
    if options:
        names = ", ".join(sorted(options))
        raise ValueError(f"unsupported TorchTitan backend options: {names}")
    optimizers = OptimizersContainer(
        [model], torch.optim.AdamW, optimizer_kwargs
    )

    def named_tensors():
        return state_adapter.to_hf(dict(model.named_parameters())).items()

    return TorchTitanTrainerBackend(
        [model],
        tokenizer,
        optimizers,
        micro_batch_size=micro_batch_size,
        grad_clip=grad_clip,
        named_tensor_source=named_tensors,
    )


def _create_megatron_backend(
    *,
    model_name: str,
    tokenizer,
    learning_rate: float,
    micro_batch_size: int,
    grad_clip: float,
    device: str,
    options: Mapping[str, object],
) -> MegatronTrainerBackend:
    del device
    options = dict(options)
    options.pop("engine_args", None)
    model_chunks = options.pop("model_chunks", None)
    optimizer = options.pop("optimizer", None)
    if model_chunks is not None or optimizer is not None:
        if model_chunks is None or optimizer is None:
            raise ValueError(
                "prebuilt Megatron backend needs both model_chunks and optimizer"
            )
        lr_scheduler = options.pop("lr_scheduler", None)
        sequence_first = bool(options.pop("sequence_first", False))
        named_tensor_source = options.pop("named_tensor_source", None)
        finalize_grads = options.pop("finalize_grads", None)
        if options:
            names = ", ".join(sorted(options))
            raise ValueError(f"unsupported Megatron backend options: {names}")
        return MegatronTrainerBackend(
            model_chunks,
            tokenizer,
            optimizer,
            lr_scheduler=lr_scheduler,
            micro_batch_size=micro_batch_size,
            grad_clip=grad_clip,
            sequence_first=sequence_first,
            named_tensor_source=named_tensor_source,
            finalize_grads=finalize_grads,
        )

    wrap_with_ddp = bool(options.pop("wrap_with_ddp", True))
    if not wrap_with_ddp:
        raise ValueError(
            "the built-in Megatron backend requires wrap_with_ddp=True so "
            "Megatron's optimizer can access distributed gradient buffers; "
            "pass a prebuilt model/optimizer for custom non-DDP execution"
        )
    try:
        from megatron.bridge import AutoBridge
        from megatron.core import parallel_state
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.optimizer import get_megatron_optimizer
        from megatron.core.optimizer.optimizer_config import OptimizerConfig
    except ImportError as exc:
        raise RuntimeError(
            "the 'megatron' backend requires Megatron-Core and "
            "Megatron Bridge with Qwen3 support"
        ) from exc

    tp_size = int(options.pop("tensor_model_parallel_size", 1))
    pp_size = int(options.pop("pipeline_model_parallel_size", 1))
    if pp_size != 1:
        raise ValueError("MegatronTrainerBackend currently requires PP=1")
    weight_decay = float(options.pop("weight_decay", 0.0))
    use_distributed_optimizer = bool(
        options.pop("use_distributed_optimizer", True)
    )
    if options:
        names = ", ".join(sorted(options))
        raise ValueError(f"unsupported Megatron backend options: {names}")

    owns_process_group = False
    owns_model_parallel = False

    def cleanup() -> None:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    try:
        if not torch.distributed.is_initialized():
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", os.environ["RANK"])
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            owns_process_group = True

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=pp_size,
            )
            owns_model_parallel = True
        else:
            actual_tp = parallel_state.get_tensor_model_parallel_world_size()
            actual_pp = parallel_state.get_pipeline_model_parallel_world_size()
            if (actual_tp, actual_pp) != (tp_size, pp_size):
                raise ValueError(
                    "existing Megatron model-parallel sizes do not match "
                    f"the backend request: existing TP={actual_tp}, PP={actual_pp}; "
                    f"requested TP={tp_size}, PP={pp_size}"
                )

        bridge = AutoBridge.from_hf_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = tp_size
        provider.pipeline_model_parallel_size = pp_size
        provider.finalize()
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=use_distributed_optimizer,
        )
        model_chunks = provider.provide_distributed_model(
            ddp_config=ddp_config,
            wrap_with_ddp=wrap_with_ddp,
        )
        if not isinstance(model_chunks, (list, tuple)):
            model_chunks = [model_chunks]
        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=learning_rate,
            min_lr=0.0,
            weight_decay=weight_decay,
            clip_grad=grad_clip,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=use_distributed_optimizer,
        )
        optimizer = get_megatron_optimizer(
            optimizer_config, list(model_chunks)
        )
        model_config = getattr(model_chunks[0], "config", None)
        finalize_grads = getattr(
            model_config, "finalize_model_grads_func", None
        )
    except Exception:
        cleanup()
        raise

    def named_tensors():
        return bridge.export_hf_weights(
            list(model_chunks), cpu=False, show_progress=False
        )

    return MegatronTrainerBackend(
        model_chunks,
        tokenizer,
        optimizer,
        micro_batch_size=micro_batch_size,
        grad_clip=grad_clip,
        named_tensor_source=named_tensors,
        finalize_grads=finalize_grads,
        cleanup=cleanup,
    )


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

    ``hf``, ``torchtitan``, and ``megatron`` are built-in choices. Any
    ``module:factory`` entry is loaded lazily and receives the same
    construction arguments for project-specific integrations.
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

    backend_options = dict(factory_kwargs or {})
    if spec == "torchtitan":
        return _create_torchtitan_backend(
            model_name=model_name,
            tokenizer=tokenizer,
            learning_rate=learning_rate,
            micro_batch_size=micro_batch_size,
            grad_clip=grad_clip,
            device=device,
            options=backend_options,
        )
    if spec == "megatron":
        return _create_megatron_backend(
            model_name=model_name,
            tokenizer=tokenizer,
            learning_rate=learning_rate,
            micro_batch_size=micro_batch_size,
            grad_clip=grad_clip,
            device=device,
            options=backend_options,
        )

    if ":" not in spec:
        raise ValueError(
            "trainer backend must be 'hf', 'torchtitan', 'megatron', "
            "or '<module>:<factory>', "
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
