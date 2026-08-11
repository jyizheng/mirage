"""Weight synchronization plans for MPK rollout engines.

The persistent kernel stores raw pointers to attached PyTorch tensors.  A
training loop therefore needs a repeatable way to update those tensors after
each optimizer step.  This module keeps the mapping logic out of experiment
scripts: build a plan once, then execute it every step with direct copies
(colocated) or with a transport layer around the same fitted tensor slices
(disaggregated).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Mapping, Sequence

import torch


TensorMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class SyncSpec:
    """One target tensor update in a synchronization plan."""

    target: str
    sources: tuple[str, ...]
    transform: str = "copy"
    split: int | None = None
    dim: int = 0


@dataclass
class SyncReport:
    """Execution summary for one sync pass."""

    tensors: int
    bytes: int
    elapsed_s: float
    missing_sources: tuple[str, ...] = ()
    missing_targets: tuple[str, ...] = ()

    @property
    def gib(self) -> float:
        return self.bytes / float(1 << 30)

    @property
    def bandwidth_gib_s(self) -> float:
        return self.gib / self.elapsed_s if self.elapsed_s > 0 else 0.0


class WeightSyncPlan:
    """A reusable trainer-to-rollout tensor synchronization plan."""

    def __init__(
        self,
        specs: Sequence[SyncSpec],
        *,
        rank: int = 0,
        world_size: int = 1,
        name: str = "weight-sync",
    ) -> None:
        self.specs = tuple(specs)
        self.rank = rank
        self.world_size = world_size
        self.name = name

    def sync(
        self,
        sources: TensorMap,
        targets: TensorMap,
        *,
        non_blocking: bool = True,
        strict: bool = True,
    ) -> SyncReport:
        """Execute the plan with local tensor copies.

        ``sources`` is typically ``trainer.state_dict()`` or
        ``dict(trainer.named_parameters())``. ``targets`` is typically
        ``runner.mpk.persistent_kernel._model_tensors`` or another mapping of
        MPK attached tensor names to tensors.
        """

        missing_sources: list[str] = []
        missing_targets: list[str] = []
        copied = 0
        copied_bytes = 0
        start = time.perf_counter()
        with torch.no_grad():
            for spec in self.specs:
                target = targets.get(spec.target)
                if target is None:
                    missing_targets.append(spec.target)
                    continue
                srcs = []
                for name in spec.sources:
                    src = sources.get(name)
                    if src is None:
                        missing_sources.append(name)
                    else:
                        srcs.append(src)
                if len(srcs) != len(spec.sources):
                    continue
                fitted = self._fit_spec(spec, srcs, target)
                _copy_with_optional_pad(fitted, target, non_blocking=non_blocking)
                copied += 1
                copied_bytes += target.numel() * target.element_size()

        if strict and (missing_sources or missing_targets):
            details = []
            if missing_sources:
                details.append(f"missing sources: {sorted(set(missing_sources))}")
            if missing_targets:
                details.append(f"missing targets: {sorted(set(missing_targets))}")
            raise KeyError("; ".join(details))

        return SyncReport(
            tensors=copied,
            bytes=copied_bytes,
            elapsed_s=time.perf_counter() - start,
            missing_sources=tuple(sorted(set(missing_sources))),
            missing_targets=tuple(sorted(set(missing_targets))),
        )

    def _fit_spec(
        self,
        spec: SyncSpec,
        srcs: Sequence[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if spec.transform == "copy":
            return _view_or_shard_to_target(
                srcs[0], target, self.rank, self.world_size)
        if spec.transform == "cat":
            pieces = [
                _view_for_partitioned_source(
                    s, target, self.rank, self.world_size, len(srcs), spec.dim)
                for s in srcs
            ]
            return torch.cat(pieces, dim=spec.dim)
        if spec.transform == "qwen_shuffle":
            if spec.split is None:
                raise ValueError(f"{spec.target}: qwen_shuffle requires split")
            pieces = [
                _view_for_partitioned_source(
                    s, target, self.rank, self.world_size, len(srcs), spec.dim)
                for s in srcs
            ]
            return _shuffle_tensors(pieces, spec.split, spec.dim)
        raise ValueError(f"unknown sync transform: {spec.transform}")


def tensor_map(obj) -> dict[str, torch.Tensor]:
    """Normalize a module, state_dict, MPK, or PersistentKernel to a tensor map."""

    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "persistent_kernel"):
        return tensor_map(obj.persistent_kernel)
    if hasattr(obj, "mpk"):
        return tensor_map(obj.mpk)
    if hasattr(obj, "_model_tensors"):
        return dict(obj._model_tensors)
    if hasattr(obj, "state_dict"):
        return dict(obj.state_dict())
    if hasattr(obj, "named_parameters"):
        return dict(obj.named_parameters())
    raise TypeError(f"cannot derive tensor map from {type(obj)!r}")


def build_name_matching_sync_plan(
    sources: TensorMap,
    targets: TensorMap,
    *,
    rank: int = 0,
    world_size: int = 1,
    tie_lm_head_to_embeddings: bool = True,
) -> WeightSyncPlan:
    """Build a conservative plan for modules with matching parameter names."""

    specs: list[SyncSpec] = []
    for target_name in targets:
        if target_name in sources:
            specs.append(SyncSpec(target=target_name, sources=(target_name,)))
    if (
        tie_lm_head_to_embeddings
        and "lm_head.weight" in targets
        and "lm_head.weight" not in sources
        and "model.embed_tokens.weight" in sources
    ):
        specs.append(
            SyncSpec(
                target="lm_head.weight",
                sources=("model.embed_tokens.weight",),
            )
        )
    return WeightSyncPlan(
        specs, rank=rank, world_size=world_size, name="name-matching-sync")


def build_qwen3_mpk_sync_plan(
    sources: TensorMap,
    targets: TensorMap,
    *,
    num_layers: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    num_local_kv_heads: int | None = None,
    gatedup_split: int | None = None,
) -> WeightSyncPlan:
    """Build a Qwen3 trainer-state -> MPK-attached-tensor sync plan.

    The online pinned engine attaches most Qwen3 projection weights directly
    (``layer_i_q_proj``, ``layer_i_k_proj``, ...).  Some offline/notoken graphs
    attach fused tensors instead; those are supported when the caller supplies
    the same shuffle split used by the builder.
    """

    specs: list[SyncSpec] = []

    def add(target: str, *src_names: str, transform: str = "copy",
            split: int | None = None) -> None:
        if target in targets and all(s in sources for s in src_names):
            specs.append(
                SyncSpec(
                    target=target,
                    sources=tuple(src_names),
                    transform=transform,
                    split=split,
                )
            )

    add("embed_tokens", "model.embed_tokens.weight")
    add("model_norm_weight", "model.norm.weight")
    if "lm_head.weight" in sources:
        add("lm_head", "lm_head.weight")
    else:
        add("lm_head", "model.embed_tokens.weight")

    if num_layers is None:
        num_layers = _infer_qwen3_num_layers(sources)

    for i in range(num_layers):
        p = f"model.layers.{i}."
        add(f"layer_{i}_input_layernorm", p + "input_layernorm.weight")
        add(f"layer_{i}_q_proj", p + "self_attn.q_proj.weight")
        add(f"layer_{i}_k_proj", p + "self_attn.k_proj.weight")
        add(f"layer_{i}_v_proj", p + "self_attn.v_proj.weight")
        add(f"layer_{i}_q_norm", p + "self_attn.q_norm.weight")
        add(f"layer_{i}_k_norm", p + "self_attn.k_norm.weight")
        add(f"layer_{i}_o_proj", p + "self_attn.o_proj.weight")
        add(f"layer_{i}_post_attn_layernorm",
            p + "post_attention_layernorm.weight")
        add(f"layer_{i}_gate_proj", p + "mlp.gate_proj.weight")
        add(f"layer_{i}_up_proj", p + "mlp.up_proj.weight")
        add(f"layer_{i}_down_proj", p + "mlp.down_proj.weight")

        add(f"layer_{i}_qkv_proj", p + "self_attn.qkv_proj.weight")
        if num_local_kv_heads is not None:
            add(
                f"layer_{i}_qkv_proj",
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                transform="qwen_shuffle",
                split=num_local_kv_heads,
            )

        add(f"layer_{i}_gatedup_proj", p + "mlp.gate_up_proj.weight")
        if gatedup_split is not None:
            add(
                f"layer_{i}_gatedup_proj",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                transform="qwen_shuffle",
                split=gatedup_split,
            )

    return WeightSyncPlan(
        specs, rank=rank, world_size=world_size, name="qwen3-mpk-sync")


def _infer_qwen3_num_layers(sources: TensorMap) -> int:
    pat = re.compile(r"^model\.layers\.(\d+)\.")
    layers = [int(m.group(1)) for name in sources for m in [pat.match(name)] if m]
    return max(layers) + 1 if layers else 0


def _view_or_shard_to_target(
    source: torch.Tensor,
    target: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    source = source.detach()
    if tuple(source.shape) == tuple(target.shape):
        return source

    candidates = [source]
    if world_size > 1:
        for dim in range(source.ndim):
            if source.shape[dim] % world_size != 0:
                continue
            per_rank = source.shape[dim] // world_size
            view = source.narrow(dim, rank * per_rank, per_rank)
            candidates.append(view)

    for view in candidates:
        if tuple(view.shape) == tuple(target.shape):
            return view
        if _can_pad_dim0(view, target):
            return view

    raise ValueError(
        f"cannot fit source shape {tuple(source.shape)} to target "
        f"{tuple(target.shape)} at rank {rank}/{world_size}"
    )


def _view_for_partitioned_source(
    source: torch.Tensor,
    fused_target: torch.Tensor,
    rank: int,
    world_size: int,
    parts: int,
    dim: int,
) -> torch.Tensor:
    source = source.detach()
    if world_size == 1:
        return source
    if source.shape[dim] % world_size == 0:
        per_rank = source.shape[dim] // world_size
        return source.narrow(dim, rank * per_rank, per_rank)
    # Some fused targets are padded after concatenation; leave the source
    # whole and let the final copy path validate or fail loudly.
    return source


def _copy_with_optional_pad(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    non_blocking: bool,
) -> None:
    # ``Tensor.copy_`` performs the device transfer and dtype conversion
    # itself, so a cross-device sync (disaggregated trainer, e.g. trainer on
    # cuda:1 -> engine tensors on cuda:0) is a direct P2P copy.  The previous
    # ``source.to(device=..., dtype=...)`` staging was a no-op view when
    # colocated but allocated a full temporary on the target device for every
    # tensor when disaggregated (double copy + allocator churn).
    if tuple(source.shape) == tuple(target.shape):
        target.copy_(source, non_blocking=non_blocking)
        return
    if _can_pad_dim0(source, target):
        target.zero_()
        target[: source.shape[0]].copy_(source, non_blocking=non_blocking)
        return
    raise ValueError(
        f"cannot copy source shape {tuple(source.shape)} into target "
        f"{tuple(target.shape)}"
    )


def _can_pad_dim0(source: torch.Tensor, target: torch.Tensor) -> bool:
    return (
        source.ndim == target.ndim
        and source.shape[0] <= target.shape[0]
        and tuple(source.shape[1:]) == tuple(target.shape[1:])
    )


def _shuffle_tensors(
    tensors: Sequence[torch.Tensor],
    split: int,
    dim: int,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("cannot shuffle an empty tensor list")
    if dim < 0:
        dim = tensors[0].ndim + dim
    if split <= 0:
        raise ValueError("split must be positive")

    base = tensors[0]
    ndim = base.ndim
    out_shape = list(base.shape)
    out_shape[dim] = sum(t.shape[dim] for t in tensors)
    out = torch.empty(out_shape, dtype=base.dtype, device=base.device)

    chunks = [t.shape[dim] // split for t in tensors]
    if any(t.shape[dim] % split != 0 for t in tensors):
        raise ValueError("all tensors must divide evenly by split")

    def slc(start: int, length: int):
        s = [slice(None)] * ndim
        s[dim] = slice(start, start + length)
        return tuple(s)

    write_base = 0
    per_group = sum(chunks)
    for group in range(split):
        write_off = 0
        for tensor, chunk in zip(tensors, chunks):
            out[slc(write_base + write_off, chunk)].copy_(
                tensor[slc(group * chunk, chunk)])
            write_off += chunk
        write_base += per_group
    return out
