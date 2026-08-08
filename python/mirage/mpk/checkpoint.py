"""Atomic checkpoint/restore for the MPK GRPO training loop.

A checkpoint is a single ``torch.save`` file written atomically (temp file in
the target directory + ``os.replace``) containing everything needed to resume
an outer-step GRPO run exactly:

- ``trainer``: the trainer backend's ``state_dict()`` (model weights and
  optimizer state; backend-defined layout),
- ``outer_step``: number of COMPLETED outer steps (resume starts here),
- ``data_cursor``: dataset cursor (equals ``outer_step`` for the sequential
  GSM8K reader in e19),
- ``rng``: host RNG states (python ``random``, torch CPU, all torch CUDA
  devices, and numpy when available),
- ``config``: JSON-serializable echo of the run args, with a small set of
  keys verified on resume.

The MPK engine's own sampling is position-keyed and seeded (``--sampling-seed``)
rather than host-RNG-driven, so engine rollouts after restore are re-armed
purely by syncing the restored trainer weights into the engine's attached
parameter tensors (the loop's existing weight-sync step).
"""

from __future__ import annotations

import os
import random
import tempfile
from typing import Mapping

import torch

FORMAT_VERSION = 1

# Config keys that must match between the checkpointing run and the resuming
# run for the resumed trajectory to be exactly the uninterrupted one.
STRICT_CONFIG_KEYS = (
    "model",
    "grpo_arm",
    "grpo_trainer_backend",
    "max_num_batched_requests",
    "max_num_batched_tokens",
    "max_seq_length",
    "sampling_seed",
    "deterministic",
    "inner_epochs",
    "grpo_lr",
    "grpo_trainer_micro_batch_size",
)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint holds CUDA RNG states for "
                f"{len(cuda_states)} device(s) but {torch.cuda.device_count()} "
                "are visible; resume with the same CUDA_VISIBLE_DEVICES"
            )
        torch.cuda.set_rng_state_all(cuda_states)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        import numpy as np

        np.random.set_state(numpy_state)


def config_echo(args) -> dict:
    """JSON-serializable snapshot of the run configuration."""
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def verify_config(saved: Mapping[str, object], args) -> None:
    mismatches = []
    current = config_echo(args)
    for key in STRICT_CONFIG_KEYS:
        if key not in saved and key not in current:
            continue
        if saved.get(key) != current.get(key):
            mismatches.append(
                f"  {key}: checkpoint={saved.get(key)!r} run={current.get(key)!r}"
            )
    if mismatches:
        raise ValueError(
            "checkpoint was written under a different configuration; a "
            "resumed run would not reproduce the uninterrupted trajectory:\n"
            + "\n".join(mismatches)
        )


def save_checkpoint(path: str, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically (temp file + rename)."""
    payload = dict(payload)
    payload["format_version"] = FORMAT_VERSION
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".tmp."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(path: str) -> dict:
    # RNG states and the config echo are plain Python objects, so full
    # (non-weights-only) unpickling is required; checkpoints are trusted
    # local artifacts of this loop.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format_version {version!r} "
            f"(expected {FORMAT_VERSION})"
        )
    return payload
