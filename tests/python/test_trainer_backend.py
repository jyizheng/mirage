import sys
import types

import pytest

torch = pytest.importorskip("torch")

from mirage.mpk.trainer_backend import (
    HuggingFaceTrainerBackend,
    bind_forward_values,
    create_trainer_backend,
)


def test_bind_forward_values_uses_authoritative_value_and_trainer_gradient():
    authoritative = torch.tensor([1.25, -3.5], dtype=torch.float32)
    differentiable = torch.tensor(
        [8.0, 9.0], dtype=torch.float32, requires_grad=True
    )

    output = bind_forward_values(authoritative, differentiable)
    assert torch.equal(output, authoritative)

    (output * torch.tensor([2.0, -4.0])).sum().backward()
    assert torch.equal(differentiable.grad, torch.tensor([2.0, -4.0]))


def test_bind_forward_values_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        bind_forward_values(torch.zeros(2), torch.zeros(3, requires_grad=True))


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.projection = torch.nn.Linear(8, 16, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        logits = self.projection(self.embedding(input_ids))
        return type("Output", (), {"logits": logits})


def test_hf_backend_batches_variable_length_selected_tokens():
    torch.manual_seed(7)
    model = _TinyCausalLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    backend = HuggingFaceTrainerBackend(model, _Tokenizer(), optimizer)
    samples = [
        {"ids": [1, 3, 4, 5], "pos": [2, 3]},
        {"ids": [1, 6, 7], "pos": [1, 2]},
    ]

    got = backend.selected_token_logprobs(samples)
    expected = []
    for sample in samples:
        ids = torch.tensor([sample["ids"]])
        logits = model(ids).logits[0]
        rows = torch.tensor([pos - 1 for pos in sample["pos"]])
        targets = torch.tensor([sample["ids"][pos] for pos in sample["pos"]])
        expected.append(
            torch.log_softmax(logits[rows].float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
        )

    assert len(got) == len(expected)
    for batched, single in zip(got, expected):
        assert torch.equal(batched, single)

    loss = torch.cat(got).sum()
    backend.zero_grad()
    grad_norm = backend.backward_and_step(loss)
    assert grad_norm > 0


def test_hf_backend_micro_batching_preserves_outputs():
    torch.manual_seed(11)
    model = _TinyCausalLM()
    samples = [
        {"ids": [1, 3, 4], "pos": [1, 2]},
        {"ids": [1, 5, 6, 7], "pos": [2, 3]},
        {"ids": [1, 8], "pos": [1]},
    ]
    full = HuggingFaceTrainerBackend(
        model, _Tokenizer(), torch.optim.SGD(model.parameters(), lr=0.1)
    )
    micro = HuggingFaceTrainerBackend(
        model,
        _Tokenizer(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        micro_batch_size=1,
    )

    full_values = full.selected_token_logprobs(samples)
    micro_values = micro.selected_token_logprobs(samples)
    for lhs, rhs in zip(full_values, micro_values):
        # Trainer kernels may choose a batch-shape-dependent reduction order.
        # MPK still owns the forward value through bind_forward_values; this
        # check only guards the differentiable replay's numerical agreement.
        torch.testing.assert_close(lhs, rhs, rtol=1e-6, atol=1e-6)


def test_create_trainer_backend_loads_external_factory_lazily():
    module = types.ModuleType("test_external_trainer")
    captured = {}

    class ExternalBackend:
        def selected_token_logprobs(self, samples):
            return []

        def zero_grad(self):
            pass

        def backward_and_step(self, loss):
            return 0.0

        def named_parameters(self):
            return iter(())

    def create(**kwargs):
        captured.update(kwargs)
        return ExternalBackend()

    module.create = create
    sys.modules[module.__name__] = module
    try:
        backend = create_trainer_backend(
            "test_external_trainer:create",
            model_name="model",
            tokenizer=_Tokenizer(),
            learning_rate=3e-6,
            micro_batch_size=2,
            factory_kwargs={"mesh": "dp=2"},
        )
    finally:
        del sys.modules[module.__name__]

    assert isinstance(backend, ExternalBackend)
    assert captured["model_name"] == "model"
    assert captured["learning_rate"] == 3e-6
    assert captured["micro_batch_size"] == 2
    assert captured["mesh"] == "dp=2"


def test_create_trainer_backend_rejects_unqualified_external_name():
    with pytest.raises(ValueError, match="module.*factory"):
        create_trainer_backend(
            "megatron",
            model_name="model",
            tokenizer=_Tokenizer(),
            learning_rate=1e-6,
        )
