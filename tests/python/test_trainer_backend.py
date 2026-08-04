import sys
import types

import pytest

torch = pytest.importorskip("torch")

from mirage.mpk.trainer_backend import (
    HuggingFaceTrainerBackend,
    MegatronTrainerBackend,
    TorchTitanTrainerBackend,
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


class _TinyNativeLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.projection = torch.nn.Linear(8, 16, bias=False)

    def forward(self, input_ids):
        return self.projection(self.embedding(input_ids))


class _TinyMegatronLM(_TinyNativeLM):
    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels=None,
        runtime_gather_output=False,
    ):
        del position_ids, attention_mask, labels
        assert runtime_gather_output
        return super().forward(input_ids)


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


def test_hf_backend_does_not_transpose_square_batch_logits():
    torch.manual_seed(13)
    model = _TinyCausalLM()
    backend = HuggingFaceTrainerBackend(
        model, _Tokenizer(), torch.optim.SGD(model.parameters(), lr=0.1)
    )
    samples = [
        {"ids": [1, 3], "pos": [1]},
        {"ids": [2, 4], "pos": [1]},
    ]

    values = backend.selected_token_logprobs(samples)
    logits = model(
        input_ids=torch.tensor([[1, 3], [2, 4]]),
        attention_mask=torch.ones(2, 2, dtype=torch.long),
        use_cache=False,
    ).logits
    expected = torch.log_softmax(logits.float(), dim=-1)[
        torch.arange(2), 0, torch.tensor([3, 4])
    ]
    assert torch.allclose(torch.cat(values), expected)


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
            "unknown",
            model_name="model",
            tokenizer=_Tokenizer(),
            learning_rate=1e-6,
        )


def test_torchtitan_backend_runs_native_replay_and_optimizer():
    pytest.importorskip("torchtitan")
    from torchtitan.components.optimizer import OptimizersContainer

    torch.manual_seed(19)
    model = _TinyNativeLM()
    optimizers = OptimizersContainer(
        [model], torch.optim.SGD, {"lr": 0.1}
    )
    backend = TorchTitanTrainerBackend(
        [model], _Tokenizer(), optimizers, micro_batch_size=1
    )
    samples = [
        {"ids": [1, 3, 4], "pos": [1, 2]},
        {"ids": [1, 5, 6, 7], "pos": [2, 3]},
    ]

    values = backend.selected_token_logprobs(samples)
    assert [len(value) for value in values] == [2, 2]
    before = model.projection.weight.detach().clone()
    backend.zero_grad()
    grad_norm = backend.backward_and_step(torch.cat(values).sum())
    assert grad_norm > 0
    assert not torch.equal(before, model.projection.weight)


def test_megatron_backend_runs_gathered_replay_and_native_step_contract():
    torch.manual_seed(23)
    model = _TinyMegatronLM()

    class FakeMegatronOptimizer:
        __module__ = "megatron.core.optimizer"

        def __init__(self, parameters):
            self.inner = torch.optim.SGD(parameters, lr=0.1)

        def zero_grad(self, set_to_none=True):
            self.inner.zero_grad(set_to_none=set_to_none)

        def step(self):
            self.inner.step()
            return True, torch.tensor(2.5), 0

    finalized = []
    backend = MegatronTrainerBackend(
        [model],
        _Tokenizer(),
        FakeMegatronOptimizer(model.parameters()),
        finalize_grads=lambda chunks: finalized.append(chunks),
    )
    samples = [{"ids": [1, 3, 4, 5], "pos": [2, 3]}]

    values = backend.selected_token_logprobs(samples)
    before = model.projection.weight.detach().clone()
    backend.zero_grad()
    grad_norm = backend.backward_and_step(values[0].sum())
    assert grad_norm == 2.5
    assert finalized == [[model]]
    assert not torch.equal(before, model.projection.weight)


def test_builtin_backend_aliases_accept_prebuilt_native_stacks():
    model = _TinyNativeLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    titan = create_trainer_backend(
        "torchtitan",
        model_name="unused",
        tokenizer=_Tokenizer(),
        learning_rate=1e-6,
        factory_kwargs={"model_parts": [model], "optimizers": optimizer},
    )
    assert isinstance(titan, TorchTitanTrainerBackend)

    megatron_model = _TinyMegatronLM()
    megatron = create_trainer_backend(
        "megatron",
        model_name="unused",
        tokenizer=_Tokenizer(),
        learning_rate=1e-6,
        factory_kwargs={
            "model_chunks": [megatron_model],
            "optimizer": torch.optim.SGD(megatron_model.parameters(), lr=0.1),
        },
    )
    assert isinstance(megatron, MegatronTrainerBackend)


def test_megatron_auto_builder_rejects_non_ddp_optimizer_stack():
    with pytest.raises(ValueError, match="wrap_with_ddp=True"):
        create_trainer_backend(
            "megatron",
            model_name="unused",
            tokenizer=_Tokenizer(),
            learning_rate=1e-6,
            factory_kwargs={"wrap_with_ddp": False},
        )


@pytest.mark.parametrize(
    "backend_cls, model",
    [
        (TorchTitanTrainerBackend, _TinyNativeLM()),
        (MegatronTrainerBackend, _TinyMegatronLM()),
    ],
)
def test_native_backends_reject_pipeline_model_parts(backend_cls, model):
    with pytest.raises(ValueError, match="pipeline|PP"):
        if backend_cls is TorchTitanTrainerBackend:
            backend_cls(
                [model, _TinyNativeLM()],
                _Tokenizer(),
                torch.optim.SGD(model.parameters(), lr=0.1),
            )
        else:
            backend_cls(
                [model, _TinyMegatronLM()],
                _Tokenizer(),
                torch.optim.SGD(model.parameters(), lr=0.1),
            )
