"""Llama-3 model builder for the MPK online engine.

Llama-3 is architecturally Qwen3 minus q/k layernorm (plus llama-3 rope
scaling and, for the 3.2 line, tied word embeddings).  All three deltas
are handled generically by :class:`Qwen3Builder`:

- no q/k-norm: ``build_layers`` detects the missing ``q_norm``/``k_norm``
  state-dict keys and wires a dummy norm tensor with
  ``enable_qk_norm=False``;
- rope theta / llama3 rope scaling: position embeddings come from the HF
  model's own ``rotary_emb`` module;
- tied lm_head: ``build_from_dict`` falls back to
  ``model.embed_tokens.weight`` when ``lm_head.weight`` is absent;
- vocab padding (128256 -> 129024) and eos id come from the generic
  ``_padded_vocab_size`` / tokenizer hooks.

So this subclass only supplies the HF model class and registry names.
Inherits the full online_pinned feature set: deterministic splitk gating
(cc 100/103), capture_logprobs prob buffer, sampling_partial with
seed/penalties, and ignore_eos.
"""

from ..qwen3.builder import Qwen3Builder
from ...model_registry import register_model_builder


@register_model_builder(
    "Llama3",
    "llama3",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-3B",
    "NousResearch/Meta-Llama-3.1-8B",
)
class Llama3Builder(Qwen3Builder):
    def __init__(self, mpk, weights=None):
        super().__init__(mpk, weights)
        # Overwritten from the tokenizer in build_from_model; this default
        # only matters for build_from_config flows.
        self.eos_token_id = 128001

    @staticmethod
    def _hf_model_class():
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        return LlamaForCausalLM
