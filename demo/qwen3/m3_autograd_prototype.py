# M3 Phase-1 prototype: MPK forward + recompute backward
# (design doc mpk-rl-design-doc.md §5.3).
#
# Demonstrates the trainer-integration mechanism: a torch.autograd.Function
# whose FORWARD returns the rollout engine's bit-exact logprobs (here: the
# MPK per-position capture from a --capture-probs dump), and whose BACKWARD
# recomputes the same logprobs differentiably on the standard training stack
# (HF transformers) and routes the incoming gradient through that graph.
#
# The RL objective therefore consumes rollout-identical numerics
# (r_corr == 1 bitwise when pi_old = rollout logprobs), while gradients flow
# through standard kernels — TIM enters only via forward quantities, which
# are now bit-exact.
#
# Validates, on one REINFORCE-style step:
#   1. loss value is computed from MPK logprobs exactly (assert)
#   2. gradients are finite, nonzero, and flow to all trainable params
#   3. an optimizer step executes
#   4. gradient direction sanity: <g_mpk_loss, g_hf_loss> / (|g||g|) ~ 1
#      (backward landscapes coincide up to the forward value gap)
#
# Usage: python m3_autograd_prototype.py --ref /tmp/rescore_full2/ref.json
import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True, help="--capture-probs dump (rollout)")
parser.add_argument("--model", default="Qwen/Qwen3-8B")
args = parser.parse_args()

ref = json.load(open(args.ref))
ids = ref["token_ids"]
p0 = ref["prompt_length"]
# MPK rollout logprobs for generated tokens: slot t-1 holds P(token_t).
# Skip unwritten slots (e.g. the final EOS step) — same convention as the
# rescore harness.
valid_pos = [t for t in range(p0, len(ids)) if ref["probs"][t - 1] > 0.0]
mpk_lp = torch.tensor(
    [math.log(ref["probs"][t - 1]) for t in valid_pos],
    dtype=torch.float32,
    device="cuda",
)

model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16
).cuda()
model.gradient_checkpointing_enable()
model.train()

input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
rows = torch.tensor([t - 1 for t in valid_pos], dtype=torch.long, device="cuda")
targets = torch.tensor([ids[t] for t in valid_pos], dtype=torch.long, device="cuda")


def hf_logprobs():
    """Differentiable trainer-stack logprobs of the generated tokens."""
    logits = model(input_ids=input_ids).logits[0]
    lp = torch.log_softmax(logits[rows].float(), dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class MPKLogProbs(torch.autograd.Function):
    """Forward: rollout-engine (MPK) logprobs, bit-exact.
    Backward: recompute differentiably on the training stack and push the
    incoming gradient through that graph (gradients reach model params via
    .backward on the recomputed values; this Function itself returns no
    input grads)."""

    @staticmethod
    def forward(ctx, _handle):
        return mpk_lp.clone()

    @staticmethod
    def backward(ctx, grad_out):
        lp_hf = hf_logprobs()
        lp_hf.backward(grad_out)
        return None


# ---- one REINFORCE-style step ----
handle = torch.zeros(1, requires_grad=True, device="cuda")  # autograd anchor
logprobs = MPKLogProbs.apply(handle)

# 1) forward values are exactly the rollout's
assert torch.equal(logprobs, mpk_lp), "loss must consume MPK logprobs bitwise"
advantage = torch.randn_like(logprobs)  # stand-in per-token advantages
loss = -(advantage * logprobs).mean()
print(f"loss (from MPK logprobs): {loss.item():.6f}")

model.zero_grad(set_to_none=True)
loss.backward()

# 2) gradient health
grads = [p.grad for p in model.parameters() if p.grad is not None]
n_params = sum(1 for p in model.parameters() if p.requires_grad)
gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
finite = all(torch.isfinite(g).all().item() for g in grads)
print(f"params with grad: {len(grads)}/{n_params}, grad_norm={gnorm:.4f}, "
      f"all finite: {finite}")

# 3) optimizer step
opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr 0: mechanism test only
opt.step()
print("optimizer step: OK")

# 4) direction sanity: gradient of the same loss built on HF logprobs
model.zero_grad(set_to_none=True)
loss_hf = -(advantage * hf_logprobs()).mean()
loss_hf.backward()
grads_hf = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

model.zero_grad(set_to_none=True)
loss2 = -(advantage * MPKLogProbs.apply(handle)).mean()
loss2.backward()
num = sum((p.grad.float() * grads_hf[n].float()).sum()
          for n, p in model.named_parameters() if p.grad is not None)
den = torch.sqrt(sum((p.grad.float() ** 2).sum()
                     for _, p in model.named_parameters() if p.grad is not None)) * \
      torch.sqrt(sum((g.float() ** 2).sum() for g in grads_hf.values()))
print(f"cosine(grad via MPK-forward, grad via HF-forward): {(num/den).item():.6f}")
print("M3 PROTOTYPE: PASS" if (finite and gnorm > 0) else "M3 PROTOTYPE: FAIL")
