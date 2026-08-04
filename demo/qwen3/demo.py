from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json

from models.qwen3_shard_loader import Qwen3ShardLoader
from mirage.mpk.base_dynamic_shard_loader import ShardType


mapping = {
    "embed_tokens": {"name": "embed", "shard_type": [(ShardType.NONE,)]},
    "input_layernorm": {"name": "attn_norm", "shard_type": [(ShardType.NONE,)]},
    "q_proj" : {"name": "wq", "shard_type": [(ShardType.COL_PARALLEL,)]},
	"q_norm" : {"name": "wq", "shard_type": [(ShardType.NONE,)]}, 
    "k_proj": {"name": "wk", "shard_type": [(ShardType.COL_PARALLEL,)]},
    "k_norm": {"name": "wk", "shard_type": [(ShardType.NONE,)]},
    "v_proj": {"name": "wv", "shard_type": [(ShardType.COL_PARALLEL,)]},
	"o_proj" : {"name": "wo", "shard_type": [(ShardType.ROW_PARALLEL)]},
    "post_attention_layernorm": {"name": "post_norm", "shard_type": [(ShardType.NONE)]}, 
	"gate": {"name": "gate", "shard_type": [(ShardType.NONE)]}, # router gate
	"gate_proj": {"name": "w1", "shard_type": [(ShardType.COL_PARALLEL)]}, 
	"down_proj": {"name": "w2", "shard_type": [(ShardType.ROW_PARALLEL)]}, 
	"up_proj": {"name": "w3", "shard_type": [(ShardType.COL_PARALLEL)]}, 
    "norm": {"name": "norm", "shard_type": [(ShardType.NONE)]},
    "lm_head": {"name": "head", "shard_type": [(ShardType.NONE)]}
}

DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")
MAX_SAVE_TOKENS = 100

# print limitation
# torch.set_printoptions(threshold=2000)

def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = True):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size % 64 == 0 and not use_cutlass_kernel:
        # TODO(Wenqin): If we set OUTPUT_SIZE too much for PTX linear kernel,
        # there is some regression.
        return size // 64
    if size / 96 > 400:
        # TODO: An add-hoc workaround for linear kernel, both MPK ptx and
        # cutlass version will output unexpected result (not same output for
        # same prompt) if the OUTPUT_SIZE is too big, try to figure it out.
        assert size % 256 == 0, "FATAL: Linear layer size not supported, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor


def load_ref_model(model_name, world_size, max_num_pages=16, page_size=4096):
    """Load a frozen reference model (its own weights + KV cache)."""
    from models.modeling_qwen3 import Qwen3ForCausalLM
    m = Qwen3ForCausalLM.from_pretrained(
        model_name, world_size, max_num_pages=max_num_pages,
        page_size=page_size).to("cuda")
    m.eval()
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=512,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-8B', help="Model path on hugging face"
    )
    parser.add_argument(
        "--no-use-cutlass-kernel",
        action="store_false",
        dest="use_cutlass_kernel",
        default=True,
        help="Not use the cutlass version kernel.",
    )
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore eos token during generation")
    parser.add_argument("--grpo-steps", type=int, default=0,
                        help="Run the E19 GRPO loop for this many steps.")
    parser.add_argument("--grpo-arm", choices=["mpk", "hf"], default="mpk")
    parser.add_argument("--grpo-lr", type=float, default=2e-6)
    parser.add_argument("--grpo-log", type=str, default=None)
    parser.add_argument(
        "--grpo-trainer-micro-batch-size",
        type=int,
        default=0,
        help="Trainer replay micro-batch size; 0 batches the full GRPO group.",
    )
    parser.add_argument(
        "--grpo-trainer-backend",
        default="hf",
        help="Backward backend: 'hf', 'torchtitan', 'megatron', or a "
        "lazy '<module>:<factory>' plugin.",
    )
    parser.add_argument(
        "--grpo-measure-old-recompute",
        action="store_true",
        help="Measure the eliminated trainer-side old-logprob recompute pass. "
        "Disabled by default so it is not charged to E2E engine time.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Sample tokens with Gumbel-Max using this seed instead of "
        "greedy argmax (deterministic given the seed).",
    )
    parser.add_argument(
        "--capture-probs",
        action="store_true",
        help="Capture P(chosen token) at every step into a per-position "
        "buffer (softmax_gather + prob_scatter) and include it in "
        "--dump-tokens-file output.",
    )
    parser.add_argument(
        "--no-fused-sampling-capture",
        action="store_false",
        dest="fused_sampling_capture",
        default=True,
        help="Use the standalone probability-capture task after sampling; "
        "intended for correctness and performance A/B runs.",
    )
    parser.add_argument(
        "--no-parallel-sampling",
        action="store_false",
        dest="parallel_sampling",
        default=True,
        help="Use the single-block vocabulary scan for sampling A/B runs.",
    )
    parser.add_argument(
        "--prompt-ids-file",
        type=str,
        default=None,
        help="JSON file with a list of prompt token ids; bypasses the chat "
        "template and tokenizer (for rescore-consistency experiments).",
    )
    parser.add_argument(
        "--dump-tokens-file",
        type=str,
        default=None,
        help="Write all token ids of request 0 (prompt + generated) to this "
        "JSON file after generation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Bitwise-deterministic and batch-invariant decoding: split-K "
        "linears store partials to a dedicated buffer and combine them in "
        "fixed order, instead of tma_reduce_add accumulation in "
        "task-completion order.",
    )

    # -------- Args for CI tests ----------
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Decode cap for CI determinism")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument(
        "--reference", action="store_true",
        help="Co-compile a reference-model forward into the SAME task graph "
        "and capture its per-token logprobs (KL-penalty reference pass with "
        "no second engine). With --reference-model unset the reference "
        "shares the policy checkpoint, so ref logprobs must equal the "
        "policy's bit-for-bit -- a correctness check for the second forward.")
    parser.add_argument(
        "--reference-model", type=str, default=None,
        help="HF name/path of the frozen reference model (defaults to the "
        "policy checkpoint for validation).")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", help="Enable sampling (default off)")
    parser.add_argument(
        "--save-tokens",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Optionally dump first N generated token_ids, text, and latency to JSON. "
            "If path omitted, saves to outputs/qwen3/{torch_output.json|mpk_output.json}."
        ),
    )
    parser.add_argument("--prompt",
        type=str,
        default="Give me a short introduction to large language model.",
        help="Custom prompt text to generate from.",
    )

    parser.add_argument("--split-kv-cache", action="store_true", help="Use split-kv cache")
    args = parser.parse_args()
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = None

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    if args.model_path is not None or world_size == 1:
      with torch.device("cuda"):
          if args.model_path is not None:
              # load model locally (necessary for multi-GPU case)
              print(f"Load model from model path: {args.model_path}")
              config = AutoConfig.from_pretrained(args.model_path)
              model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)
              load_model(
                  model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
              )
              # model = Qwen3ForCausalLM.from_pretrained(args.model_path, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
              tokenizer = AutoTokenizer.from_pretrained(args.model_path)
          else:
              model = Qwen3ForCausalLM.from_pretrained(model_name, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
              tokenizer = AutoTokenizer.from_pretrained(model_name)
    else: # Use dynamic shard loader to load directly from HF and shard.
        print("Detected multi-GPU run without a local path specified. Will use the DynamicShardLoader class.")
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(model_name)
            model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)

        device = torch.device(f"cuda:{rank}")
        loader = Qwen3ShardLoader(model, model_name, mapping, rank, world_size, device)
        loader.load()

        with torch.device("cuda"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    prompt = args.prompt
    # This prompt is copied from https://github.com/apoorvumang/prompt-lookup-decoding/blob/main/demo-pld.ipynb
    code_text = """import numpy as np
                import matplotlib.pyplot as plt

                # Calculate the average
                average_throughput = np.mean(tokens_per_sec_arr)
                print(f"Average Throughput: {average_throughput} tokens/sec")

                # Plotting the histogram
                plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
                plt.title('Histogram of Throughput Values')
                plt.xlabel('Tokens per Second')
                plt.ylabel('Frequency')
                plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
                plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
                plt.show()
                """
    #question = "Can you please change x axis to start from 0"
    #prompt = code_text + "\n" + question
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    if args.prompt_ids_file:
        # Feed the prompt as raw token ids (no chat template, no
        # re-tokenization) — required by the decode-vs-rescore consistency
        # harness, where a previous run's tokens become the next run's prompt.
        with open(args.prompt_ids_file) as f:
            prompt_ids = json.load(f)
        model_inputs.input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=model.device
        )
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device="cuda")
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    # get all model weight tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prob_buffer_torch = torch.zeros(
        (args.max_num_batched_tokens, args.max_seq_length),
        dtype=torch.float32,
        device="cuda",
    )
    ref_prob_buffer_torch = torch.zeros(
        (args.max_num_batched_tokens, args.max_seq_length),
        dtype=torch.float32,
        device="cuda",
    )
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")

    if args.use_mirage:
        import mirage as mi

        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        # pad vocab_size to facilitate task graph creation
        lm_head_weight = torch.cat(
            (
                model.lm_head.weight,
                torch.full(
                    (153600 - model.config.vocab_size, hidden_size), 0, device="cuda"
                ),
            ),
            0,
        )
        assert lm_head_weight.stride()[0] == hidden_size
        vocab_size = 153600
        num_q_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        num_local_q_heads = num_q_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = model.config.head_dim
        fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused_outdim_2 = 2 * intermediate_size
        num_kv_cache_chunks = max(1, args.max_seq_length // 256)

        if args.profiling:
            profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None
            
        spec_decode_config = mi.mpk.spec_decode_class(
            args.spec_decode,
            ngram_size=args.ngram_size,
            spec_length=args.spec_length,
        )
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indices_buffer = torch.empty(
            args.max_num_pages, dtype=torch.int32, device="cuda")
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda")
        mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=model.config.eos_token_id if not args.ignore_eos else -1,
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=args.use_cutlass_kernel
        )
        
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            all_tokens = mpk.attach_input(torch_tensor=tokens, name="all_tokens")
            num_tokens_extend = spec_decode_config.spec_length + 1
        else:
            num_tokens_extend = 1
        
        # TODO: Make the code run well even if 96 % max_num_batched_tokens != 0
        # assert(96 % args.max_num_batched_tokens == 0)
        
        x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
        cos_pos_embed = mpk.attach_input(
            torch_tensor=position_embeddings[0][0, :4096, :],
            name="cos_position_embedding",
        )
        sin_pos_embed = mpk.attach_input(
            torch_tensor=position_embeddings[1][0, :4096, :],
            name="sin_position_embedding",
        )

        y = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="embed_out",
            io_category="cuda_tensor",
        )
        rmsnorm_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )
        attn_in = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size), # [6, 6144]
            dtype=mi.bfloat16,
            name="attn_in",
            io_category="cuda_tensor",
        )
        lse = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads),
            dtype=mi.float32,
            name="lse",
            io_category="cuda_tensor",
        )
        attn_out_tmp = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out_tmp",
            io_category="cuda_tensor",
        )
        attn_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_local_q_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        attn_proj_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="attn_proj_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        allreduce_buf = mpk.new_tensor(
            dims=(world_size, args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="all_reduce_buf",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        attn_allreduce_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="attn_allreduce_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_mid = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, fused_outdim_2 // world_size),
            dtype=mi.bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        silu_mul_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, intermediate_size // world_size),
            dtype=mi.bfloat16,
            name="silu_mul_out",
            io_category="cuda_tensor",
        )
        mlp_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_final = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_final",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        argmax_in = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, vocab_size),
            dtype=mi.bfloat16,
            name="argmax_in",
            io_category="cuda_tensor",
        )
        argmax_part_value = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, mpk.num_workers),
            dtype=mi.bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        argmax_part_index = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, mpk.num_workers),
            dtype=mi.int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )
        argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")
        #argmax_out = mpk.new_tensor(
        #    dims=(args.max_num_batched_tokens, 1),
        #    dtype=mi.int64,
        #    name="argmax_out",
        #    io_category="cuda_tensor",
        #)

        # add spec tokens layer
        if spec_decode_config:
            spec_tokens = mpk.draft_forward_layer_dispatcher(
                spec_decode_config = spec_decode_config, 
                tokens = all_tokens,
                grid_dim=(96, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = spec_tokens
        # Add Embed
        w = mpk.attach_input(
            torch_tensor=model.model.embed_tokens.weight, name="embed_tokens"
        )
        
        mpk.embed_layer(
            input=x, 
            weight=w, 
            output=y, 
            # grid_dim=(max_factor_leq_n(hidden_size, 96 // args.max_num_batched_tokens), total_tokens_per_iter, 1), 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1),
            input_source=1,
        )
        x = y
        target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        # A current workaround to use splitk for only B200 GPUs.
        # Split-K linear combines K-partials via tma_reduce_add (atomic, in
        # task-completion order), so results are NOT deterministic across
        # runs and NOT batch-invariant; --deterministic keeps split-K but
        # routes it through splitk_linear_det_layer, which stores partials
        # to a dedicated buffer and combines them in fixed order.
        use_splitk = (target_cc == 100)
        for i, layer in enumerate(model.model.layers):
            # if i > 0:
            #     break
            # add rmsnorm + linear
            w_norm = mpk.attach_input(
                torch_tensor=layer.input_layernorm.weight,
                name=f"layer_{i}_input_layernorm",
            )
            w_q = mpk.attach_input(
                torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj"
            )
            w_k = mpk.attach_input(
                torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj"
            )
            w_v = mpk.attach_input(
                torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj"
            )
            w_qkv = mpk.shuffle_tensors(
                inputs=[w_q, w_k, w_v],
                shuffled_dim=0,
                num_groups=model.config.num_key_value_heads // world_size,
                name=f"layer_{i}_qkv_proj",
            )
            mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_qkv,
                output=attn_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0), args.use_cutlass_kernel), 1, 1),
                block_dim=(128, 1, 1),
            )
            #mpk.rmsnorm_linear_layer(
            #    input=x,
            #    weight_norm=w_norm,
            #    weight_linear=w_qkv,
            #    output=attn_in,
            #    grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
            #    block_dim=(128, 1, 1),
            #)
            # add attention
            w_q_norm = mpk.attach_input(
                torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm"
            )
            w_k_norm = mpk.attach_input(
                torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm"
            )
            k_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache"
            ) 
            v_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache"
            )
            # TODO: Later attention kernels should be merged as one
            if spec_decode_config:
                mpk.single_batch_extend_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(1, num_local_kv_heads, 1), #TODO: further divide across batch dim
                    block_dim=(128, 1, 1),
                )
            elif args.split_kv_cache:
                mpk.paged_attention_split_kv_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    lse=lse,
                    output=attn_out_tmp,
                    attention_params=(num_local_q_heads, num_kv_cache_chunks),
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, num_kv_cache_chunks),
                    block_dim=(128, 1, 1),
                )

                mpk.paged_attention_split_kv_merge_layer(
                    lse=lse,
                    output_tmp=attn_out_tmp,
                    output=attn_out,
                    attention_params=(num_local_q_heads, head_dim),
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                )
            else:
                mpk.paged_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                )
            
            
            # add linear w/ residual
            w = mpk.attach_input(
                torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj"
            )
            if use_splitk and args.deterministic:
                num_splits = 128 * 128 // hidden_size
                o_proj_partials = mpk.new_tensor(
                    dims=(num_splits * args.max_num_batched_tokens, hidden_size),
                    dtype=mi.bfloat16,
                    name=f"layer_{i}_o_proj_partials",
                    io_category="cuda_tensor",
                )
                mpk.splitk_linear_det_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    partials=o_proj_partials,
                    output=attn_proj_out,
                    grid_dim=(hidden_size // 128, num_splits, 1),
                    block_dim=(256, 1, 1),
                    reduce_grid_dim=(hidden_size // 128, 1, 1),
                )
            elif use_splitk:
                attn_proj_out = x
                mpk.splitk_linear_layer(
                    input=attn_out,
                    weight=w,
                    output=attn_proj_out,
                    grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.linear_with_residual_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    output=attn_proj_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
            # reset residual input as x
            x = attn_proj_out
            # add allreduce if needed
            if world_size > 1:
                mpk.allreduce_layer(
                    input=attn_proj_out,
                    buffer=allreduce_buf,
                    output=attn_allreduce_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = attn_allreduce_out
            # add rmsnorm_linear layer
            w_norm = mpk.attach_input(
                torch_tensor=layer.post_attention_layernorm.weight,
                name=f"layer_{i}_post_attn_layernorm",
            )
            w_gate_proj = mpk.attach_input(
                torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj"
            )
            w_up_proj = mpk.attach_input(
                torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj"
            )
            rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0), args.use_cutlass_kernel)
            w_gatedup = mpk.shuffle_tensors(
                inputs=[w_gate_proj, w_up_proj],
                shuffled_dim=0,
                num_groups=rmsnorm_num_tasks//2,
                name=f"layer_{i}_gatedup_proj",
            )
            mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_gatedup,
                output=mlp_mid,
                grid_dim=(rmsnorm_num_tasks, 1, 1),
                block_dim=(128, 1, 1),
            )
            #mpk.rmsnorm_linear_layer(
            #    input=x,
            #    weight_norm=w_norm,
            #    weight_linear=w_gatedup,
            #    output=mlp_mid,
            #    grid_dim=(rmsnorm_num_tasks, 1, 1),
            #    block_dim=(128, 1, 1),
            #)
            mpk.silu_mul_layer(
                input=mlp_mid,
                output=silu_mul_out,
                grid_dim=(rmsnorm_num_tasks//2, 1, 1),
                block_dim=(128, 1, 1),
            )
            # add silu_mul_linear layer
            w = mpk.attach_input(
                torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj"
            )
            if use_splitk and args.deterministic:
                num_splits = 128 * 128 // hidden_size
                down_proj_partials = mpk.new_tensor(
                    dims=(num_splits * args.max_num_batched_tokens, hidden_size),
                    dtype=mi.bfloat16,
                    name=f"layer_{i}_down_proj_partials",
                    io_category="cuda_tensor",
                )
                mpk.splitk_linear_det_layer(
                    input=silu_mul_out,
                    weight=w,
                    residual=x,
                    partials=down_proj_partials,
                    output=mlp_out,
                    grid_dim=(hidden_size // 128, num_splits, 1),
                    block_dim=(256, 1, 1),
                    reduce_grid_dim=(hidden_size // 128, 1, 1),
                )
            elif use_splitk:
                mlp_out = x
                mpk.splitk_linear_layer(
                    input=silu_mul_out,
                    weight=w,
                    output=mlp_out,
                    grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.linear_with_residual_layer(
                    input=silu_mul_out,
                    weight=w,
                    residual=x,
                    output=mlp_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
            # reset residual input as x
            x = mlp_out
            if world_size > 1:
                mpk.allreduce_layer(
                    input=mlp_out,
                    buffer=allreduce_buf,
                    output=mlp_final,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = mlp_final

        # add rmsnorm_linear layer
        w_norm = mpk.attach_input(
            torch_tensor=model.model.norm.weight, name="model_norm_weight"
        )
        w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
        mpk.rmsnorm_layer(
            input=x,
            weight=w_norm,
            output=rmsnorm_out,
            grid_dim=(mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.linear_layer(
            input=rmsnorm_out,
            weight=w_proj,
            output=argmax_in,
            grid_dim=(mpk.num_workers, 1, 1),
            block_dim=(128, 1, 1),
        )
        #mpk.rmsnorm_linear_layer(
        #    input=x,
        #    weight_norm=w_norm,
        #    weight_linear=w_proj,
        #    output=argmax_in,
        #    grid_dim=(grid_for_rmsnorm_linear_layer(w_proj.dim(0)), 1, 1),
        #    block_dim=(128, 1, 1),
        #)
        # add argmax layer
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            argmax_partial_grid_dim = (max_factor_leq_n(153600, 96 // (spec_decode_config.spec_length + 1)), 
                                       spec_decode_config.spec_length + 1, 
                                       1)
            argmax_reduce_grid_dim = (1, spec_decode_config.spec_length + 1, 1)
        else:
            argmax_partial_grid_dim = (mpk.num_workers, 1, 1)
            argmax_reduce_grid_dim = (1, 1, 1)
        prob_buffer = None
        prompt_lengths_for_prob = None
        if args.capture_probs:
            prob_buffer = mpk.attach_input(
                torch_tensor=prob_buffer_torch, name="prob_buffer"
            )
            prompt_lengths_for_prob = mpk.attach_input(
                torch_tensor=prompt_lengths, name="prompt_lengths_for_prob"
            )

        sampling_capture_fused = False
        if args.sampling_seed is not None:
            # Gumbel-Max sampling (deterministic given the seed; the philox
            # offset advances with the runtime step so every step draws
            # fresh noise). When logprobs are requested, sampling also reuses
            # this vocab scan for the softmax max and emits the probability;
            # only the fixed-order exp-sum scan remains.
            sampling_temperature = (
                args.temperature if args.temperature > 0 else 1.0
            )
            use_parallel_sampling = (
                args.parallel_sampling and args.top_k == 0 and args.top_p == 1.0
            )
            if use_parallel_sampling:
                mpk.sampling_partial_sm100_layer(
                    logits=argmax_in,
                    output=(argmax_part_value, argmax_part_index),
                    grid_dim=(mpk.num_workers, 1, 1),
                    block_dim=(256, 1, 1),
                    seed=args.sampling_seed,
                    temperature=sampling_temperature,
                )
                mpk.argmax_reduce_layer(
                    input=(argmax_part_value, argmax_part_index),
                    output=argmax_out,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )
            else:
                sampling_capture_fused = (
                    args.capture_probs and args.fused_sampling_capture
                )
                mpk.sampling_sm100_layer(
                    logits=argmax_in,
                    output=argmax_out,
                    grid_dim=(total_num_requests, 1, 1),
                    block_dim=(256, 1, 1),
                    seed=args.sampling_seed,
                    temperature=sampling_temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    prompt_lengths=(
                        prompt_lengths_for_prob
                        if sampling_capture_fused else None
                    ),
                    prob_buffer=(
                        prob_buffer if sampling_capture_fused else None
                    ),
                )
        else:
            mpk.argmax_partial_layer(
                input=argmax_in,
                output=(argmax_part_value, argmax_part_index),
                grid_dim=argmax_partial_grid_dim,
                block_dim=(128, 1, 1),
            )
            mpk.argmax_reduce_layer(
                input=(argmax_part_value, argmax_part_index),
                output=argmax_out,
                grid_dim=argmax_reduce_grid_dim,
                block_dim=(128, 1, 1),
            )
        if args.capture_probs and not sampling_capture_fused:
            # Per-step probability capture for the rescore-consistency
            # harness: P(chosen token) = softmax(logits)[argmax_out],
            # scattered into prob_buffer[0, step]. Only row 0 (request 0,
            # single-token decode) is meaningful; prefill-chunk iterations
            # write don't-care values at prefill positions.
            # unified per-token probability capture: teacher-forcing rows
            # (prefill/rescore) and the generating row (decode) in ONE task
            # with qo-derived row->request mapping. Replaces the earlier
            # softmax_gather+prob_scatter pair, whose row-indexed scatter
            # miscaptured once requests in a batch finished at different
            # times and the decode window re-packed.
            mpk.prefill_prob_capture_layer(
                logits=argmax_in,
                prompt_lengths=prompt_lengths_for_prob,
                chosen_tokens=argmax_out,
                buffer=prob_buffer,
                page_size=args.page_size,
                grid_dim=(total_num_requests, 1, 1),
            )
        if spec_decode_config:
            verify_out = mpk.verify_layer_dispatcher(
                spec_decode_config = spec_decode_config,
                spec_tokens = spec_tokens,
                target_output = argmax_out,
                grid_dim = (1, 1, 1),
                block_dim = (128, 1, 1),
            )

        if args.reference:
            assert args.capture_probs, "--reference requires --capture-probs"
            # ── Reference-model forward, co-compiled into the SAME tGraph ──
            # A second full forward (embed -> layers -> norm -> lm_head) over
            # the same input token, with the reference model's weights and
            # its OWN paged KV cache, feeding a teacher-forcing capture that
            # gathers P_ref(chosen token) into ref_prob_buffer. No sampling
            # (the reference only scores). Its tasks share the SMs with the
            # policy decode -- the KL reference pass with no second engine.
            ref_model = model if args.reference_model is None else \
                load_ref_model(args.reference_model, world_size,
                               args.max_num_pages, args.page_size)
            # ALWAYS a fresh physical KV cache (same shape / paged metadata):
            # even when ref shares the policy checkpoint, it must not write
            # into the policy's cache slots -- that would alias the two
            # forwards' K/V. Shape mirrors model.model.kv_cache.
            _pkc, _pvc = model.model.kv_cache
            ref_kv = (torch.empty_like(_pkc), torch.empty_like(_pvc))
            def rin(t, name):
                return mpk.attach_input(torch_tensor=t, name=name)
            rx = rin(input_tokens, "input_token_ref")
            ry = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size),
                                dtype=mi.bfloat16, name="embed_out_ref",
                                io_category="cuda_tensor")
            r_rms = mpk.new_tensor(dims=(args.max_num_batched_tokens, hidden_size),
                                   dtype=mi.bfloat16, name="rmsnorm_out_ref",
                                   io_category="cuda_tensor")
            r_attn_in = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size),
                dtype=mi.bfloat16, name="attn_in_ref", io_category="cuda_tensor")
            r_attn_out = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, num_local_q_heads * head_dim),
                dtype=mi.bfloat16, name="attn_out_ref", io_category="cuda_tensor")
            r_attn_proj = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, hidden_size),
                dtype=mi.bfloat16, name="attn_proj_out_ref",
                io_category="cuda_tensor")
            r_mlp_mid = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, fused_outdim_2 // world_size),
                dtype=mi.bfloat16, name="mlp_mid_ref", io_category="cuda_tensor")
            r_silu = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, intermediate_size // world_size),
                dtype=mi.bfloat16, name="silu_mul_out_ref",
                io_category="cuda_tensor")
            r_logits = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, vocab_size),
                dtype=mi.bfloat16, name="argmax_in_ref", io_category="cuda_tensor")

            rw_embed = rin(ref_model.model.embed_tokens.weight, "embed_tokens_ref")
            mpk.embed_layer(input=rx, weight=rw_embed, output=ry,
                            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
                            input_source=1)
            rx_h = ry
            for i, layer in enumerate(ref_model.model.layers):
                p = f"ref_layer_{i}_"
                w_norm = rin(layer.input_layernorm.weight, p + "input_ln")
                w_q = rin(layer.self_attn.q_proj.weight, p + "q_proj")
                w_k = rin(layer.self_attn.k_proj.weight, p + "k_proj")
                w_v = rin(layer.self_attn.v_proj.weight, p + "v_proj")
                w_qkv = mpk.shuffle_tensors(
                    inputs=[w_q, w_k, w_v], shuffled_dim=0,
                    num_groups=model.config.num_key_value_heads // world_size,
                    name=p + "qkv_proj")
                mpk.rmsnorm_layer(input=rx_h, weight=w_norm, output=r_rms,
                                  grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                                  block_dim=(128, 1, 1))
                mpk.linear_layer(
                    input=r_rms, weight=w_qkv, output=r_attn_in,
                    grid_dim=(grid_for_rmsnorm_linear_layer(
                        w_qkv.dim(0), args.use_cutlass_kernel), 1, 1),
                    block_dim=(128, 1, 1))
                w_q_norm = rin(layer.self_attn.q_norm.weight, p + "q_norm")
                w_k_norm = rin(layer.self_attn.k_norm.weight, p + "k_norm")
                r_kc = rin(ref_kv[0][i], p + "k_cache")
                r_vc = rin(ref_kv[1][i], p + "v_cache")
                mpk.paged_attention_layer(
                    input=r_attn_in, k_cache=r_kc, v_cache=r_vc,
                    q_norm=w_q_norm, k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed, sin_pos_embed=sin_pos_embed,
                    output=r_attn_out,
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1))
                w_o = rin(layer.self_attn.o_proj.weight, p + "o_proj")
                if use_splitk and args.deterministic:
                    num_splits = 128 * 128 // hidden_size
                    o_part = mpk.new_tensor(
                        dims=(num_splits * args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16, name=p + "o_partials",
                        io_category="cuda_tensor")
                    mpk.splitk_linear_det_layer(
                        input=r_attn_out, weight=w_o, residual=rx_h,
                        partials=o_part, output=r_attn_proj,
                        grid_dim=(hidden_size // 128, num_splits, 1),
                        block_dim=(256, 1, 1),
                        reduce_grid_dim=(hidden_size // 128, 1, 1))
                elif use_splitk:
                    r_attn_proj = rx_h
                    mpk.splitk_linear_layer(
                        input=r_attn_out, weight=w_o, output=r_attn_proj,
                        grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1),
                        block_dim=(256, 1, 1))
                else:
                    mpk.linear_with_residual_layer(
                        input=r_attn_out, weight=w_o, residual=rx_h,
                        output=r_attn_proj, grid_dim=(hidden_size // 64, 1, 1),
                        block_dim=(128, 1, 1))
                rx_h = r_attn_proj
                w_pnorm = rin(layer.post_attention_layernorm.weight,
                              p + "post_ln")
                w_gate = rin(layer.mlp.gate_proj.weight, p + "gate")
                w_up = rin(layer.mlp.up_proj.weight, p + "up")
                n_tasks = grid_for_rmsnorm_linear_layer(
                    w_gate.dim(0) + w_up.dim(0), args.use_cutlass_kernel)
                w_gu = mpk.shuffle_tensors(inputs=[w_gate, w_up], shuffled_dim=0,
                                           num_groups=n_tasks // 2,
                                           name=p + "gatedup")
                mpk.rmsnorm_layer(input=rx_h, weight=w_pnorm, output=r_rms,
                                  grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                                  block_dim=(128, 1, 1))
                mpk.linear_layer(input=r_rms, weight=w_gu, output=r_mlp_mid,
                                 grid_dim=(n_tasks, 1, 1), block_dim=(128, 1, 1))
                mpk.silu_mul_layer(input=r_mlp_mid, output=r_silu,
                                   grid_dim=(n_tasks // 2, 1, 1),
                                   block_dim=(128, 1, 1))
                w_down = rin(layer.mlp.down_proj.weight, p + "down")
                if use_splitk and args.deterministic:
                    num_splits = 128 * 128 // hidden_size
                    d_part = mpk.new_tensor(
                        dims=(num_splits * args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16, name=p + "down_partials",
                        io_category="cuda_tensor")
                    mpk.splitk_linear_det_layer(
                        input=r_silu, weight=w_down, residual=rx_h,
                        partials=d_part, output=r_attn_proj,
                        grid_dim=(hidden_size // 128, num_splits, 1),
                        block_dim=(256, 1, 1),
                        reduce_grid_dim=(hidden_size // 128, 1, 1))
                    rx_h = r_attn_proj
                elif use_splitk:
                    mpk.splitk_linear_layer(
                        input=r_silu, weight=w_down, output=rx_h,
                        grid_dim=(hidden_size // 128, 128 * 128 // hidden_size, 1),
                        block_dim=(256, 1, 1))
                else:
                    r_mlp_out = mpk.new_tensor(
                        dims=(args.max_num_batched_tokens, hidden_size),
                        dtype=mi.bfloat16, name=p + "mlp_out",
                        io_category="cuda_tensor")
                    mpk.linear_with_residual_layer(
                        input=r_silu, weight=w_down, residual=rx_h,
                        output=r_mlp_out, grid_dim=(hidden_size // 64, 1, 1),
                        block_dim=(128, 1, 1))
                    rx_h = r_mlp_out
            w_fnorm = rin(ref_model.model.norm.weight, "ref_norm")
            ref_lm_head = torch.cat(
                (ref_model.lm_head.weight,
                 torch.full((vocab_size - model.config.vocab_size, hidden_size),
                            0, device="cuda", dtype=torch.bfloat16)), 0)
            w_head = rin(ref_lm_head, "ref_lm_head")
            mpk.rmsnorm_layer(input=rx_h, weight=w_fnorm, output=r_rms,
                              grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                              block_dim=(128, 1, 1))
            mpk.linear_layer(input=r_rms, weight=w_head, output=r_logits,
                             grid_dim=(mpk.num_workers, 1, 1),
                             block_dim=(128, 1, 1))
            ref_prob_buffer = mpk.attach_input(
                torch_tensor=ref_prob_buffer_torch, name="ref_prob_buffer")
            # chosen_tokens re-attaches output_tokens as a fresh graph INPUT
            # ("ref_chosen") rather than reusing the sampling task's output
            # DTensor. Reusing argmax_out would make the sampling task feed
            # BOTH captures -- a fork the annotated-graph builder rejects
            # (a task cannot be both join- and fork-producer). Teacher-forcing
            # rows (which we validate) read the committed token buffer, not
            # chosen_tokens, so this input carries no scheduling dependency.
            ref_chosen = mpk.attach_input(
                torch_tensor=output_tokens, name="ref_chosen")
            # order_dep = the policy prob_buffer: forces this capture to run
            # after the policy capture (hence after sampling wrote
            # output_tokens), so ref_chosen reads the committed chosen tokens
            # -- without forking the sampling task.
            mpk.prefill_prob_capture_layer(
                logits=r_logits,
                prompt_lengths=mpk.attach_input(
                    torch_tensor=prompt_lengths, name="prompt_lengths_for_ref"),
                chosen_tokens=ref_chosen,
                buffer=ref_prob_buffer,
                page_size=args.page_size,
                grid_dim=(total_num_requests, 1, 1),
                order_dep=prob_buffer)

        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)

    # g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    warmup = 0
    # Decode up to user cap or buffer size
    output_len = args.max_new_tokens if args.max_new_tokens is not None else (tokens.size(1) - prompt_lengths[0].item())
    output_len = max(0, min(output_len, tokens.size(1) - prompt_lengths[0].item()))
    if not args.use_mirage:
        prompt_len = prompt_lengths[0].item()
        decode_limit = prompt_len + output_len
        for cur_pos in range(prompt_len, decode_limit):
            step.fill_(cur_pos - 1)
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            logits = model.forward(
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
                stream=stream,
            )
            next_token = logits.argmax(dim=-1)
            next_token = next_token[0, -1]
            tokens[0, cur_pos] = next_token
            prev_pos = cur_pos
            if next_token == model.config.eos_token_id:
                break
            if cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        end_idx = prev_pos + 1
        generated_ids = tokens[:, :end_idx]
        tokens_generated = max(0, end_idx - prompt_len)
        per_tok_ms = run_time / max(prompt_len + tokens_generated, 1)

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(
            "Prompt length {}, generate length {}, per-token latency {:.3f} ms".format(
                prompt_len, tokens_generated, per_tok_ms
            )
        )

        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            out = {
                "token_ids": token_ids,
                "text": tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True),
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    elif args.grpo_steps > 0:
        # E19: scaled-down GRPO stability experiment (see e19_grpo.py)
        import e19_grpo

        e19_grpo.run(
            args,
            mpk,
            model,
            tokenizer,
            tokens,
            step,
            prompt_lengths,
            num_new_tokens,
            prob_buffer_torch,
            model.config.eos_token_id,
        )
        import sys

        sys.exit(0)
    else:
        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        print("tokens.shape = ", tokens.shape)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(response)
        
        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        tokens_generated = step.max().item() + 1 - prompt_lengths[0].item()
        per_tok_ms = run_time / max(prompt_lengths[0].item() + tokens_generated, 1)

        print("Prompt length {}, generate length {}, per-token latency: {:.3f} ms".format(
              prompt_lengths[0], tokens_generated, per_tok_ms
            )
        )

        if args.dump_tokens_file and rank == 0:
            out = {
                "prompt_length": prompt_lengths[0].item(),
                "token_ids": tokens[0, : step[0].item() + 1].tolist(),
                "latency_ms_per_token": per_tok_ms,
            }
            if args.capture_probs:
                # raw float32 bit patterns so the harness can compare bitwise
                probs = prob_buffer_torch[0, : step[0].item() + 1]
                out["prob_bits"] = probs.view(torch.int32).tolist()
                out["probs"] = probs.tolist()
            if args.reference:
                rprobs = ref_prob_buffer_torch[0, : step[0].item() + 1]
                out["ref_prob_bits"] = rprobs.view(torch.int32).tolist()
                out["ref_probs"] = rprobs.tolist()
            with open(args.dump_tokens_file, "w") as f:
                json.dump(out, f)
            print(f"Dumped token ids to {args.dump_tokens_file}")

        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            end_idx = step[0].item() + 1
            prompt_len = prompt_lengths[0].item()
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = per_tok_ms
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            response_text = tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True)
            out = {
                "token_ids": token_ids,
                "text": response_text,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "mpk",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
