import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.long_kv_cache import LongKVCache
from streaming_llm.utils import parse_args, load

device = "cuda"

args = parse_args()
# python examples/eval_long_ppl.py --num_samples=10

data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None
past_key_values_mask = None

# if args.enable_start_recent_kv_cache or args.enable_start_full_kv_cache:
if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    elif "mpt" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = LongKVCache(
        batch_size=1,  ### Fix lol
        n_layers=model.config.n_layer,
        n_heads=model.config.n_head,
        start_size=args.start_size,
        cache_size=args.cache_size,  ### Need to add as arguement
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )

else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        from streaming_llm.pos_shift.modify_mpt import (
            enable_mpt_masked_cache,
        )

        enable_mpt_masked_cache(model)
    else:
        raise ValueError(f"got {model.config.model_type}")


os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for text in data["text"][: args.num_samples]:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])
    ### Shouldn't encodings.input_ids be 1 dimensional? A number for every word? dim0 is batch size then?

    seq_len = encodings.input_ids.size(1)
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        ### We only pass in 1 token at a time...
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                past_key_values_mask=past_key_values_mask,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values_new = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                kv_cache.add_recent_to_cache(past_key_values_new, 1)
                past_key_values, past_key_values_mask = kv_cache.get_kv_mask()
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
