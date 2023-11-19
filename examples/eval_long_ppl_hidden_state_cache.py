import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load
from streaming_llm.hidden_state_cache.hidden_state_cache import HiddenStateCache

device = "cuda"

args = parse_args()
# python examples/eval_long_ppl.py --num_samples=10

data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for text in data["text"][: args.num_samples]:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])
    seq_len = encodings.input_ids.size(1)
    print("seq_len: ", seq_len)

    batch_size = encodings.input_ids.size(0)
    if args.enable_start_recent_kv_cache:
        if "mpt" in model.config.model_type:
            seq_dim = 1
        else:
            raise ValueError(f"got {model.config.model_type}")
        hidden_state_cache = HiddenStateCache(
            batch_size=batch_size,
            n_layers=model.config.n_layers,
            keep_hidden_layer_idx=None,
            start_size=args.start_size,
            recent_size=args.recent_size,
            cache_size=args.cache_size,
            seq_dim=seq_dim,
        )
    else:
        hidden_state_cache = None

    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                hidden_state_cache=hidden_state_cache,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if hidden_state_cache is not None:  # update the cache!
                hidden_state_cache.add_to_cache(outputs.hidden_states)
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
