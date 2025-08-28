import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from wordfreq import zipf_frequency

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
base_dir = "C:/Users/liel.farber/Documents/Tranformers Project/pythia_1b_checkpoints"
revisions = [
    "step64", "step256", "step1000", "step3000", "step18000", "step48000",
    "step78000", "step108000", "step143000"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 300000               # number of tokens to sample from vocabulary
output_dir = "layer0_token_freq_convergence"
layer_to_probe = 0      # which hidden layer to use
EPS = 0.45

FREQ_MIN, FREQ_MAX = 0.0, 7+EPS  # keep words with freq in this range

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(os.path.join(base_dir, revisions[0]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def sample_token_ids(tokenizer, N):
    vocab = tokenizer.get_vocab()
    all_ids = list(vocab.values())
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    elig_ids = [tid for tid in all_ids if tid not in special_ids and tid >= 0]
    if N > len(elig_ids):
        N = len(elig_ids)
    return random.sample(elig_ids, N)

def token_zipf_frequency(tokenizer, token_id):
    s = tokenizer.decode([token_id]).strip().lower()
    if not s:
        return 0.0
    return float(zipf_frequency(s, "en"))

def bucket_tokens_by_frequency_eps(tokenizer, token_ids, eps=EPS):
    """
    Bucket token IDs into integer buckets i where freq ∈ [i-eps, i+eps].
    Keeps only tokens with freq in [FREQ_MIN, FREQ_MAX].
    """
    buckets = {i: [] for i in range(int(FREQ_MIN), int(FREQ_MAX) + 1)}
    for tid in token_ids:
        f = token_zipf_frequency(tokenizer, tid)
        if f > FREQ_MAX or f < FREQ_MIN:
            continue
        for i in buckets.keys():
            if (i - eps) <= f <= (i + eps):
                buckets[i].append(tid)
                break   # assign token to the first matching bucket

    # remove empty buckets
    buckets = {k: v for k, v in buckets.items() if len(v) > 0 and k in [1, 2, 3, 4, 5, 6, 7]}
    return buckets

def balance_buckets(buckets, min_size, max_size):
    """
    Force each bucket to have size between [min_size, max_size].
    - If bucket has more than max_size → randomly downsample to max_size.
    - If bucket has fewer than min_size → drop that bucket entirely.
    """
    balanced = {}
    for k, v in buckets.items():
        if len(v) < min_size:
            # skip too-small buckets
            continue
        elif len(v) > max_size:
            balanced[k] = random.sample(v, max_size)
        else:
            balanced[k] = v
    return balanced


def get_hidden_states_for_tokens(token_ids, layer, tokenizer):
    hidden_dict = {tid: [] for tid in token_ids}
    for rev in tqdm(revisions, desc="Processing revisions"):
        model_path = os.path.join(base_dir, rev)
        model = GPTNeoXForCausalLM.from_pretrained(model_path).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id

        batch_size = 512
        for i in range(0, len(token_ids), batch_size):
            batch_tids = token_ids[i:i+batch_size]
            input_ids = torch.tensor(batch_tids, dtype=torch.long, device=device).unsqueeze(1)
            attention_mask = torch.ones_like(input_ids, device=device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
                hs_layer = outputs.hidden_states[layer]  # (B,1,H)
                reps = hs_layer[:, -1, :].detach().cpu().numpy()

            for tid, vec in zip(batch_tids, reps):
                hidden_dict[tid].append(vec)

        del model
        torch.cuda.empty_cache()
    return hidden_dict

def compute_similarity_trajectories(hidden_dict):
    trajectories = []
    expected_revs = len(revisions)
    for vecs in hidden_dict.values():
        if len(vecs) != expected_revs:
            continue
        reps = np.stack(vecs, axis=0)
        sims = cosine_similarity(reps[:-1], reps[-1:]).flatten()
        trajectories.append(sims)
    if not trajectories:
        return np.empty((0, expected_revs - 1))
    return np.stack(trajectories, axis=0)

def plot_bucket_convergence(all_results):
    os.makedirs(output_dir, exist_ok=True)
    x_labels = revisions[:-1]
    x = np.arange(len(x_labels))

    plt.figure(figsize=(11, 6))
    for sims, label in sorted(all_results, key=lambda t: t[1]):
        if sims.size > 0:
            mean_curve = sims.mean(axis=0)
            plt.plot(x, mean_curve, marker="o", label=f"Zipf≈{label}")

    plt.title(f"Token Convergence by Zipf-Frequency Buckets (Layer {layer_to_probe})")
    plt.xlabel("Revision")
    plt.ylabel("Cosine Similarity to Final Revision")
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.grid(True)
    plt.legend(title="Buckets", ncol=2, fontsize=9)
    plt.tight_layout()
    outpath = os.path.join(output_dir, "token_freq_bucket_convergence.png")
    plt.savefig(outpath)
    plt.close()

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer = get_tokenizer()
    sampled_token_ids = sample_token_ids(tokenizer, N)

    # Filter + bucket by frequency
    freq_buckets = bucket_tokens_by_frequency_eps(tokenizer, sampled_token_ids, EPS)

    # Balance
    balanced_buckets = balance_buckets(freq_buckets, min_size=50, max_size=1000)

    print("Final bucket sizes:")
    for k, v in sorted(balanced_buckets.items()):
        print(f"Zipf≈{k}: {len(v)}")

    # Compute + plot
    results = []
    for bucket_label, tids in sorted(balanced_buckets.items()):
        hs = get_hidden_states_for_tokens(tids, layer=layer_to_probe, tokenizer=tokenizer)
        sims = compute_similarity_trajectories(hs)
        results.append((sims, bucket_label))

    plot_bucket_convergence(results)
