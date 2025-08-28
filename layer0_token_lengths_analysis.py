import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, GPTNeoXForCausalLM, AutoConfig

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
base_dir = "C:/Users/liel.farber/Documents/Tranformers Project/pythia_1b_checkpoints"
revisions = [
    "step64", "step256", "step1000", "step3000", "step18000",  "step48000",
    "step78000", "step108000", "step143000"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = os.path.join(base_dir, revisions[0])
config = AutoConfig.from_pretrained(config_path)

# ──────────────────────────────────────────────────────────────
# Generate random tokens and bucket by string length
# ──────────────────────────────────────────────────────────────
def generate_random_tokens(tokenizer, n=10000, seed=42):
    random.seed(seed)
    vocab = list(tokenizer.get_vocab().keys())
    special_ids = set(tokenizer.all_special_ids)
    valid_tokens = [t for t in vocab if tokenizer.convert_tokens_to_ids(t) not in special_ids]
    chosen = random.sample(valid_tokens, min(n, len(valid_tokens)))
    return chosen

from collections import defaultdict

def bucket_by_length(tokens, min_size=100, allowed_lengths=None):
    """
    Group tokens by their string length.
    Only keep buckets of size ≥ min_size and whose length is in allowed_lengths (if provided).
    
    Args:
        tokens (list[str]): Input tokens.
        min_size (int): Minimum number of tokens required for a bucket to be kept.
        allowed_lengths (list[int] | None): List of token lengths to include.
                                            If None, all lengths are allowed.
    """
    buckets = defaultdict(list)
    for t in tokens:
        buckets[len(t)].append(t)

    filtered = {
        length: group
        for length, group in buckets.items()
        if len(group) >= min_size and (allowed_lengths is None or length in allowed_lengths)
    }

    # Print distribution
    print("\nToken length distribution:")
    for length, group in sorted(filtered.items()):
        print(f"  Length {length}: {len(group)} tokens")

    return filtered


# ──────────────────────────────────────────────────────────────
# Extract hidden states (layer 0 only)
# ──────────────────────────────────────────────────────────────
def get_hidden_states(tokens, layer, tokenizer):
    hidden_dict = {tok: [] for tok in tokens}
    for rev in tqdm(revisions, desc="Processing revisions"):
        model_path = os.path.join(base_dir, rev)
        model = GPTNeoXForCausalLM.from_pretrained(model_path).to(device)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        batch_size = 64
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i+batch_size]
            inputs = tokenizer(batch_tokens, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                for j, tok in enumerate(batch_tokens):
                    vec = outputs.hidden_states[layer][j, -1, :].cpu().numpy()
                    hidden_dict[tok].append(vec)

        del model
        torch.cuda.empty_cache()
    return hidden_dict

# ──────────────────────────────────────────────────────────────
# Compute convergence trajectories
# ──────────────────────────────────────────────────────────────
def compute_similarity_trajectories(hidden_dict):
    trajectories = []
    for reps in hidden_dict.values():
        reps = np.stack(reps)  # (R, D)
        sims = cosine_similarity(reps[:-1], reps[-1:]).flatten()
        trajectories.append(sims)
    return np.array(trajectories)  # (W, R-1)

# ──────────────────────────────────────────────────────────────
# Plot convergence for all buckets in single plot (layer 0 only)
# ──────────────────────────────────────────────────────────────
def plot_all_convergence_curves(all_sim_dicts):
    os.makedirs("layer0_token_length_convergence_plots", exist_ok=True)
    x_labels = revisions[:-1]
    x = np.arange(len(x_labels))

    plt.figure(figsize=(10, 6))
    for length, sims in sorted(all_sim_dicts.items()):
        mean_curve = sims.mean(axis=0)
        plt.plot(x, mean_curve, marker="o", label=f"Len {length}")

    plt.title("Mean Cosine Similarity to Final Revision • Layer 0")
    plt.xlabel("Revision")
    plt.ylabel("Cosine Similarity")
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("layer0_token_length_convergence_plots", "layer0_all_lengths.png"))
    plt.close()

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tok0 = AutoTokenizer.from_pretrained(os.path.join(base_dir, revisions[0]))
    tok0.pad_token = tok0.eos_token

    # Generate random tokens
    tokens = generate_random_tokens(tok0, n=5000)

    # Only keep string lengths at most 10
    tokens_lengths = [2,4,6,8]
    buckets = bucket_by_length(tokens, 100, tokens_lengths)

    # Compute hidden states + similarities
    all_sim_dicts = {}
    for length, toks in tqdm(buckets.items(), desc="Processing length buckets"):
        hs = get_hidden_states(toks, layer=0, tokenizer=tok0)
        sims = compute_similarity_trajectories(hs)
        all_sim_dicts[length] = sims

    # Plot results
    plot_all_convergence_curves(all_sim_dicts)
