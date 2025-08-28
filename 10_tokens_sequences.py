import os
import torch
import numpy as np
import json
from transformers import AutoTokenizer
import random
from tqdm import tqdm
from transformers import AutoConfig
from token_convergence_analysis import get_hidden_states, compute_similarity_trajectories, plot_similarity_by_layer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
base_dir = "C:/Users/liel.farber/Documents/Tranformers Project/pythia_1b_checkpoints"
revisions = [
    "step64", "step256", "step1000", "step3000", "step18000",
    "step48000", "step78000", "step108000", "step143000"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok0 = AutoTokenizer.from_pretrained(os.path.join(base_dir, revisions[0]))
tok0.pad_token = tok0.eos_token

config_path = os.path.join(base_dir, revisions[0])
config = AutoConfig.from_pretrained(config_path)
All_layers = list(range(config.num_hidden_layers))

# ─────────────────────────────────────────────────────────────────────────────
# Generate random 10-token sequences
# ─────────────────────────────────────────────────────────────────────────────
def generate_random_sequences(num_sequences=5000, seq_len=10):
    vocab_size = tok0.vocab_size
    ids = np.random.randint(0, vocab_size, size=(num_sequences, seq_len))
    return [tok0.decode(seq_ids) for seq_ids in ids]

# ─────────────────────────────────────────────────────────────────────────────
# Collect words by token count
# ─────────────────────────────────────────────────────────────────────────────
def bucket_words_by_token_count(words, tokenizer, min_size=10, verbose=False):
    token_buckets = {}
    for w in words:
        tok_count = len(tokenizer.tokenize(w))
        token_buckets.setdefault(tok_count, []).append(w)

    # filter
    token_buckets = {k: v for k, v in token_buckets.items() if len(v) >= min_size}

    if verbose:
        print("Token count distribution:")
        for k in sorted(token_buckets):
            print(f"  {k}-token words: {len(token_buckets[k])}")

    return token_buckets


# ─────────────────────────────────────────────────────────────────────────────
# Create modified sequences
# ─────────────────────────────────────────────────────────────────────────────
def create_modified_sequences(base_sequences, token_buckets, groups=5):
    """
    For each i = 1..groups:
        Take |bucket[i]| base sequences (in order) and replace last i tokens
        with the words from bucket[i] (one-to-one mapping).
    """
    modified = {i: [] for i in range(1, groups + 1)}
    start_idx = 0

    for i in range(1, groups + 1):
        if i not in token_buckets:
            continue  # no words for this group

        words = token_buckets[i]
        count = len(words)
        end_idx = start_idx + count

        # Make sure we have enough base sequences
        if end_idx > len(base_sequences):
            raise ValueError(
                f"Not enough base sequences: need {end_idx}, "
                f"but only have {len(base_sequences)}"
            )

        seq_slice = base_sequences[start_idx:end_idx]

        for seq, replacement_word in zip(seq_slice, words):
            seq_ids = tok0(seq, add_special_tokens=False)["input_ids"]
            if len(seq_ids) != 10:
                continue  # skip malformed

            replacement_ids = tok0(replacement_word, add_special_tokens=False)["input_ids"]
            if len(replacement_ids) != i:
                continue  # safety check

            # Replace last i tokens with replacement word
            new_ids = seq_ids[:-i] + replacement_ids
            assert len(new_ids) == 10, f"Expected 10 tokens, got {len(new_ids)}"

            new_text = tok0.decode(new_ids)
            modified[i].append(new_text)

        start_idx = end_idx  # advance slice window

    return modified


def run_experiment(modified_groups, layers):
    """
    Run convergence experiment on modified sequences grouped by token-count replacement.

    Args:
        modified_groups (dict): {i: [seq1, seq2, ...]} where i is token-count group (1..5).
        layers (list[int]): Layers at which to extract hidden states.
    """
    all_sim_dicts = dict()

    for token_count, sequences in tqdm(sorted(modified_groups.items()), desc="Processing token count groups"):
        if not sequences:
            continue

        print(f"\nGroup {token_count} tokens: {len(sequences)} sequences")

        # Extract hidden states
        hs = get_hidden_states(sequences, layers)

        # Compute similarity trajectories
        print("Compute similarities...")
        sim = compute_similarity_trajectories(hs)

        all_sim_dicts[token_count] = sim

    # Plot results
    plot_similarity_by_layer(all_sim_dicts, "10_tokens_sequences_convergence_plots")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load & sample words
    with open("words_list.json", "r", encoding="utf-8") as f:
        word_data = json.load(f)
    words = list(word_data.keys())
    sampled_words = random.sample(words, 5000)

    # Bucket words
    token_buckets = bucket_words_by_token_count(sampled_words, tok0, min_size=10, verbose=True)

    # Generate base sequences
    total_needed = sum(len(token_buckets[i]) for i in range(1, 6) if i in token_buckets)
    base_sequences = generate_random_sequences(total_needed, 10)

    # Create modified sequences
    modified_groups = create_modified_sequences(base_sequences, token_buckets, groups=5)

    # Run experiment
    run_experiment(modified_groups, All_layers)


