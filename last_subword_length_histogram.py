import os
import json
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
CHECKPOINT_PATH = "C:/Users/liel.farber/Documents/Tranformers Project/pythia_1b_checkpoints/step64"


def load_words(words_file, sample_size):
    """Load words from JSON and randomly sample."""
    with open(words_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_words = list(data.keys())
    return random.sample(all_words, sample_size)


def bucket_words_by_token_count(words, tokenizer, verbose=False):
    """Group words by number of tokens according to the tokenizer."""
    token_buckets = defaultdict(list)
    for w in tqdm(words, desc="Bucketing words"):
        tokens = tokenizer.tokenize(w)
        token_buckets[len(tokens)].append(w)

    token_buckets = {k: v for k, v in token_buckets.items() if len(v) >= 10}

    if verbose:
        print("\nToken count distribution:")
        for k in sorted(token_buckets):
            print(f"  {k}-token words: {len(token_buckets[k])}")

    return dict(token_buckets)


def compute_last_subword_lengths(words, tokenizer):
    """Get the length (in characters) of the last subword for each word."""
    lengths = []
    for w in words:
        tokens = tokenizer.tokenize(w)
        last_subword = tokens[-1]
        lengths.append(len(last_subword))
    return lengths


def plot_histogram(lengths, token_count, output_dir):
    """Plot and save histogram of last subword lengths for a token-count group."""
    mean_len = sum(lengths) / len(lengths)
    std_len  = (sum((x - mean_len) ** 2 for x in lengths) / len(lengths)) ** 0.5

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=range(1, max(lengths) + 2), alpha=0.7,
             color="skyblue", edgecolor="black")
    plt.title(f"Last Subword Lengths • {token_count}-token words")
    plt.xlabel("Subword Length (characters)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # legend
    legend_text = f"Mean length = {mean_len:.2f}   •   Std = {std_len:.2f}"
    plt.legend([legend_text], loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"last_subword_lengths_{token_count}_tokens.png")
    plt.savefig(save_path)
    plt.close()

    


if __name__ == "__main__":
    os.makedirs("last_subword_length_histograms", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    sampled_words = load_words("words_list.json", 30000)

    token_buckets = bucket_words_by_token_count(sampled_words, tokenizer, True)

    for token_count, words in tqdm(sorted(token_buckets.items()), desc="Token-count groups"):
        lengths = compute_last_subword_lengths(words, tokenizer)
        plot_histogram(lengths, token_count, "last_subword_length_histograms")
