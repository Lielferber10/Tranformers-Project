import os
import random
import torch
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, GPTNeoXForCausalLM, AutoConfig
from wordfreq import zipf_frequency
import nltk
nltk.download('words')
nltk.download('wordnet')
from nltk.corpus import words
from nltk.corpus import wordnet as wn


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
english_words = set(words.words())

# ──────────────────────────────────────────────────────────────
# Generate random words
# ──────────────────────────────────────────────────────────────
def bucket_words_by_token_count(words, tokenizer, verbose=False, min_size=10):
    """
    Group words by token count according to the tokenizer.

    Args:
        words (list[str]): List of words to bucket.
        tokenizer: HuggingFace tokenizer.
        verbose (bool): If True, print count per token group.
        min_size (int): minimum number of words required per bucket.

    Returns:
        dict[int, list[str]]: Mapping from token count to word list.
    """
    token_buckets = defaultdict(list)
    for w in words:
        tok_count = len(tokenizer.tokenize(w))
        token_buckets[tok_count].append(w)

    # Filter out small buckets
    token_buckets = {k: v for k, v in token_buckets.items() if len(v) >= min_size}

    if verbose:
        print("Token count distribution:")
        for k in sorted(token_buckets):
            print(f"  {k}-token words: {len(token_buckets[k])}")

    return dict(token_buckets)


def split_decoded_tokens(tokens):
    """
    Split decoded tokens into:
      Group A: real English word (freq >= 5 or in wordlist, len>=2)
      Group B: non-word
    """
    group_A, group_B = [], []
    for t in tokens:
        freq = zipf_frequency(t, "en")
        if (t in english_words or freq >= 5.0) and len(t) >= 2:
            group_A.append(t)
        elif freq == 0.0:
            group_B.append(t)
    return group_A, group_B




def bucket_and_split_decoded(words, tokenizer, verbose=False, min_size=10):
    mapping = map_to_decoded(words, tokenizer)  # word → decoded
    token_buckets = bucket_words_by_token_count(words, tokenizer, verbose, min_size)

    split_buckets = {}
    for length, ws in token_buckets.items():
        # convert words → decoded tokens
        decoded_ws = [mapping[w] for w in ws if w in mapping]

        # split decoded tokens into groups A/B
        group_A, group_B = split_decoded_tokens(decoded_ws)

        # filter: only keep decoded tokens that tokenize back into a single token
        group_A = filter_single_token_decoded(group_A, tokenizer)
        group_B = filter_single_token_decoded(group_B, tokenizer)

        # only keep buckets with enough decoded tokens
        if len(group_A) >= 10 and len(group_B) >= 10:
            split_buckets[length] = {"A": group_A, "B": group_B}

    if verbose:
        print("\nDecoded buckets after A/B split:")
        for k, v in split_buckets.items():
            print(f"  {k}-token | A={len(v['A'])}, B={len(v['B'])}")

    return split_buckets






# ──────────────────────────────────────────────────────────────
# Extract hidden states (layer 0 only)
# ──────────────────────────────────────────────────────────────
def get_hidden_states(words, layer, tokenizer):
    hidden_dict = {w: [] for w in words}
    for rev in tqdm(revisions, desc="Processing revisions"):
        model_path = os.path.join(base_dir, rev)
        model = GPTNeoXForCausalLM.from_pretrained(model_path).to(device)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        batch_size = 64
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            inputs = tokenizer(batch_words, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                for j, w in enumerate(batch_words):
                    vec = outputs.hidden_states[layer][j, -1, :].cpu().numpy()
                    hidden_dict[w].append(vec)

        del model
        torch.cuda.empty_cache()
    return hidden_dict

# ──────────────────────────────────────────────────────────────
# Compute convergence trajectories
# ──────────────────────────────────────────────────────────────
def compute_similarity_trajectories(hidden_dict):
    trajectories = []
    expected_revs = len(revisions)  # number of checkpoints

    for reps in hidden_dict.values():
        reps = np.stack(reps)  # (N, D) where N should == expected_revs
        if reps.shape[0] != expected_revs:
            # skip tokens that didn’t appear in all revisions
            continue

        sims = cosine_similarity(reps[:-1], reps[-1:]).flatten()
        trajectories.append(sims)

    if not trajectories:
        return np.empty((0, expected_revs - 1))

    return np.stack(trajectories)  # (W, R-1)

# ──────────────────────────────────────────────────────────────
# Plot A vs B for each token-length bucket
# ──────────────────────────────────────────────────────────────
def plot_group_convergence(all_results):
    os.makedirs("layer0_last_token_realword_convergence", exist_ok=True)
    x_labels = revisions[:-1]
    x = np.arange(len(x_labels))

    plt.figure(figsize=(10, 6))

    # Store all A and B curves (for global averaging)
    all_A_sims, all_B_sims = [], []

    for length, group_dict in all_results.items():
        for group_name, group_data in group_dict.items():
            sims = group_data["sims"]
            if sims.size == 0:
                continue
            mean_curve = sims.mean(axis=0)

            # collect for global mean
            if group_name == "A":
                all_A_sims.append(mean_curve)
            else:
                all_B_sims.append(mean_curve)

            # plot individual length curves
            plt.plot(
                x,
                mean_curve,
                marker="o",
                label=f"Len {length} • Group {group_name}"
            )

    # Now compute and plot mean curves across lengths
    if all_A_sims:
        mean_A = np.mean(all_A_sims, axis=0)
        plt.plot(
            x, mean_A, marker="s", linewidth=3, linestyle="--",
            label="Mean Group A", color="blue"
        )
    if all_B_sims:
        mean_B = np.mean(all_B_sims, axis=0)
        plt.plot(
            x, mean_B, marker="s", linewidth=3, linestyle="--",
            label="Mean Group B", color="red"
        )

    plt.title("Convergence Curves (Layer 0)")
    plt.xlabel("Revision")
    plt.ylabel("Cosine Similarity to Final Revision")
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("layer0_last_token_realword_convergence", "all_lengths.png"))
    plt.close()

def to_decoded_words(words, tokenizer):
    decoded_words = []
    for w in words:
        toks = tokenizer.tokenize(w)
        if not toks:
            continue
        last_tok = toks[-1]
        decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(last_tok)).strip().lower()
        decoded_words.append(decoded)
    return decoded_words

def map_to_decoded(words, tokenizer):
    """
    Map each word to its decoded last token.
    Returns: dict[word -> decoded_token]
    """
    mapping = {}
    for w in words:
        toks = tokenizer.tokenize(w)
        if not toks:
            continue
        last_tok = toks[-1]
        decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(last_tok)).strip().lower()
        mapping[w] = decoded
    return mapping

def filter_single_token_decoded(tokens, tokenizer):
    """
    Keep only decoded words that map back to a single token.
    """
    filtered = []
    for t in tokens:
        toks = tokenizer.tokenize(t)
        if len(toks) == 1:
            filtered.append(t)
    return filtered


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tok0 = AutoTokenizer.from_pretrained(os.path.join(base_dir, revisions[0]))
    tok0.pad_token = tok0.eos_token

    # Load sentence dictionary
    with open("words_list.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract only the words (keys)
    words = list(data.keys())



    # Get random words (no repeats)
    sample_words = random.sample(words, 10000)


    # Create buckets on decoded words (not original words)
    buckets = bucket_and_split_decoded(sample_words, tok0, verbose=False, min_size=100)


    
    # Ignore sub group B of bucket 1 (correspondinng to single-token words)
    buckets[1]["B"] = []

    all_results = {}
    bucket_distributions = {}  # store counts + skip info

    for length, groups in tqdm(buckets.items(), desc="Processing length buckets"):
        group_A = groups["A"]
        group_B = groups["B"]

        skipped = (len(group_A) < 10 or len(group_B) < 10) and length != 1
        bucket_distributions[length] = (len(group_A), len(group_B), skipped)

        if skipped:
            continue

        results_for_length = {}
        for group_name, group_words in [("A", group_A), ("B", group_B)]:
            hs = get_hidden_states(group_words, layer=0, tokenizer=tok0)
            sims = compute_similarity_trajectories(hs)
            results_for_length[group_name] = {
                "sims": sims,
                "words": group_words
            }

        all_results[length] = results_for_length

    # Print distribution
    print("\nToken-count distribution:")
    for L in sorted(all_results.keys()):
        counts = {
            "A": len(all_results[L]["A"]["words"]) if "A" in all_results[L] else 0,
            "B": len(all_results[L]["B"]["words"]) if "B" in all_results[L] else 0,
        }
        total = counts["A"] + counts["B"]
        if L==1:
            print(f"  {L} tokens | total={total} | A={counts['A']} | B={0}")
        else:
            print(f"  {L} tokens | total={total} | A={counts['A']} | B={counts['B']}")

    # Plot
    plot_group_convergence(all_results)

