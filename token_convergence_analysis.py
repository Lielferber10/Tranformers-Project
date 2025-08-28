import os
import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoConfig
from collections import defaultdict
import json
import random
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
base_dir = "C:/Users/liel.farber/Documents/Tranformers Project/pythia_1b_checkpoints"
revisions = [
    "step64", "step256", "step1000", "step3000", "step18000",  "step48000",
    "step78000", "step108000", "step143000"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = os.path.join(base_dir, revisions[0])
config = AutoConfig.from_pretrained(config_path)
All_layers = list(range(config.num_hidden_layers))

# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────



def bucket_words_by_token_count(words, tokenizer, verbose=False):
    """
    Group words by token count according to the tokenizer.

    Args:
        words (list[str]): List of words to bucket.
        tokenizer: HuggingFace tokenizer.
        verbose (bool): If True, print count per token group.

    Returns:
        dict[int, list[str]]: Mapping from token count to word list.
    """
    token_buckets = defaultdict(list)
    for w in words:
        tok_count = len(tokenizer.tokenize(w))
        token_buckets[tok_count].append(w)

    # Filter out buckets with fewer than 10 words
    token_buckets = {k: v for k, v in token_buckets.items() if len(v) >= 10}

    if verbose:
        print("Token count distribution:")
        for k in sorted(token_buckets):
            print(f"  {k}-token words: {len(token_buckets[k])}")

    return dict(token_buckets)



def get_hidden_states(words, layers):
    """
    Returns:
        hidden_dict[word][layer] -> list[np.ndarray] (one per revision)
    """
    hidden_dict = {w: {l: [] for l in layers} for w in words}

    for rev in tqdm(revisions, desc="Processing revisions"):
        model_path = os.path.join(base_dir, rev)
        model = GPTNeoXForCausalLM.from_pretrained(model_path).to(device)
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.pad_token_id

        for word in tqdm(words, desc=f"Revision {rev} - extracting hidden states", leave=False):
            inputs = tok(word, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                for l in layers:
                    vec = outputs.hidden_states[l][0, -1, :].cpu().numpy()
                    hidden_dict[word][l].append(vec)

        del model, tok
        torch.cuda.empty_cache()

    return hidden_dict


def compute_similarity_trajectories(hidden_dict):
    """
    Returns:
        dict[layer] -> np.ndarray  shape (num_words, num_revisions-1)
    """
    results = {}
    for l in next(iter(hidden_dict.values())).keys():
        trajectories = []
        for layer_dict in hidden_dict.values():
            reps = np.stack(layer_dict[l])        # (R, D)
            sims = cosine_similarity(reps[:-1], reps[-1:]).flatten()
            trajectories.append(sims)
        results[l] = np.array(trajectories)       # (W, R-1)
    return results


def plot_similarity_by_layer(all_sim_dicts, results_dir_name):
    """
    Args:
        all_sim_dicts: dict[token_count][layer] -> np.ndarray (W, R-1)
        Each array holds similarity trajectories for W words in that token-count group.
    """
    os.makedirs(results_dir_name, exist_ok=True)

    layers = sorted({l for layer_dict in all_sim_dicts.values() for l in layer_dict.keys()})
    x_labels = revisions[:-1]
    x = np.arange(len(x_labels))

    for layer in layers:
        # Similarity plot (mean)
        plt.figure(figsize=(10, 6))
        for token_count, sim_dict in sorted(all_sim_dicts.items()):
            if layer not in sim_dict:
                continue

            sims = sim_dict[layer]  # shape: (W, R-1)
            mean_curve = sims.mean(axis=0)

            plt.plot(x, mean_curve, marker="o", label=f"{token_count}-token")

        plt.title(f"Mean Cosine Similarity to Final Revision • Layer {layer}")
        plt.xlabel("Revision")
        plt.ylabel("Cosine Similarity")
        plt.xticks(ticks=x, labels=x_labels, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir_name, f"layer_{layer}_similarity.png"))
        plt.close()

        # Derivative plot (mean ± STD)
        plt.figure(figsize=(10, 6))
        for token_count, sim_dict in sorted(all_sim_dicts.items()):
            if layer not in sim_dict:
                continue

            sims = sim_dict[layer]  # (W, R-1)

            # Compute per-word derivatives
            word_derivatives = np.diff(sims, axis=1)  # shape: (W, R-2)

            # Mean and std across words
            mean_deriv_curve = word_derivatives.mean(axis=0)
            std_deriv_curve = word_derivatives.std(axis=0)

            x_deriv = np.arange(len(mean_deriv_curve))
            x_labels_deriv = [f"{revisions[i]}→{revisions[i+1]}" for i in range(len(revisions)-2)]

            plt.plot(x_deriv, mean_deriv_curve, marker="o", label=f"{token_count}-token")
            plt.fill_between(x_deriv,
                             mean_deriv_curve - std_deriv_curve,
                             mean_deriv_curve + std_deriv_curve,
                             color=plt.gca().lines[-1].get_color(), alpha=0.2)

        plt.title(f"Mean Derivative of Cosine Similarity • Layer {layer}")
        plt.xlabel("Revision Transition")
        plt.ylabel("Δ Cosine Similarity")
        plt.xticks(ticks=np.arange(len(x_labels_deriv)), labels=x_labels_deriv, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir_name, f"layer_{layer}_derivatives.png"))
        plt.close()







# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_token_count_experiment(all_words, results_dir_name):
    # Use tokenizer from first checkpoint to tokenize all words
    tok0 = AutoTokenizer.from_pretrained(os.path.join(base_dir, revisions[0]))
    tok0.pad_token = tok0.eos_token

    # Bucket words by token count
    bucketed = bucket_words_by_token_count(all_words, tok0, verbose=True)

    # For each token count group: extract hidden states & compute similarities
    all_sim_dicts = dict()
    for token_count, words in tqdm(sorted(bucketed.items()), desc="Processing token count groups"):
        hs = get_hidden_states(words, All_layers)
        print("Compute similarities")
        sim = compute_similarity_trajectories(hs)
        all_sim_dicts[token_count] = sim

    
    plot_similarity_by_layer(all_sim_dicts, results_dir_name)



if __name__ == "__main__":
    # Load sentence dictionary
    with open("words_list.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract only the words (keys)
    words = list(data.keys())

    # Get random words (no repeats)
    sample_words = random.sample(words, 5000)
    run_token_count_experiment(sample_words, "convergence_plots")
