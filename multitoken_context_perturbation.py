"""
Token-embedding tracking experiment
-----------------------------------
Plots, per layer, the cosine similarity curves for

  • a base sentence that ends with a multi-token word
  • context variations of that sentence
  • perturbed versions (penultimate token randomised)

"""

import random
import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import csv


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
base_dir = "C:/Users/liel.farber/Documents/Tranformers Project/pythia_1b_checkpoints"
revisions = [
    "step64", "step256", "step1000", "step3000", "step18000",  "step48000",
    "step78000", "step108000", "step143000"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL + TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(path: str):
    model = (
        GPTNeoXForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


# ─────────────────────────────────────────────────────────────────────────────
# HIDDEN-STATE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def get_last_token_hs(sentence: str, *, model, tokenizer, sentence_tokens=None):
    """
    Returns list[layer] -> tensor[hidden_size] for the *last* token in `sentence`.
    """
    if sentence_tokens is None:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
    else:
        inputs = {k: v.to(device) for k, v in sentence_tokens.items()}

    outputs = model(**inputs, output_hidden_states=True)
    return [layer[0, -1, :] for layer in outputs.hidden_states]  # len == 17


def get_embedding_for_last_token(sentence: str, *, model, tokenizer):
    """
    Returns the *input embedding* vector (not contextual) of the last token.
    """
    ids = tokenizer(sentence, return_tensors="pt")["input_ids"][0]
    last_id = ids[-1].item()
    return model.get_input_embeddings().weight[last_id]


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()


def avg_similarity(
    base_hs,
    comparison_sentences,
    *,
    model,
    tokenizer
):
    """
    For each layer, compute the mean cosine similarity between `base_hs`
    and the last-token hidden states of `comparison_sentences`.
    """
    sims_per_layer = [[] for _ in range(len(base_hs))]

    for sent in comparison_sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(device)
        hs = get_last_token_hs(
            sent,
            model=model,
            tokenizer=tokenizer,
            sentence_tokens=inputs,
        )
        for i, (ref, cur) in enumerate(zip(base_hs, hs)):
            sims_per_layer[i].append(cosine_sim(ref, cur))

    return [sum(layer) / len(layer) for layer in sims_per_layer]


def average_curves(curves):
    """
    Stack a list of lists and return mean and std per layer.
    """
    arr = np.stack(curves)  # (N_words, N_layers)
    return arr.mean(axis=0), arr.std(axis=0)



# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(base, contexts_variants, word_variants, model, tokenizer):
    """
    Returns P1, P2 curves for `base` sentence.
    """
    base_hs = get_last_token_hs(base, model=model, tokenizer=tokenizer)

    P1 = avg_similarity(base_hs, contexts_variants, model=model, tokenizer=tokenizer)
    P2 = avg_similarity(base_hs, word_variants, model=model, tokenizer=tokenizer)
    return P1, P2


# ---------------------------------------------------------------------------
# Compute and cache curves  word -> revision -> (P1, P2)
# ---------------------------------------------------------------------------
def collect_all_curves(cut_sentences, words):
    """
    Returns dict[word][revision] = (P1, P2)
    """
    curves = {w: {} for w in words}

    for rev in revisions:
        print(f"Loading {rev} …")
        model_path = os.path.join(base_dir, rev)
        model, tok = load_model_and_tokenizer(model_path)

        for word in words:
            base = cut_sentences[word]["sentences"][0]
            context_variants =  cut_sentences[word]["sentences"][1:]
            word_variants =  cut_sentences[word]["replaced_sentences"]

            P1, P2 = run_experiment(base, context_variants, word_variants, model, tok)
            curves[word][rev] = (P1, P2)

        del model
        torch.cuda.empty_cache()

    return curves


# ---------------------------------------------------------------------------
# Plot for each word: one figure with |revisions| lines
# ---------------------------------------------------------------------------
def plot_word_grids(curves, cut_sentences):
    plots_dir = os.path.join(os.path.dirname(__file__), "multitoken_words_plots")

    for word, rev_dict in curves.items():
        # Get group based on token_split
        token_count = cut_sentences[word]["token_split"]
        group_dir = os.path.join(plots_dir, f"{token_count}-tokens words")
        os.makedirs(group_dir, exist_ok=True)

        n = len(rev_dict)
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
        axes = axes.flatten()

        # CSV path inside group dir
        csv_path = os.path.join(group_dir, f"{word}_data.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["revision", "layer", "P1", "P2"])  # header

            for ax, (rev, (P1, P2)) in zip(axes, rev_dict.items()):
                # Write table rows
                for layer_idx, (p1_val, p2_val) in enumerate(zip(P1, P2)):
                    writer.writerow([rev, layer_idx, p1_val, p2_val])

                # Plot curves
                ax.plot(P1, label='P1 (context)', marker='o')
                ax.plot(P2, label='P2 (perturbed)', marker='x')
                ax.set_title(rev)
                ax.grid(True)
                ax.set_xlabel("Layer")
                ax.set_ylabel("Cosine sim")
                ax.legend(fontsize="x-small")

            for ax in axes[n:]:
                ax.axis("off")

        fig.suptitle(f"P1 & P2 Curves for '{word}' across Revisions", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # PNG path inside group dir
        save_path = os.path.join(group_dir, f"{word}_curves_grid.png")

        plt.savefig(save_path)
        plt.close()


# ---------------------------------------------------------------------------
# Random sampling without replacement (lazily)
# ---------------------------------------------------------------------------
def random_sample_iterable(iterable, k):
    """Reservoir sampling: pick k random items from a large iterable without loading it all."""
    iterator = iter(iterable)
    result = []

    # Fill the reservoir with first k items
    for _ in range(k):
        result.append(next(iterator))

    # Replace elements with gradually decreasing probability
    n = k
    for item in iterator:
        n += 1
        r = random.randint(0, n - 1)
        if r < k:
            result[r] = item

    return result

# ---------------------------------------------------------------------------
# MAIN – load JSON, collect, plot
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load sentence dictionary
    with open("multitoken_words_contexts.json", "r", encoding="utf-8") as f:
        cut_sentences = json.load(f)

    # Pick the words we care about (all keys or first N)
    words = random_sample_iterable(cut_sentences, 10)

    # Collect curves for every word & revision
    curves = collect_all_curves(cut_sentences, words)
    plot_word_grids(curves, cut_sentences)


