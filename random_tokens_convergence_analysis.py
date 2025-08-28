import os
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from token_convergence_analysis import bucket_words_by_token_count, get_hidden_states, compute_similarity_trajectories, plot_similarity_by_layer, All_layers, revisions
import numpy as np

# ──────────────────────────────────────────────────────────────
# Garbage word generation
# ──────────────────────────────────────────────────────────────
def generate_and_save_garbage_words(
    tokenizer_path, 
    output_file="garbage_words_list.json", 
    n_per_group=5000, 
    max_tokens=5, 
    seed=42
):
    """
    Generate garbage words of exact token lengths [1..max_tokens],
    save them to JSON.
    """
    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    tok.pad_token = tok.eos_token

    vocab = list(tok.get_vocab().keys())
    special_ids = set(tok.all_special_ids)
    valid_tokens = [t for t in vocab if tok.convert_tokens_to_ids(t) not in special_ids]

    garbage_dict = {}
    for tok_count in range(1, max_tokens + 1):
        words = []
        attempts = 0
        while len(words) < n_per_group and attempts < n_per_group * 100:
            attempts += 1
            chosen = random.choices(valid_tokens, k=tok_count)
            candidate = "".join(chosen)
            if len(tok.tokenize(candidate)) == tok_count:
                words.append(candidate)
        garbage_dict[tok_count] = words
        print(f"Generated {len(words)} valid garbage words for {tok_count}-token group")

    # Flatten into a simple list of words
    all_words = [w for group in garbage_dict.values() for w in group]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_words)} garbage words to {output_file}")

def run_token_count_experiment(all_words, results_dir_name):
    # Use tokenizer from first checkpoint to tokenize all words
    tok0 = AutoTokenizer.from_pretrained(os.path.join(base_dir, revisions[0]))
    tok0.pad_token = tok0.eos_token

    # Bucket words by token count
    bucketed = bucket_words_by_token_count(all_words, tok0, verbose=True)

    # For each token count group: extract hidden states & compute similarities
    all_sim_dicts = dict()
    for token_count, words in tqdm(sorted(bucketed.items()), desc="Processing token count groups"):
        hs = get_hidden_states(words, All_layers)  # hs: word -> {layer -> [reps per checkpoint]}

        if not hs:
            print(f"Skipping token count {token_count} group (empty hidden states)")
            continue

        # Expected number of checkpoints from first word
        expected_len = len(next(iter(hs.values())))

        # Filter out words with inconsistent number of checkpoints
        hs_filtered = {w: traj for w, traj in hs.items() if len(traj) == expected_len}

        # Now enforce same shape across *all words and layers*
        uniform_words = {}
        expected_shapes = None
        for w, traj in hs_filtered.items():
            # traj: dict(layer -> list of reps)
            shapes = {l: tuple(np.shape(rep) for rep in reps) for l, reps in traj.items()}
            if expected_shapes is None:
                expected_shapes = shapes
                uniform_words[w] = traj
            elif shapes == expected_shapes:
                uniform_words[w] = traj

        if len(uniform_words) < 2:
            print(f"Skipping token count {token_count} group (not enough uniform words)")
            continue

        sim = compute_similarity_trajectories(uniform_words)
        all_sim_dicts[token_count] = sim

    plot_similarity_by_layer(all_sim_dicts, results_dir_name)



# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = "C:/Users/liel.farber/Documents/Tranformers Project/pythia_1b_checkpoints"
    first_revision = "step64"
    tokenizer_path = os.path.join(base_dir, first_revision)

    # Generate words
    generate_and_save_garbage_words(
        tokenizer_path,
        output_file="garbage_words_list.json",
        n_per_group=200,
        max_tokens=5
    )

    # Run existing analysis
    with open("garbage_words_list.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    words = list(data)
    run_token_count_experiment(words, "random_tokens_convergence_plots")
