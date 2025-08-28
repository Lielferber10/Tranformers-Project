import os
import json
import random
import csv
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
base_dir = "C:/Users/liel.farber/Documents/Tranformers Project/pythia_1b_checkpoints"
revisions = [
    "step64", "step256", "step1000", "step3000", "step18000",
    "step48000", "step78000", "step108000", "step143000"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_PER_BUCKET = 20     # at most n words per bucket
MIN_BUCKET_SIZE = 20    # discard buckets smaller than this
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "multitoken_words_group_csvs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def get_last_token_hs(sentence: str, *, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    return [layer[0, -1, :] for layer in outputs.hidden_states]


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()


def avg_similarity(base_hs, comparison_sentences, *, model, tokenizer):
    sims_per_layer = [[] for _ in range(len(base_hs))]
    for sent in comparison_sentences:
        hs = get_last_token_hs(sent, model=model, tokenizer=tokenizer)
        for i, (ref, cur) in enumerate(zip(base_hs, hs)):
            sims_per_layer[i].append(cosine_sim(ref, cur))
    return [sum(layer) / len(layer) for layer in sims_per_layer]


def run_experiment(base, contexts_variants, word_variants, model, tokenizer):
    base_hs = get_last_token_hs(base, model=model, tokenizer=tokenizer)
    P1 = avg_similarity(base_hs, contexts_variants, model=model, tokenizer=tokenizer)
    P2 = avg_similarity(base_hs, word_variants, model=model, tokenizer=tokenizer)
    return P1, P2


def bucket_words_by_token_count(words, tokenizer, verbose=False, min_size=100, max_per_bucket=None):
    token_buckets = defaultdict(list)
    for w in words:
        tok_count = len(tokenizer.tokenize(w))
        token_buckets[tok_count].append(w)

    token_buckets = {k: v for k, v in token_buckets.items() if len(v) >= min_size}

    if max_per_bucket is not None:
        for k in token_buckets:
            if len(token_buckets[k]) > max_per_bucket:
                token_buckets[k] = random.sample(token_buckets[k], max_per_bucket)

    if verbose:
        print("Token count distribution after sampling:")
        for k in sorted(token_buckets):
            print(f"  {k}-token words: {len(token_buckets[k])}")

    return dict(token_buckets)


def integral_difference(P2):
    """Compute Δ = (#layers - sum(P2))"""
    return len(P2) - sum(P2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("multitoken_words_contexts.json", "r", encoding="utf-8") as f:
        cut_sentences = json.load(f)

    all_words = list(cut_sentences.keys())

    # Use tokenizer from first checkpoint only for bucketing
    first_model, first_tok = load_model_and_tokenizer(os.path.join(base_dir, revisions[0]))
    token_buckets = bucket_words_by_token_count(
        all_words,
        first_tok,
        verbose=True,
        min_size=MIN_BUCKET_SIZE,
        max_per_bucket=MAX_PER_BUCKET
    )
    del first_model, first_tok
    torch.cuda.empty_cache()

    # Prepare per-bucket CSV writers
    bucket_writers = {}
    bucket_files = {}

    for tok_count in token_buckets:
        csv_path = os.path.join(OUTPUT_DIR, f"{tok_count}_tokens.csv")
        f_csv = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(f_csv)
        writer.writerow(["word", "revision", "layer", "P1", "P2"])
        bucket_writers[tok_count] = writer
        bucket_files[tok_count] = f_csv

    # Main loop: iterate revisions, load model once, then process all buckets
    for rev in tqdm(revisions, desc="Revisions", leave=True):
        model_path = os.path.join(base_dir, rev)
        model, tok = load_model_and_tokenizer(model_path)

        for tok_count, bucket_words in token_buckets.items():
            writer = bucket_writers[tok_count]
            for word in tqdm(bucket_words, desc=f"{tok_count}-token words in {rev}", leave=False):
                base = cut_sentences[word]["sentences"][0]
                context_variants = cut_sentences[word]["sentences"][1:]
                word_variants = cut_sentences[word]["replaced_sentences"]

                P1, P2 = run_experiment(base, context_variants, word_variants, model, tok)

                for layer_idx, (p1_val, p2_val) in enumerate(zip(P1, P2)):
                    writer.writerow([word, rev, layer_idx, p1_val, p2_val])

        del model, tok
        torch.cuda.empty_cache()

    # Close CSVs
    for _, f_csv in bucket_files.items():
        f_csv.close()
