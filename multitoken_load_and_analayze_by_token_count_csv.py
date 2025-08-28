import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_DIR = "C:/Users/liel.farber/Documents/Tranformers Project/multitoken_words_group_csvs"
OUTPUT_DIR = os.path.join(INPUT_DIR, "revision_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Revisions order must match experiment
revisions = [
    "step64", "step256", "step1000", "step3000", "step18000",
    "step48000", "step78000", "step108000", "step143000"
]


# ─────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────
def load_bucket_csv(path):
    """
    Load one bucket CSV into:
    word -> {revision -> list of (P1, P2) per layer}
    """
    data = defaultdict(lambda: defaultdict(list))
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"]
            rev = row["revision"]
            p1 = float(row["P1"])
            p2 = float(row["P2"])
            data[word][rev].append((p1, p2))
    return data


def compute_bucket_curve_for_revision(data, revision):
    """
    For one bucket and one revision:
      returns mean curve across layers (mean(P1 - P2) across words)
    """
    # find max number of layers
    max_layers = max((len(rev_dict[revision]) for rev_dict in data.values() if revision in rev_dict), default=0)
    if max_layers == 0:
        return []

    layer_means = []
    for layer_idx in range(max_layers):
        vals = []
        for word, rev_dict in data.items():
            if revision not in rev_dict:
                continue
            pairs = rev_dict[revision]
            if layer_idx < len(pairs):  # safeguard
                p1, p2 = pairs[layer_idx]
                vals.append(p1 - p2)
        if vals:
            layer_means.append(np.mean(vals))
        else:
            layer_means.append(np.nan)
    return layer_means


def plot_per_revision_curves(input_dir, output_dir):
    bucket_files = [f for f in os.listdir(input_dir) if f.endswith("_tokens.csv")]
    bucket_files.sort(key=lambda x: int(x.split("_")[0]))  # sort by token count

    # For each revision, make one plot
    for rev in revisions:
        plt.figure(figsize=(10, 6))
        for file in bucket_files:
            tok_count = int(file.split("_")[0])
            path = os.path.join(input_dir, file)

            data = load_bucket_csv(path)
            curve = compute_bucket_curve_for_revision(data, rev)
            if curve:
                plt.plot(range(len(curve)), curve, marker="o", label=f"{tok_count}-token")

        plt.title(f"Mean (P1 - P2) Curves — {rev}")
        plt.xlabel("Layer")
        plt.ylabel("Mean (P1 - P2)")
        plt.legend(title="Buckets")
        plt.grid(True)
        plt.tight_layout()

        output_plot = os.path.join(output_dir, f"{rev}_bucket_curves.png")
        plt.savefig(output_plot)
        plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_per_revision_curves(INPUT_DIR, OUTPUT_DIR)
