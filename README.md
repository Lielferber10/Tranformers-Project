# Token Representation Evolution Analysis

This repository contains code and data for analyzing how token representations evolve during training in large language models, with a focus on the **Pythia-1B** model family across multiple training checkpoints.

## Project Overview

The project consists of **two main experiments**:

### 1. Token Convergence Analysis
We track token convergence across training revisions by comparing **layerwise cosine similarity** between intermediate hidden states of:
- Single-token words
- Multi-token words  
and their **final trained counterparts**.

The analysis:
- Averages results within each token group
- Produces convergence graphs for each token group
- Aggregates results into a single plot per layer
- Reveals how representational stability changes during training

**Script:** `token_convergence_analysis.py`  
**Word list source:** `words_list.json` (derived from [`word_freq_32k.json`](https://github.com/BilboBlockins/word-frequency-list-json/tree/master))

---

### 2. Multi-Token Context Perturbation
We analyze individual multi-token word representations across layers and training steps by comparing their base representations to two variants:
- **P1:** The same word in different contexts
- **P2:** Versions with the last word perturbed

This produces **layer-by-layer similarity profiles** for each training step, giving insight into the development of **compositional word representations**.

**Script:** `multitoken_context_perturbation.py`  
**Word/context source:** `multitoken_words_contexts.json` (obtained from _____)

---

## Repository Structure

