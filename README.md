# Token Representation Evolution Analysis

This repository contains code, data, and results for analyzing how token representations evolve during training in large language models, with a focus on the **Pythia-1B** model family across multiple training checkpoints.

## Project Overview

The project consists of **two main experimental tracks**:

---

### 1. Convergence Analysis of Token Groups

**Experiment 1**  
- **Script:** `token_convergence_analysis.py`  
- **Word list source:** `words_list.json` (derived from [`word_freq_32k.json`](https://github.com/BilboBlockins/word-frequency-list-json/tree/master))  
- **Results folder:** `convergence_plots/`

**Experiment 2**  
- **Script:** `random_tokens_convergence_analysis.py`  
- **Results folder:** `random_tokens_convergence_plots/`

**Experiment 3**  
- **Script:** `10_tokens_sequences.py`  
- **Results folder:** `10_tokens_sequences_convergence_plots/`

**Experiment 4.1**  
- **Script:** `layer0_token_lengths_analysis.py`  
- **Results folder:** `layer0_token_length_convergence_plots/`

**Experiment 4.2**  
- **Script:** `layer0_realword_analysis.py`  
- **Results folder:** `layer0_last_token_realword_convergence/`

**Experiment 4.3**  
- **Script:** `layer0_freq_analysis.py`  
- **Results folder:** `layer0_token_freq_convergence/`

---

### 2. Analysis of Multi-Token Representations

**Experiment 1**  
- **Script:** `multitoken_context_perturbation.py`  
- **Word/context source:** `multitoken_words_contexts.json` (from [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext))  
- **Results folder:** `multitoken_words_plots/`

**Experiment 2**  
- **Scripts:**  
  - `multitoken_create_csv_files.py`  
  - `multitoken_load_and_analyze_by_token_count_csv.py`  
- **Word/context source:** `multitoken_words_contexts.json` (from [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext))  
- **Results folders:**  
  - `multitoken_words_group_csvs/` (CSV files)  
  - `multitoken_words_group_csvs/revision_plots/` (plots)  

---
