# Token Representation Evolution Analysis

This repository contains code, data, and results for analyzing how token representations evolve during training in large language models, with a focus on the **Pythia-1B** model family across multiple training checkpoints.

## Project Overview

The project consists of **two main experimental parts**:

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
```## Repository Structure
├── .venv/ # Local Python virtual environment
├── pycache/ # Auto-generated Python bytecode cache
├── pythia_1b_checkpoints/ # Downloaded Pythia-1B model checkpoints
├── download_checkpoints.py # Script to download Pythia-1B checkpoints
│
├── token_convergence_analysis.py # (part 1 - experiment 1)
├── convergence_plots/ # (part 1 - experiment 1)
├── random_tokens_convergence_analysis.py # (part 1 - experiment 2)
├── random_tokens_convergence_plots/ # (part 1 - experiment 2)
├── 10_tokens_sequences.py # (part 1 - experiment 3)
├── 10_tokens_sequences_convergence_plots/ # (part 1 - experiment 3)
├── layer0_token_lengths_analysis.py # (part 1 - experiment 4.1)
├── layer0_token_length_convergence_plots/ # (part 1 - experiment 4.1)
├── layer0_realword_analysis.py # (part 1 - experiment 4.2)
├── layer0_last_token_realword_convergence/ # (part 1 - experiment 4.2)
├── layer0_freq_analysis.py # (part 1 - experiment 4.3)
├── layer0_token_freq_convergence/ # (part 1 - experiment 4.3)
│
├── multitoken_context_perturbation.py # (part 2 - experiment 1)
├── multitoken_words_plots/ # (part 2 - experiment 1)
├── multitoken_create_csv_files.py # (part 2 - experiment 2)
├── multitoken_load_and_analyze_by_token_count_csv.py # (part 2 - experiment 2)
├── multitoken_words_group_csvs/ # (part 2 - experiment 2)
│ └── revision_plots/ # (part 2 - experiment 2)
│
├── last_subword_length_histogram.py # Script to analyze last-subword token length distributions
├── last_subword_length_histograms/ # Histograms of last-subword token lengths
│
├── token_research_get_data.py # Script to generate multitoken_words_contexts.json
├── token_research_utils.py # Utility functions used in token_research_get_data.py
│
├── words_list.json # Word list for token convergence experiments
├── garbage_words_list.json # Generated random_tokens in part 1, experiment 2 
├── multitoken_words_contexts.json # Multi-token words + contexts (from wikitext dataset)
└── README.md # Project documentation
```
