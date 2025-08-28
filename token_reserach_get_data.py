import json
from datasets import load_dataset
from transformers import AutoTokenizer
from token_research_utils import (
    extract_sentences_ending_at_multitoken_words,
    preprocess_json_sentences,
    add_replaced_last_word_sentences,
)

def main():
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:5%]")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")

    print("Extracting sentences ending at multi-token words...")
    cut_sentences = extract_sentences_ending_at_multitoken_words(dataset, tokenizer)

    print("Preprocessing sentences...")
    preprocessed_sentences = preprocess_json_sentences(cut_sentences, tokenizer)

    print("Adding replaced last word sentences...")
    final_data = add_replaced_last_word_sentences(preprocessed_sentences, tokenizer, max_matches=5)

    output_file = "multitoken_words_contexts.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"Final JSON saved to '{output_file}'.")

if __name__ == "__main__":
    main()
