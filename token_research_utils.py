
from collections import defaultdict
import json
import re

def extract_sentences_ending_at_multitoken_words(dataset, tokenizer, min_word_len=9, max_words=100, max_sentences=5):
    """
    Goes through the dataset and returns sentences that end with a multi-token word.
    Returns a dict: word -> list of sentence fragments ending at that word.
    """
    word_to_cut_sentences = defaultdict(list)

    for sample in dataset:
        text = sample["text"]
        if not text or "." not in text:
            continue

        sentences = text.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 5:
                continue
                
            encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding["offset_mapping"]
            tokens = encoding["input_ids"]
            token_strs = tokenizer.convert_ids_to_tokens(tokens)

            for i, word in enumerate(sentence.split()):
                # Strip common punctuation from word
                clean_word = word.strip(".,!?;:\"'()[]{}")
                if len(clean_word) < min_word_len:
                    continue

                # Find character span of the word in the sentence
                word_start = sentence.find(clean_word)
                if word_start == -1:
                    continue
                word_end = word_start + len(clean_word)

                # Count how many tokens fall within the word span
                split_count = 0
                for offset in offsets:
                    token_start, token_end = offset
                    if token_start >= word_start and token_end <= word_end:
                        split_count += 1

                if split_count < 2:
                    continue

                # Reconstruct sentence up to and including this word
                # Since `find()` may return the first match only, be cautious with repeats
                cut_index = sentence.find(clean_word) + len(clean_word)
                cut_sentence = sentence[:cut_index].strip()

                word_to_cut_sentences[clean_word.lower()].append(cut_sentence.lower())

                if len(word_to_cut_sentences) >= max_words:
                    break

    return {
        word: sents[:max_sentences]
        for word, sents in word_to_cut_sentences.items()
        if len(sents) >= max_sentences
    }

def preprocess_json_sentences(json_data, tokenizer):
    """
    Cleans and preprocesses the extracted sentences:
    - Removes duplicate sentences (ignoring spaces and case).
    - Removes sentences identical to the word itself.
    - Adds info for each word about how many tokens it is divided into.
    Returns a dict: word -> {"sentences": [...], "token_split": int}
    """
    processed = {}
    seen_sentences = set()

    for word, sentences in json_data.items():
        clean_word = re.sub(r"\s+", "", word.lower())
        unique_sents = []
        for sent in sentences:
            clean_sent = re.sub(r"\s+", "", sent.lower())
            if clean_sent == clean_word:
                continue
            if clean_sent in seen_sentences:
                continue
            seen_sentences.add(clean_sent)
            unique_sents.append(sent)
        if not unique_sents or len(unique_sents) < 2:
            continue
        # Count tokens for the word
        encoding = tokenizer(word, add_special_tokens=False)
        token_count = len(encoding["input_ids"])

        if unique_sents:
            processed[word] = {
                "sentences": unique_sents,
                "token_split": token_count
            }

    return processed



from nltk.corpus import words as nltk_words
import nltk
from tqdm import tqdm

from collections import defaultdict

def build_suffix_index(vocab, tokenizer, max_suffix_len=10):
    suffix_map = defaultdict(list)
    for word in vocab:
        tokens = tokenizer.tokenize(word)
        if not tokens:
            continue
        suffix = tokens[-1]
        if len(suffix) <= max_suffix_len:
            suffix_map[suffix].append(word)
    return suffix_map

def replace_last_word_same_last_token(sentence, tokenizer, suffix_map, max_matches=5):
    sentence = sentence.strip()
    if not sentence:
        return []

    enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(enc.input_ids)
    offsets = enc.offset_mapping

    match = list(re.finditer(r"\b\w+\b", sentence))
    if not match:
        return []

    last_word_span = match[-1]
    word_start, word_end = last_word_span.span()
    last_token = tokens[-1] if tokens else None
    if not last_token or last_token not in suffix_map:
        return []
    prefix = sentence[:word_start]
    original_last_word = sentence[word_start:word_end].lower()

    candidates = suffix_map.get(last_token, [])
    results = []

    for word in candidates:
        if word == original_last_word:
            continue
        new_sentence = prefix + word
        new_tokens = tokenizer.tokenize(new_sentence)
        if new_tokens and new_tokens[-1] == last_token:
            results.append({"replacement": word, "text": new_sentence})
        if len(results) >= max_matches:
            break

    return results


def add_replaced_last_word_sentences(json_data, tokenizer, max_matches=5):
    """
    For each word in json_data, generates new sentences by replacing the last word
    in each sentence with other words that share the same last token.
    Adds these new sentences under a new key "replaced_sentences" for each word.
    """
    if not nltk_words.words():
        nltk.download("words", quiet=True)
    english_vocab = set(w.lower() for w in nltk_words.words())
    suffix_map = build_suffix_index(english_vocab, tokenizer)
    for word, data in tqdm(json_data.items(), desc="Replacing last words"):
        replaced_sentences = []
        replacements = replace_last_word_same_last_token(data["sentences"][0], tokenizer, suffix_map, max_matches)
        if not replacements:
            continue
        for rep in replacements:
            replaced_sentences.append(rep["text"])
        # Remove duplicates and sentences identical to the word itself
        replaced_sentences = list({s.lower() for s in replaced_sentences if s.lower() != word.lower()})
        if replaced_sentences:
            data["replaced_sentences"] = replaced_sentences
    return json_data
