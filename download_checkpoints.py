from transformers import GPTNeoXForCausalLM, AutoTokenizer
import os

# Choosen training steps to analyze
revisions = [
    "step64", "step256", "step1000", "step3000", "step18000", "step33000", "step48000", "step63000",
    "step78000", "step93000", "step108000", "step123000", "step143000"
]

model_id = "EleutherAI/pythia-1b-deduped"
base_dir = os.path.join(os.getcwd(), "pythia_1b_checkpoints")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Downloading model pretrained checkpoints
for rev in revisions:
    print(f"\n>>> Downloading revision: {rev}")
    model_dir = os.path.join(base_dir, rev)
    os.makedirs(model_dir, exist_ok=True)

    model = GPTNeoXForCausalLM.from_pretrained(
        model_id,
        revision=rev,
        cache_dir=model_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=rev,
        cache_dir=model_dir
    )

    # Explicitly save model + tokenizer to disk
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    del model, tokenizer