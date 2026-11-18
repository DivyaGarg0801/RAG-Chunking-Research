
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RUN_ID = "20261115T113239215880Z"

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_DIR = (
    BASE_DIR / "experiments" / "results"
    / "embeddings" / RUN_ID
)

STRATEGIES = [
    "fixed_size",
    "recursive",
    "semantic",
    "structure",
    "logical_unit"
]

print("Loading model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

tokenizer = model.tokenizer
max_tokens = model.max_seq_length

print("Model maximum sequence length:", max_tokens)

print("\nTOKEN LENGTH AUDIT")
print("=" * 65)

for strategy in STRATEGIES:

    file_path = EMBEDDING_DIR / f"{strategy}_embeddings.json"

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    token_counts = []

    for chunk in data["chunks"]:
        tokens = tokenizer(
            chunk["text"],
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        token_counts.append(len(tokens))

    over_limit = sum(
        count > max_tokens for count in token_counts
    )

    print(f"\nStrategy: {strategy}")
    print("Total chunks:", len(token_counts))
    print("Average tokens:", round(sum(token_counts) / len(token_counts), 2))
    print("Maximum tokens:", max(token_counts))
    print("Chunks exceeding model limit:", over_limit)
