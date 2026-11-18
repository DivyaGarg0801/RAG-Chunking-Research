
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# 1. Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RUN_ID = "20261115T113239215880Z"

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "experiments"
    / "results"
    / "chunk_statistics"
    / RUN_ID
)

OUTPUT_DIR = (
    BASE_DIR
    / "experiments"
    / "results"
    / "embeddings"
    / RUN_ID
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load embedding model
print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Model loaded successfully!")


# 3. Load chunk data
input_file = INPUT_DIR / "fixed_size_chunks.json"

with open(input_file, "r", encoding="utf-8") as file:
    chunk_data = json.load(file)

chunks = chunk_data["chunks"]

print("Total chunks:", len(chunks))


# 4. Extract chunk texts
texts = [chunk["text"] for chunk in chunks]


# 5. Generate embeddings
print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embedding generation completed!")


# 6. Combine chunk information and embeddings
embedded_chunks = []

for chunk, embedding in zip(chunks, embeddings):

    embedded_chunks.append({
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
        "embedding": embedding.tolist()
    })


# 7. Save results
output_data = {
    "strategy": chunk_data["strategy"],
    "model_name": MODEL_NAME,
    "embedding_dimensions": model.get_sentence_embedding_dimension(),
    "total_chunks": len(embedded_chunks),
    "chunks": embedded_chunks
}

output_file = OUTPUT_DIR / "fixed_size_embeddings.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(output_data,
              file,
              ensure_ascii=False,
              indent=4
    )

print("\nEmbedding results saved!")
print("Output:", output_file)
print("Embedding shape:", embeddings.shape)
