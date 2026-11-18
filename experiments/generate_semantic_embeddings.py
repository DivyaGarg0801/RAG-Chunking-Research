
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RUN_ID = "20261115T113239215880Z"

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR / "experiments" / "results"
    / "chunk_statistics" / RUN_ID
)

OUTPUT_DIR = (
    BASE_DIR / "experiments" / "results"
    / "embeddings" / RUN_ID
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")
print("Model loaded successfully!")

# Load semantic chunks
input_file = INPUT_DIR / "semantic_chunks.json"

with open(input_file, "r", encoding="utf-8") as file:
    chunk_data = json.load(file)

chunks = chunk_data["chunks"]

print("Strategy:", chunk_data["strategy"])
print("Total chunks:", len(chunks))

# Extract chunk text
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embedding generation completed!")

# Preserve metadata and attach embeddings
embedded_chunks = []

for chunk, embedding in zip(chunks, embeddings):
    embedded_chunk = {
        **chunk,
        "embedding": embedding.tolist()
    }
    embedded_chunks.append(embedded_chunk)

output_data = {
    "strategy": chunk_data["strategy"],
    "model_name": MODEL_NAME,
    "embedding_dimensions": model.get_sentence_embedding_dimension(),
    "total_chunks": len(embedded_chunks),
    "chunks": embedded_chunks
}

# Save output
output_file = OUTPUT_DIR / "semantic_embeddings.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(
        output_data,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\nEmbedding results saved!")
print("Output:", output_file)
print("Embedding matrix shape:", embeddings.shape)
