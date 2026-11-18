
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

# Load recursive chunks
input_file = INPUT_DIR / "recursive_chunks.json"

with open(input_file, "r", encoding="utf-8") as file:
    chunk_data = json.load(file)

chunks = chunk_data["chunks"]

print("Strategy:", chunk_data["strategy"])
print("Total chunks:", len(chunks))

# Extract text from every chunk
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embedding generation completed!")

# Combine chunk information with embeddings
embedded_chunks = []

for index, (chunk, embedding) in enumerate(
    zip(chunks, embeddings), start=1
):
    embedded_chunks.append({
        "chunk_index": index,
        "text": chunk["text"],
        "overlap_length": chunk["overlap_length"],
        "embedding": embedding.tolist()
    })

# Prepare output
output_data = {
    "strategy": chunk_data["strategy"],
    "model_name": MODEL_NAME,
    "embedding_dimensions": model.get_sentence_embedding_dimension(),
    "total_chunks": len(embedded_chunks),
    "chunks": embedded_chunks
}

# Save output
output_file = OUTPUT_DIR / "recursive_embeddings.json"

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
