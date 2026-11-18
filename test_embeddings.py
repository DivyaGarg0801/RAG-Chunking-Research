
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Model loaded successfully!")
print("Embedding dimensions:", model.get_sentence_embedding_dimension())



sample_text = "Retrieval Augmented Generation combines retrieval with language models."

embedding = model.encode(sample_text)
assert embedding.shape == (384,), "Unexpected embedding dimensions!"

print("\nEmbedding generated!")
print("Shape:", embedding.shape)
print("First 10 values:", embedding[:10])
