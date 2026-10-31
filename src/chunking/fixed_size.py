"""Character-based fixed-size text chunking."""


def fixed_size_chunk(
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[str]:
    """Split text into consecutive character-based chunks.

    Args:
        text: Already-extracted and preprocessed text.
        chunk_size: Maximum number of characters in each chunk.
        overlap: Number of characters shared by adjacent chunks.

    Raises:
        ValueError: If chunk_size is not positive or overlap is out of range.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and less than chunk_size.")
    if not text:
        return []

    step_size = chunk_size - overlap
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break
        start += step_size

    return chunks
