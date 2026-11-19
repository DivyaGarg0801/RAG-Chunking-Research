
import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RUN_ID = "20261115T113239215880Z"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

WINDOW_SIZE = 250
OVERLAP = 30

BASE_DIR = Path(__file__).resolve().parents[1]

CHUNK_DIR = (
    BASE_DIR / "experiments" / "results"
    / "chunk_statistics" / RUN_ID
)

OUTPUT_DIR = (
    BASE_DIR / "experiments" / "results"
    / "window_embeddings" / RUN_ID
)

CHUNK_FILES = {
    "fixed_size": "fixed_size_chunks.json",
    "recursive": "recursive_chunks.json",
    "semantic": "semantic_chunks.json",
    "structure": "structure_chunks.json",
    "logical_unit": "logical_unit_chunks.json",
}


# --------------------------------------------------
# TOKEN WINDOW GENERATION
# --------------------------------------------------

def create_windows(text, tokenizer, max_length):
    """
    Split original token IDs into overlapping windows.

    No decoding and re-tokenization is used to generate
    the model inputs.
    """

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    if not token_ids:
        return []

    windows = []
    start = 0

    special_token_count = (
        tokenizer.num_special_tokens_to_add(pair=False)
    )

    while start < len(token_ids):

        end = min(start + WINDOW_SIZE, len(token_ids))

        window_ids = token_ids[start:end]

        # Construct the exact model input, including
        # special tokens such as [CLS] and [SEP].
        model_input_ids = (
            [tokenizer.cls_token_id]
            + window_ids
            + [tokenizer.sep_token_id]
        )

        if len(model_input_ids) > max_length:
            raise ValueError(
                f"Window exceeds model limit: "
                f"{len(model_input_ids)} > {max_length}"
            )

        # Human-readable representation for inspection.
        # Embeddings are NOT generated from this text.
        window_text = tokenizer.decode(
            window_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

        windows.append({
            "text": window_text,
            "token_ids": window_ids,
            "model_input_token_ids": model_input_ids,
            "window_start_token": start,
            "window_end_token": end,
            "window_token_count": len(window_ids),
            "model_input_token_count": len(model_input_ids),
        })

        if end == len(token_ids):
            break

        next_start = end - OVERLAP

        if next_start <= start:
            raise ValueError(
                "Window generation failed to make progress."
            )

        start = next_start

    return windows


# --------------------------------------------------
# GENERATE EMBEDDING DIRECTLY FROM TOKEN IDS
# --------------------------------------------------

def generate_embedding(model, tokenizer, window):
    """
    Generate a SentenceTransformer embedding directly
    from the original window token IDs.
    """

    model_input_ids = window["model_input_token_ids"]

    if len(model_input_ids) > model.max_seq_length:
        raise ValueError(
            f"Model input exceeds limit: "
            f"{len(model_input_ids)} > {model.max_seq_length}"
        )

    # Construct model inputs directly from exact token IDs.
    features = {
        "input_ids": torch.tensor(
            [model_input_ids],
            dtype=torch.long,
            device=model.device
        ),
        "attention_mask": torch.ones(
            (1, len(model_input_ids)),
            dtype=torch.long,
            device=model.device
        ),
        "token_type_ids": torch.zeros(
            (1, len(model_input_ids)),
            dtype=torch.long,
            device=model.device
        )
    }

    with torch.inference_mode():
        output = model(features)

    embedding = (
        output["sentence_embedding"][0]
        .detach()
        .cpu()
        .numpy()
    )

    return embedding

# --------------------------------------------------
# GENERATE WINDOW EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(strategy, model):

    chunk_file = CHUNK_DIR / CHUNK_FILES[strategy]

    with open(chunk_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = data["chunks"]

    tokenizer = model.tokenizer
    max_length = model.max_seq_length

    special_tokens = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    if WINDOW_SIZE + special_tokens > max_length:
        raise ValueError(
            "Window size exceeds the model's token limit."
        )

    output_file = OUTPUT_DIR / f"{strategy}_windows.jsonl"

    total_windows = 0

    print(f"\nStrategy: {strategy}")
    print(f"Original chunks: {len(chunks)}")

    model.eval()

    with open(output_file, "w", encoding="utf-8") as output:

        for position, chunk in enumerate(chunks, start=1):

            original_text = chunk["text"]

            windows = create_windows(
                original_text,
                tokenizer,
                max_length
            )

            if not windows:
                raise ValueError(
                    f"Chunk {position} produced no windows."
                )

            for window_index, window in enumerate(
                windows,
                start=1
            ):

                embedding = generate_embedding(
                    model,
                    tokenizer,
                    window
                )

                record = {
                    **{
                        key: value
                        for key, value in chunk.items()
                        if key != "embedding"
                    },

                    "parent_chunk_index": chunk.get(
                        "chunk_index", position
                    ),

                    "window_index": window_index,

                    "text": window["text"],

                    "window_start_token": (
                        window["window_start_token"]
                    ),

                    "window_end_token": (
                        window["window_end_token"]
                    ),

                    "window_token_count": (
                        window["window_token_count"]
                    ),

                    "model_input_token_count": (
                        window["model_input_token_count"]
                    ),

                    # Exact content token IDs.
                    "window_token_ids": window["token_ids"],

                    # Exact IDs passed into the model.
                    "model_input_token_ids": (
                        window["model_input_token_ids"]
                    ),

                    "embedding": embedding.tolist(),
                }

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":")
                    ) + "\n"
                )

                total_windows += 1

            if position % 25 == 0:
                print(
                    f"Processed {position}/{len(chunks)} chunks"
                )

    # Save experiment metadata separately.
    manifest = {
        "strategy": strategy,
        "model_name": MODEL_NAME,
        "embedding_dimensions": (
            model.get_sentence_embedding_dimension()
        ),
        "model_max_sequence_length": max_length,
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "original_chunks": len(chunks),
        "total_windows": total_windows,
        "embedding_method": "direct_token_id_forward_pass",
        "output_file": output_file.name,
    }

    manifest_file = OUTPUT_DIR / f"{strategy}_manifest.json"

    with open(manifest_file, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print(f"\nCompleted: {strategy}")
    print(f"Total windows: {total_windows}")
    print(f"Output: {output_file}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate token-window embeddings."
    )

    parser.add_argument(
        "strategy",
        choices=CHUNK_FILES.keys()
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")
    print("Maximum sequence length:", model.max_seq_length)

    generate_embeddings(args.strategy, model)


if __name__ == "__main__":
    main()
