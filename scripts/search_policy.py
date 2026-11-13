
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "corpus" / "policy_index"

INDEX_PATH = INDEX_DIR / "policy.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
CONFIG_PATH = INDEX_DIR / "index_config.json"


def load_retriever():
    required_files = [INDEX_PATH, CHUNKS_PATH, CONFIG_PATH]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required index file not found: {path}\n"
                "Run scripts/build_policy_index.py first."
            )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    index = faiss.read_index(str(INDEX_PATH))

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index contains {index.ntotal} vectors but "
            f"metadata contains {len(chunks)} chunks."
        )

    model = SentenceTransformer(config["embedding_model"])

    return index, chunks, model


def search_policy(query, index, chunks, model, top_k=5):
    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    result_count = min(top_k, index.ntotal)
    scores, positions = index.search(query_embedding, result_count)

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        chunk = chunks[int(position)]

        results.append(
            {
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "document_title": chunk["document_title"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            }
        )

    return results


def main():
    try:
        index, chunks, model = load_retriever()

        print("Policy retrieval is ready.")
        print("Enter a question about the policy.")
        print("Type 'exit' to quit.\n")

        while True:
            query = input("Policy question: ").strip()

            if query.lower() in {"exit", "quit"}:
                print("Search session ended.")
                break

            if not query:
                print("Please enter a question.\n")
                continue

            results = search_policy(
                query=query,
                index=index,
                chunks=chunks,
                model=model,
                top_k=5,
            )

            print("\nRetrieved policy passages:")

            for rank, result in enumerate(results, start=1):
                print("\n" + "=" * 75)
                print(f"Result: {rank}")
                print(f"Similarity score: {result['score']:.4f}")
                print(f"Source: {result['document_title']}")
                print(f"PDF page: {result['page_number']}")
                print(f"Chunk ID: {result['chunk_id']}")
                print("-" * 75)
                print(result["text"])

            print("\n" + "=" * 75 + "\n")

    except Exception as error:
        print(f"ERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
