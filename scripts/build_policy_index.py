
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "corpus" / "policy_pages.jsonl"
INDEX_DIR = PROJECT_ROOT / "corpus" / "policy_index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def load_page_records():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run scripts/extract_policy.py first."
        )

    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if record.get("text", "").strip():
                records.append(record)

    if not records:
        raise ValueError("No extractable policy text was found.")

    return records


def build_chunks(page_records):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []

    for page_record in page_records:
        page_text = page_record["text"].strip()
        text_parts = splitter.split_text(page_text)

        for part_number, part_text in enumerate(text_parts, start=1):
            if not part_text.strip():
                continue

            chunks.append(
                {
                    "chunk_id": f"page-{page_record['page_number']}-part-{part_number}",
                    "document_id": page_record["document_id"],
                    "document_title": page_record["document_title"],
                    "source_file": page_record["source_file"],
                    "page_number": page_record["page_number"],
                    "text": part_text.strip(),
                }
            )

    if not chunks:
        raise ValueError("Text splitting produced no chunks.")

    return chunks


def main():
    try:
        print("Loading extracted policy pages...")
        page_records = load_page_records()

        print(f"Loaded {len(page_records)} pages containing text.")
        print("Splitting policy text into chunks...")

        chunks = build_chunks(page_records)
        print(f"Created {len(chunks)} chunks.")

        print(f"Loading embedding model: {MODEL_NAME}")
        print("The first run may download the model and take some time.")

        model = SentenceTransformer(MODEL_NAME)

        texts = [chunk["text"] for chunk in chunks]

        print("Generating embeddings...")
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        print("Building FAISS search index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        index_path = INDEX_DIR / "policy.faiss"
        chunks_path = INDEX_DIR / "chunks.json"
        config_path = INDEX_DIR / "index_config.json"

        faiss.write_index(index, str(index_path))

        with chunks_path.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, ensure_ascii=False, indent=2)

        config = {
            "embedding_model": MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "similarity": "inner_product_on_normalized_embeddings",
            "source_document": "policy_wording.pdf",
            "number_of_pages_with_text": len(page_records),
            "number_of_chunks": len(chunks),
            "embedding_dimension": int(embeddings.shape[1]),
        }

        with config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        print("\nPolicy indexing completed successfully.")
        print(f"FAISS index: {index_path}")
        print(f"Chunk metadata: {chunks_path}")
        print(f"Index configuration: {config_path}")
        print(f"Total indexed chunks: {index.ntotal}")
        print("Original PDF page numbers are preserved in chunk metadata.")

    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
