
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF


# Resolve paths from the project root, not from the terminal's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "corpus" / "policy_wording.pdf"
JSONL_PATH = PROJECT_ROOT / "corpus" / "policy_pages.jsonl"
TEXT_PATH = PROJECT_ROOT / "corpus" / "policy_extracted.txt"


def extract_policy_pages():
    """Extract policy text while preserving the original PDF page numbers."""

    if not PDF_PATH.exists():
        print(f"ERROR: Policy PDF not found:\n{PDF_PATH}")
        print(
            "Place the policy PDF in the corpus folder "
            "and name it policy_wording.pdf."
        )
        sys.exit(1)

    page_records = []
    text_sections = []
    empty_pages = []

    try:
        with fitz.open(PDF_PATH) as document:
            total_pages = len(document)

            for page_index, page in enumerate(document):
                # PDF page numbers are one-based for human-readable citations.
                page_number = page_index + 1
                page_text = page.get_text("text").strip()

                if not page_text:
                    empty_pages.append(page_number)

                record = {
                    "document_id": "equicover_health_policy",
                    "document_title": "HDFC ERGO EquiCover Health Policy Wording",
                    "source_file": PDF_PATH.name,
                    "page_number": page_number,
                    "text": page_text,
                }
                page_records.append(record)

                text_sections.append(
                    f"===== PAGE {page_number} =====\n{page_text}\n"
                )

    except Exception as error:
        print(f"ERROR: Could not read the PDF: {error}")
        sys.exit(1)

    # Store one JSON object per page for downstream retrieval and citation.
    with JSONL_PATH.open("w", encoding="utf-8") as output_file:
        for record in page_records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Also save a readable text version for manual inspection.
    TEXT_PATH.write_text(
        "\n".join(text_sections),
        encoding="utf-8",
    )

    pages_with_text = sum(
        1 for record in page_records if record["text"]
    )

    print("PDF extraction completed successfully.")
    print(f"Input PDF: {PDF_PATH.name}")
    print(f"Total PDF pages: {total_pages}")
    print(f"Pages containing extractable text: {pages_with_text}")
    print(f"Pages with no extractable text: {empty_pages or 'None'}")
    print(f"Page-level JSONL output: {JSONL_PATH}")
    print(f"Readable text output: {TEXT_PATH}")

    if pages_with_text < total_pages:
        print(
            "WARNING: Some pages may contain scanned images or "
            "non-extractable text. Review those pages before retrieval."
        )


if __name__ == "__main__":
    extract_policy_pages()
