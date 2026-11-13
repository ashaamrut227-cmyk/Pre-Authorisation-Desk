
import re
from pathlib import Path
from collections import Counter

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"

TEST_QUESTIONS = [
    "What waiting period applies to cataract treatment?",
    "What waiting period applies to a pre-existing disability?",
    "What co-payment applies under the policy?",
    "What is the exact room-rent limit?",
    "What is the cataract treatment sub-limit?",
]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be",
    "to", "of", "and", "or", "in", "on", "at", "for",
    "from", "by", "with", "as", "it", "this", "that",
    "what", "which", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "should", "would",
    "will", "may", "might", "under", "applies", "apply",
    "policy", "treatment"
}

# Add related terms to help questions find policy clauses.
SYNONYMS = {
    "cataract": {"cataract", "eye", "eyes"},
    "waiting": {"waiting", "period", "months", "expiry"},
    "disability": {"disability", "disabled", "pre-existing"},
    "copayment": {"copayment", "co-payment", "percentage", "admissible"},
    "room": {"room", "rent", "boarding", "nursing"},
    "limit": {"limit", "sub-limit", "sublimit", "sum", "insured"},
}


def tokens(text):
    words = re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def expand_query(question):
    original = set(tokens(question))
    expanded = set(original)

    for key, related_terms in SYNONYMS.items():
        if key in original:
            expanded.update(related_terms)

    # Detect common compound terms.
    lowered = question.lower()

    if "co-payment" in lowered or "copayment" in lowered:
        expanded.update({"copayment", "percentage", "admissible", "claim"})

    if "room-rent" in lowered or "room rent" in lowered:
        expanded.update({"room", "rent", "boarding", "nursing", "daily"})

    if "waiting period" in lowered:
        expanded.update({"waiting", "period", "expiry", "months"})

    return expanded


def load_chunks():
    if not CORPUS.exists():
        raise RuntimeError(f"Corpus folder not found: {CORPUS}")

    pdf_files = sorted(CORPUS.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {CORPUS}")

    chunks = []
    seen_text = set()

    for pdf_path in pdf_files:
        # Exclude unrelated PDFs from the health-policy retrieval test.
        name = pdf_path.name.lower()

        if "equicover" not in name and "policy_wording" not in name:
            continue

        print(f"Reading: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))

        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            words = text.split()

            # Smaller overlapping chunks preserve clause-level detail.
            chunk_size = 110
            overlap = 25
            start = 0

            while start < len(words):
                chunk_text = " ".join(words[start:start + chunk_size]).strip()

                if chunk_text:
                    normalized = re.sub(r"\s+", " ", chunk_text.lower())

                    # Remove identical chunks from duplicate policy PDFs.
                    if normalized not in seen_text:
                        seen_text.add(normalized)
                        chunks.append({
                            "source": pdf_path.name,
                            "page": page_no,
                            "text": chunk_text
                        })

                if start + chunk_size >= len(words):
                    break

                start += chunk_size - overlap

    if not chunks:
        raise RuntimeError(
            "No policy text extracted. Check the PDF names and their contents."
        )

    return chunks


def score_chunk(question, chunk):
    query = expand_query(question)
    chunk_tokens = tokens(chunk["text"])
    counts = Counter(chunk_tokens)
    unique = set(chunk_tokens)

    # Reward matching important terms, but avoid repeatedly counting a word.
    overlap = query & unique
    score = sum(1.0 + min(counts[word] - 1, 2) * 0.15 for word in overlap)

    # Give a modest bonus to passages containing important question concepts.
    text = chunk["text"].lower()
    question_lower = question.lower()

    if "cataract" in question_lower and "cataract" in text:
        score += 3

    if "waiting period" in question_lower:
        if "waiting period" in text or "waiting periods" in text:
            score += 3
        if re.search(r"\b\d+\s*months?\b", text):
            score += 1.5

    if "disability" in question_lower and "disability" in text:
        score += 2

    if "co-payment" in question_lower or "copayment" in question_lower:
        if "co-payment" in text or "copayment" in text:
            score += 3
        if "20%" in text:
            score += 1.5

    if "room-rent" in question_lower or "room rent" in question_lower:
        if "room rent" in text or "room/boarding" in text:
            score += 3
        if "1%" in text:
            score += 1.5

    if "exact" in question_lower and "limit" in question_lower:
        if "policy schedule" in text or "sub-limit" in text:
            score += 1

    return round(score, 3), sorted(overlap)


def search(question, chunks, top_k=5):
    results = []

    for chunk in chunks:
        score, matched_terms = score_chunk(question, chunk)

        if score > 0:
            results.append({
                **chunk,
                "score": score,
                "matched_terms": matched_terms
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


def main():
    chunks = load_chunks()
    print(f"\nUnique policy chunks loaded: {len(chunks)}")

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 78)
        print(f"QUESTION: {question}")
        print("=" * 78)

        results = search(question, chunks)

        if not results:
            print("No matching evidence found.")
            continue

        for rank, item in enumerate(results, start=1):
            print(
                f"\n[{rank}] Score: {item['score']} | "
                f"Source: {item['source']} | PDF page: {item['page']}"
            )
            print(f"Matched terms: {', '.join(item['matched_terms'])}")
            print(item["text"][:900])

    print("\nRetrieval test completed.")
    print(
        "\nImportant: these scores rank passages; they do not prove "
        "that a passage answers the question correctly."
    )


if __name__ == "__main__":
    main()
