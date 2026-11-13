import json
import re
from pathlib import Path
from collections import Counter

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "evaluation" / "real_llmops_results.json"
OUTPUT_FILE = ROOT / "evaluation" / "retrieval_comparison.json"
CORPUS_DIR = ROOT / "corpus"

TOP_K = 3


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_questions(data):
    """Read questions from the actual scenario_results structure."""
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = None

        for key in (
            "scenario_results",
            "results",
            "questions",
            "scenarios",
            "evaluation_results",
            "items",
        ):
            value = data.get(key)
            if isinstance(value, list):
                records = value
                break

        if records is None:
            raise ValueError(
                "Could not find question records. "
                f"Available keys: {list(data.keys())}"
            )
    else:
        raise ValueError("The evaluation JSON must contain a list or object.")

    questions = []

    for record in records:
        if not isinstance(record, dict):
            continue

        question = (
            record.get("question")
            or record.get("query")
            or record.get("input")
            or record.get("user_query")
        )

        if question:
            questions.append(
                {
                    "question_id": record.get(
                        "question_id", record.get("id", f"Q{len(questions) + 1}")
                    ),
                    "question": str(question),
                    "original_evidence": record.get(
                        "retrieved_evidence", []
                    ),
                }
            )

    if not questions:
        preview = json.dumps(data, ensure_ascii=False, indent=2)[:2000]
        raise ValueError(
            "No policy questions found inside scenario_results. "
            "The records may use a different schema.\n"
            f"JSON preview:\n{preview}"
        )

    return questions


def load_policy_chunks():
    """Extract PDF text into page-level retrieval chunks."""
    pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF policy files found in: {CORPUS_DIR}"
        )

    chunks = []

    for pdf_path in pdf_files:
        print(f"Reading policy: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            print(f"  Skipping unreadable PDF: {exc}")
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            # Split long pages into smaller overlapping passages.
            words = text.split()
            window_size = 180
            overlap = 35

            if not words:
                continue

            start = 0
            while start < len(words):
                passage = " ".join(words[start:start + window_size])

                if passage.strip():
                    chunks.append(
                        {
                            "source": pdf_path.name,
                            "page": page_number,
                            "text": passage,
                        }
                    )

                if start + window_size >= len(words):
                    break

                start += window_size - overlap

    if not chunks:
        raise RuntimeError(
            "PDF files were found, but no extractable text was obtained."
        )

    print(f"Loaded {len(chunks)} policy passages.")
    return chunks


def baseline_score(question, passage):
    """Simple query-term overlap baseline."""
    query_terms = set(tokenize(question))
    passage_terms = set(tokenize(passage))

    if not query_terms or not passage_terms:
        return 0.0

    return len(query_terms & passage_terms) / len(query_terms)


def improved_score(question, passage):
    """Overlap score with term-frequency and phrase-match bonuses."""
    query_tokens = tokenize(question)
    passage_tokens = tokenize(passage)

    if not query_tokens or not passage_tokens:
        return 0.0

    query_terms = set(query_tokens)
    passage_counts = Counter(passage_tokens)
    passage_terms = set(passage_tokens)

    overlap = len(query_terms & passage_terms) / len(query_terms)

    # Reward query terms that appear in the passage without allowing
    # repeated terms to dominate the ranking.
    coverage = sum(
        1 for term in query_terms if passage_counts.get(term, 0) > 0
    ) / len(query_terms)

    # Give a small bonus for exact multiword phrases from the question.
    normalized_question = " ".join(query_tokens)
    normalized_passage = " ".join(passage_tokens)
    phrase_bonus = 0.0

    for phrase_length in (2, 3):
        for index in range(len(query_tokens) - phrase_length + 1):
            phrase = " ".join(
                query_tokens[index:index + phrase_length]
            )
            if phrase in normalized_passage:
                phrase_bonus += 0.02

    phrase_bonus = min(phrase_bonus, 0.10)

    # Small bonus for policy-specific terms when present in both.
    policy_terms = {
        "waiting", "period", "preexisting", "disease", "cataract",
        "room", "rent", "limit", "coverage", "excluded", "exclusion",
        "copayment", "co", "payment", "sum", "insured", "claim",
        "policy", "hospitalization", "maternity", "icu",
    }
    shared_policy_terms = query_terms & passage_terms & policy_terms
    policy_bonus = min(0.03 * len(shared_policy_terms), 0.12)

    return min(coverage + phrase_bonus + policy_bonus, 1.0)


def rank_chunks(question, chunks, scorer):
    ranked = []

    for chunk in chunks:
        score = scorer(question, chunk["text"])
        ranked.append(
            {
                **chunk,
                "score": round(score, 6),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:TOP_K]


def evidence_summary(evidence):
    """Normalize existing evidence records for output."""
    if not isinstance(evidence, list):
        return []

    normalized = []

    for item in evidence:
        if not isinstance(item, dict):
            continue

        normalized.append(
            {
                "source": item.get("source"),
                "page": item.get("page"),
                "text": str(item.get("text", ""))[:500],
                "retrieval_score": item.get("retrieval_score"),
            }
        )

    return normalized


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input evaluation file not found: {INPUT_FILE}"
        )

    data = load_json(INPUT_FILE)
    questions = load_questions(data)
    chunks = load_policy_chunks()

    results = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]

        baseline = rank_chunks(question, chunks, baseline_score)
        improved = rank_chunks(question, chunks, improved_score)

        baseline_keys = {
            (entry["source"], entry["page"], entry["text"])
            for entry in baseline
        }
        improved_keys = {
            (entry["source"], entry["page"], entry["text"])
            for entry in improved
        }

        common = baseline_keys & improved_keys
        union = baseline_keys | improved_keys
        top_k_overlap = len(common) / len(union) if union else 1.0

        results.append(
            {
                "question_id": item["question_id"],
                "question": question,
                "original_retrieved_evidence": evidence_summary(
                    item["original_evidence"]
                ),
                "baseline_top_k": baseline,
                "improved_top_k": improved,
                "top_k_overlap": round(top_k_overlap, 4),
                "baseline_average_score": round(
                    sum(x["score"] for x in baseline) / len(baseline), 4
                ) if baseline else 0.0,
                "improved_average_score": round(
                    sum(x["score"] for x in improved) / len(improved), 4
                ) if improved else 0.0,
            }
        )

        print(f"[{index}/{len(questions)}] {question}")
        if baseline:
            print(
                "  Baseline top passage:",
                f"{baseline[0]['source']} page {baseline[0]['page']}",
                f"(score={baseline[0]['score']:.3f})",
            )
        if improved:
            print(
                "  Improved top passage:",
                f"{improved[0]['source']} page {improved[0]['page']}",
                f"(score={improved[0]['score']:.3f})",
            )

    baseline_avg = sum(
        row["baseline_average_score"] for row in results
    ) / len(results)

    improved_avg = sum(
        row["improved_average_score"] for row in results
    ) / len(results)

    output = {
        "report_type": "heuristic_retrieval_comparison",
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "question_count": len(results),
        "policy_passage_count": len(chunks),
        "top_k": TOP_K,
        "aggregate_metrics": {
            "baseline_mean_top_k_score": round(baseline_avg, 4),
            "improved_mean_top_k_score": round(improved_avg, 4),
            "score_change": round(improved_avg - baseline_avg, 4),
            "mean_top_k_overlap": round(
                sum(row["top_k_overlap"] for row in results) / len(results),
                4,
            ),
        },
        "limitations": [
            "Scores are lexical heuristics, not semantic relevance judgments.",
            "A higher score does not prove that a passage answers the question.",
            "Review the returned passages and policy page numbers manually.",
            "This script does not modify the application retrieval pipeline.",
        ],
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\nRetrieval comparison completed.")
    print(f"Questions evaluated: {len(results)}")
    print(f"Policy passages: {len(chunks)}")
    print(
        "Baseline mean score:",
        output["aggregate_metrics"]["baseline_mean_top_k_score"],
    )
    print(
        "Improved mean score:",
        output["aggregate_metrics"]["improved_mean_top_k_score"],
    )
    print(f"Saved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
