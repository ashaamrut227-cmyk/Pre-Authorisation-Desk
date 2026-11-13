
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
EVALUATION_DIR = ROOT / "evaluation"

OUTPUT_JSON = EVALUATION_DIR / "real_llmops_results.json"
OUTPUT_REPORT = EVALUATION_DIR / "real_llmops_report.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

QUESTIONS = [
    "What waiting period applies to cataract treatment?",
    "What waiting period applies to a pre-existing disability?",
    "What co-payment applies under the policy?",
    "Are cataract expenses subject to sub-limits?",
    "What is the exact monetary cataract sub-limit?",
    "Does the policy guarantee approval of every pre-authorisation request?",
    "What documents may be needed for pre-authorisation?",
    "What should happen when submitted documents are insufficient?",
    "What should happen if policy wording is unclear?",
    "Does completing a waiting period guarantee that a claim is payable?",
    "Can applicable policy exclusions be ignored for an urgent request?",
    "Does a passage that does not mention a treatment establish coverage?",
    "What is the exact room-rent limit?",
    "Does a diagnosis alone prove policy coverage?",
    "What should happen when submitted documents conflict?",
    "Can a policy clause be invented when retrieval finds no matching clause?",
    "Is a benefit sub-limit necessarily equal to the total sum insured?",
    "What should happen when the policy schedule is missing?",
    "Can a different insurance policy establish this policy's coverage?",
    "What should the system do when evidence is insufficient?"
]

PROMPTS = {
    "baseline": (
        "You are an assistant for a health insurance pre-authorisation desk. "
        "Answer helpfully using the supplied policy context and your general "
        "knowledge. Explain the answer briefly."
    ),
    "evidence_first": (
        "You are an assistant for a health insurance pre-authorisation desk. "
        "Answer using only the supplied policy context. Never invent policy "
        "clauses, waiting periods, limits, exclusions, or approval guarantees. "
        "If the evidence does not establish the answer, clearly state that "
        "the evidence is insufficient. Recommend retrieving the relevant "
        "policy wording or referring the case for human review. "
        "Distinguish confirmed facts from missing information."
    )
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "to", "of", "and", "or", "in", "on", "at", "for", "from",
    "by", "with", "as", "it", "this", "that", "these", "those",
    "what", "which", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "should", "would",
    "will", "may", "might", "must", "has", "have", "had",
    "not", "no", "yes", "under", "after", "before", "also",
    "only", "every", "all", "any", "each", "their", "them",
    "they", "your", "you", "we", "our", "policy", "context",
    "provided", "supplied"
}


# ============================================================
# 2. EXTRACT POLICY TEXT FROM PDF FILES
# ============================================================

def split_chunks(text, page_number, chunk_size=140, overlap=30):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": page_number
            })

        if end == len(words):
            break

        start = end - overlap

    return chunks


def load_policy_chunks():
    pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(
            f"No policy PDFs found in {CORPUS_DIR}"
        )

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is missing. Install it with:\n"
            "python -m pip install pypdf"
        ) from exc

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"Reading policy PDF: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                print(
                    f"Warning: could not read {pdf_path.name}, "
                    f"page {page_number}: {exc}"
                )
                continue

            for chunk in split_chunks(text, page_number):
                chunk["source"] = pdf_path.name
                all_chunks.append(chunk)

    if not all_chunks:
        raise RuntimeError(
            "No extractable text found in the policy PDFs. "
            "Scanned PDFs may require OCR."
        )

    return all_chunks


# ============================================================
# 3. RETRIEVE RELEVANT POLICY PASSAGES
# ============================================================

def tokenize(text):
    words = re.findall(
        r"[a-zA-Z]+|\d+(?:\.\d+)?",
        str(text).lower()
    )

    return [
        word for word in words
        if len(word) > 1 and word not in STOPWORDS
    ]


def retrieve_evidence(question, chunks, top_k=3):
    question_tokens = set(tokenize(question))

    if not question_tokens:
        return []

    ranked = []

    for chunk in chunks:
        chunk_tokens = set(tokenize(chunk["text"]))
        overlap = len(question_tokens & chunk_tokens)
        score = overlap / len(question_tokens)

        if score > 0:
            ranked.append({
                **chunk,
                "retrieval_score": round(score, 4)
            })

    ranked.sort(
        key=lambda item: item["retrieval_score"],
        reverse=True
    )

    selected = []
    seen = set()

    for item in ranked:
        key = (
            item["source"],
            item["page"],
            item["text"][:100]
        )

        if key in seen:
            continue

        seen.add(key)
        selected.append(item)

        if len(selected) >= top_k:
            break

    return selected


def format_evidence(evidence):
    """Accept evidence dictionaries or a plain text string."""

    if isinstance(evidence, str):
        return evidence.strip()

    if not evidence:
        return (
            "No relevant policy passage was retrieved. "
            "The available evidence does not establish coverage."
        )

    passages = []

    for item in evidence:
        passages.append(
            f"[Source: {item['source']}, PDF page {item['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(passages)


# ============================================================
# 4. CALL THE LOCAL OLLAMA MODEL
# ============================================================

def call_llm(system_prompt, question, evidence):
    """
    Call Ollama locally.
    No OpenAI API key or paid API request is required.
    """

    context = format_evidence(evidence)

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved policy evidence:\n{context}"
                )
            }
        ],
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=240
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:
        details = exc.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}: {details[:1000]}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama at "
            f"{OLLAMA_URL}. Make sure Ollama is installed "
            "and running."
        ) from exc

    except TimeoutError as exc:
        raise RuntimeError(
            "Ollama did not respond within 240 seconds. "
            "Try again or use a smaller model."
        ) from exc

    try:
        answer = result["message"]["content"]

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The model returned an empty answer.")

        return answer.strip()

    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Ollama's response did not contain a valid answer."
        ) from exc


# ============================================================
# 5. CALCULATE FOUR EVALUATION METRICS
# ============================================================

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def sentence_claims(answer):
    parts = re.split(
        r"(?<=[.!?])\s+|;\s*",
        answer.strip()
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def evaluate_answer(question, evidence_text, answer):
    """
    Offline lexical proxy metrics.

    These scores use token overlap. They are not semantic
    entailment judgments or proof of policy correctness.
    """

    question_tokens = set(tokenize(question))
    evidence_tokens = set(tokenize(evidence_text))
    answer_tokens = tokenize(answer)

    claims = sentence_claims(answer)
    claim_scores = []

    for claim in claims:
        claim_tokens = set(tokenize(claim))

        score = safe_divide(
            len(claim_tokens & evidence_tokens),
            len(claim_tokens)
        )

        claim_scores.append(score)

    threshold = 0.50

    supported_claims = sum(
        score >= threshold
        for score in claim_scores
    )

    groundedness = safe_divide(
        supported_claims,
        len(claims)
    )

    context_relevance = safe_divide(
        len(question_tokens & evidence_tokens),
        len(question_tokens)
    )

    faithfulness = safe_divide(
        sum(
            token in evidence_tokens
            for token in answer_tokens
        ),
        len(answer_tokens)
    )

    hallucination_rate = (
        1.0 - groundedness
        if claims
        else 0.0
    )

    return {
        "groundedness": round(groundedness, 4),
        "context_relevance": round(context_relevance, 4),
        "faithfulness": round(faithfulness, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "claim_count": len(claims),
        "supported_claim_count": supported_claims,
        "claim_support_scores": [
            round(score, 4)
            for score in claim_scores
        ]
    }


def mean_metric(rows, metric):
    values = [row[metric] for row in rows]

    return round(
        safe_divide(sum(values), len(values)),
        4
    )


# ============================================================
# 6. RUN BOTH PROMPT VARIANTS
# ============================================================

def run_evaluation():
    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks = load_policy_chunks()

    print(f"\nLoaded {len(chunks)} text chunks from policy PDFs.")
    print(f"Model: {MODEL}")
    print(f"Questions: {len(QUESTIONS)}")
    print("LLM provider: Local Ollama")
    print("API charges: No OpenAI API calls are made.\n")

    all_results = []

    variant_rows = {
        "baseline": [],
        "evidence_first": []
    }

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{index}/{len(QUESTIONS)}] {question}")

        evidence = retrieve_evidence(question, chunks)
        evidence_text = format_evidence(evidence)

        if not evidence:
            print("  Warning: no matching policy passage was retrieved.")

        question_result = {
            "question_id": f"Q{index:02d}",
            "question": question,
            "retrieved_evidence": evidence,
            "variants": {}
        }

        for variant_name, prompt in PROMPTS.items():
            print(f"  Running {variant_name} prompt...")

            answer = call_llm(
                system_prompt=prompt,
                question=question,
                evidence=evidence
            )

            metrics = evaluate_answer(
                question,
                evidence_text,
                answer
            )

            result = {
                "answer": answer,
                **metrics
            }

            question_result["variants"][variant_name] = result
            variant_rows[variant_name].append(result)

        all_results.append(question_result)

    metric_names = [
        "groundedness",
        "context_relevance",
        "faithfulness",
        "hallucination_rate"
    ]

    aggregate = {}

    for variant_name, rows in variant_rows.items():
        aggregate[variant_name] = {
            metric: mean_metric(rows, metric)
            for metric in metric_names
        }

    changes = {
        metric: round(
            aggregate["evidence_first"][metric]
            - aggregate["baseline"][metric],
            4
        )
        for metric in metric_names
    }

    report = {
        "report_type": "local_llm_policy_retrieval_evaluation",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": MODEL,
        "provider": "Ollama",
        "llm_api_called": True,
        "api_is_local": True,
        "question_count": len(QUESTIONS),
        "policy_pdf_files": sorted(
            path.name
            for path in CORPUS_DIR.glob("*.pdf")
        ),
        "retrieval_method": (
            "Lexical overlap over locally extracted PDF text; "
            "up to three passages retrieved per question."
        ),
        "prompt_variants": PROMPTS,
        "metric_definitions": {
            "groundedness": (
                "Fraction of answer sentences with at least 50% "
                "meaningful token overlap with evidence."
            ),
            "context_relevance": (
                "Fraction of meaningful question tokens found in evidence."
            ),
            "faithfulness": (
                "Fraction of meaningful answer tokens found in evidence."
            ),
            "hallucination_rate": (
                "Fraction of answer sentences below the lexical-support threshold."
            )
        },
        "limitations": [
            "Metrics are lexical proxies, not semantic judgments.",
            "Word overlap cannot prove that a claim is factually correct.",
            "Retrieval uses lexical ranking, not the existing hybrid-search index.",
            "Retrieved passages and model answers need human review.",
            "Local model output quality depends on available hardware and model capability."
        ],
        "aggregate_metrics": aggregate,
        "change_evidence_first_minus_baseline": changes,
        "scenario_results": all_results
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    labels = {
        "groundedness": "Groundedness",
        "context_relevance": "Context relevance",
        "faithfulness": "Faithfulness",
        "hallucination_rate": "Hallucination rate"
    }

    baseline = aggregate["baseline"]
    evidence_first = aggregate["evidence_first"]

    lines = [
        "LOCAL LLM POLICY EVALUATION REPORT",
        "=" * 40,
        f"Model: {MODEL}",
        "Provider: Ollama",
        f"Questions: {len(QUESTIONS)}",
        "LLM generated answers: Yes",
        "OpenAI API calls: No",
        "Metrics: Lexical proxies",
        "",
        "AGGREGATE METRICS (0.0 TO 1.0)",
        "-" * 40
    ]

    for metric in metric_names:
        lines.append(
            f"{labels[metric]}: "
            f"baseline={baseline[metric]:.4f}, "
            f"evidence-first={evidence_first[metric]:.4f}, "
            f"change={changes[metric]:+.4f}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 40,
        "Higher is better: groundedness, context relevance, faithfulness.",
        "Lower is better: hallucination rate.",
        "These metrics are lexical proxies, not semantic verification.",
        "",
        "LIMITATIONS",
        "-" * 40,
        "- Answers were generated by a local model.",
        "- Retrieval uses lexical ranking over extracted PDF text.",
        "- Verify policy clauses and individual answers before drawing conclusions.",
        "",
        f"JSON results: {OUTPUT_JSON}",
        f"Text report: {OUTPUT_REPORT}"
    ])

    OUTPUT_REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\n" + "\n".join(lines))
    print("\nDetailed results saved successfully.")


# ============================================================
# 7. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        run_evaluation()

    except KeyboardInterrupt:
        print("\nEvaluation stopped by the user.")
        sys.exit(130)

    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("The evaluation did not complete successfully.")
        sys.exit(1)
