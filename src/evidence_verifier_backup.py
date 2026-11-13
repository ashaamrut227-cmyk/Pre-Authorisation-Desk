import re
from typing import Any


CONDITION_PHRASES = {
    "cataract": ("cataract",),
    "age-related eye ailments": ("age-related eye ailments",),
    "dental implants": ("dental implants", "dental implant"),
    "pre-existing disability": (
        "pre-existing disability",
        "pre-existing disabilities",
    ),
    "pre-existing disease": (
        "pre-existing disease",
        "pre-existing diseases",
    ),
    "pre-existing condition": (
        "pre-existing condition",
        "pre-existing conditions",
    ),
}

CONCEPT_PHRASES = {
    "waiting_period": ("waiting period", "waiting periods"),
    "exclusion": ("exclusion", "exclusions", "excluded"),
    "continuous_coverage": ("continuous coverage",),
    "per_eye_limit": ("per each eye", "per eye", "each eye"),
}

NO_WAITING_PERIOD_PATTERNS = (
    r"\bno waiting period\b",
    r"\bwithout a waiting period\b",
    r"\bwithout waiting period\b",
    r"\bzero waiting period\b",
    r"\bimmediately covered\b",
    r"\bcovered immediately\b",
)

GUARANTEE_PATTERNS = (
    r"\balways approved\b",
    r"\bguaranteed approval\b",
    r"\bautomatically approved\b",
    r"\bwill definitely be approved\b",
    r"\bapproval is guaranteed\b",
)


def normalize_text(text: str) -> str:
    """Normalize text for deterministic comparisons."""
    text = str(text).lower().replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_durations(text: str) -> set[tuple[float, str]]:
    """Extract durations such as 24 months, 2 years, or 30 days."""
    normalized = normalize_text(text)

    pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(months?|years?|days?|weeks?)\b"
    )

    multipliers = {
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365,
    }

    durations = set()

    for match in pattern.finditer(normalized):
        number = float(match.group(1))
        unit = match.group(2)

        if unit.endswith("s"):
            unit = unit[:-1]

        durations.add((number * multipliers[unit], unit))

    return durations


def extract_numeric_facts(text: str) -> set[str]:
    """Extract numeric facts for compatibility with earlier code."""
    normalized = normalize_text(text)

    pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*"
        r"(months?|years?|days?|weeks?|hours?|per\s+eye|%)?\b"
    )

    facts = set()

    for match in pattern.finditer(normalized):
        number = match.group(1)
        unit = (match.group(2) or "").strip()

        if unit.endswith("s") and unit != "hours":
            unit = unit[:-1]

        facts.add(f"{number} {unit}".strip())

    return facts


def _conditions_in_text(text: str) -> set[str]:
    normalized = normalize_text(text)

    return {
        condition
        for condition, phrases in CONDITION_PHRASES.items()
        if any(
            normalize_text(phrase) in normalized
            for phrase in phrases
        )
    }


def _concepts_in_text(text: str) -> set[str]:
    normalized = normalize_text(text)

    return {
        concept
        for concept, phrases in CONCEPT_PHRASES.items()
        if any(
            normalize_text(phrase) in normalized
            for phrase in phrases
        )
    }


def _matches_any_pattern(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    normalized = normalize_text(text)
    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def _format_days(days: float) -> str:
    value = int(days) if days.is_integer() else days
    return f"{value} days"


def _condition_specific_durations(
    record: dict[str, Any],
    claim_conditions: set[str],
) -> set[float]:
    """
    Find durations that can be associated with the requested condition.

    Prefer sentence-level matches. If the duration and condition are
    listed in separate sentences, allow the fallback only when the
    passage has one distinct duration. Multiple competing durations
    remain ambiguous unless a sentence directly links one to the
    requested condition.
    """
    text = str(record.get("text", ""))
    sentences = re.split(r"(?<=[.!?])\s+", text)

    matched_days: set[float] = set()

    for sentence in sentences:
        sentence_conditions = _conditions_in_text(sentence)
        sentence_concepts = _concepts_in_text(sentence)

        if (
            claim_conditions & sentence_conditions
            and "waiting_period" in sentence_concepts
        ):
            matched_days.update(
                days for days, _ in extract_durations(sentence)
            )

    if matched_days:
        return matched_days

    passage_conditions = _conditions_in_text(text)
    passage_concepts = _concepts_in_text(text)

    if not (
        claim_conditions & passage_conditions
        and "waiting_period" in passage_concepts
    ):
        return set()

    all_days = {
        days for days, _ in extract_durations(text)
    }

    # A unique duration in a passage listing the applicable conditions
    # can support a claim even if the duration appears in a prior
    # sentence, as in the existing workflow test fixture.
    if len(all_days) == 1:
        return all_days

    # Do not select between competing durations by guessing.
    return set()


def verify_evidence(
    claim: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Conservatively screen a claim against retrieved policy passages.

    This deterministic lexical check is not semantic proof and does not
    make a final insurance coverage or approval decision.
    """
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Claim must be a non-empty string.")

    if not isinstance(evidence, list) or not evidence:
        return {
            "verified": False,
            "status": "unverified",
            "reason": "No evidence passages were supplied.",
            "matched_passages": [],
            "matched_terms": [],
            "unmatched_terms": [],
            "matched_numeric_facts": [],
            "unmatched_numeric_facts": [],
        }

    claim_text = normalize_text(claim)
    claim_conditions = _conditions_in_text(claim_text)
    claim_concepts = _concepts_in_text(claim_text)
    claim_durations = extract_durations(claim_text)
    claim_duration_days = {
        days for days, _ in claim_durations
    }

    records = []

    for item in evidence:
        raw_text = str(item.get("text", ""))
        normalized = normalize_text(raw_text)

        records.append(
            {
                "item": item,
                "text": raw_text,
                "normalized": normalized,
                "conditions": _conditions_in_text(normalized),
                "concepts": _concepts_in_text(normalized),
                "durations": extract_durations(normalized),
            }
        )

    all_text = " ".join(
        record["normalized"] for record in records
    )
    all_conditions = _conditions_in_text(all_text)
    all_concepts = _concepts_in_text(all_text)

    matched_terms = sorted(
        (claim_conditions & all_conditions)
        | (claim_concepts & all_concepts)
    )
    unmatched_terms = sorted(
        (claim_conditions - all_conditions)
        | (claim_concepts - all_concepts)
    )
    missing_conditions = sorted(
        claim_conditions - all_conditions
    )

    matched_passages = []
    relevant_waiting_passages = []

    for record in records:
        condition_overlap = (
            claim_conditions & record["conditions"]
        )
        concept_overlap = (
            claim_concepts & record["concepts"]
        )
        passage_duration_days = {
            days for days, _ in record["durations"]
        }

        if (
            condition_overlap
            and "waiting_period" in record["concepts"]
        ):
            relevant_waiting_passages.append(record)

        if condition_overlap or concept_overlap:
            matched_passages.append(
                {
                    "page_number": record["item"].get(
                        "page_number"
                    ),
                    "chunk_id": record["item"].get("chunk_id"),
                    "matched_terms": sorted(
                        condition_overlap | concept_overlap
                    ),
                    "matched_numeric_facts": sorted(
                        _format_days(days)
                        for days in passage_duration_days
                    ),
                    "text": record["text"],
                }
            )

    relevant_duration_days: set[float] = set()

    for record in relevant_waiting_passages:
        relevant_duration_days.update(
            _condition_specific_durations(
                record,
                claim_conditions,
            )
        )

    no_waiting_claim = _matches_any_pattern(
        claim_text,
        NO_WAITING_PERIOD_PATTERNS,
    )
    guarantee_claim = _matches_any_pattern(
        claim_text,
        GUARANTEE_PATTERNS,
    )

    status = "unverified"
    reason = (
        "The supplied evidence does not establish the full claim. "
        "Retrieve a more specific provision or escalate for human review."
    )
    matched_numeric_facts = []
    unmatched_numeric_facts = []

    if guarantee_claim:
        reason = (
            "The claim guarantees approval, which the supplied policy "
            "passages do not establish. Human review is required."
        )

    elif missing_conditions:
        reason = (
            "The claim mentions a condition or procedure not found in "
            "the supplied evidence: "
            + ", ".join(missing_conditions)
            + "."
        )

    elif no_waiting_claim:
        if relevant_waiting_passages:
            status = "contradicted"
            reason = (
                "The claim says there is no waiting period, but a passage "
                "for the same condition specifies a waiting period."
            )
        else:
            reason = (
                "No condition-specific passage establishes that the "
                "named condition has no waiting period."
            )

    elif "waiting_period" in claim_concepts:
        if not relevant_waiting_passages:
            reason = (
                "No retrieved passage links the named condition to a "
                "waiting-period rule. Human review is required."
            )

        elif not claim_duration_days:
            reason = (
                "A relevant waiting-period passage was found, but the "
                "claim does not specify a duration to compare."
            )

        elif not relevant_duration_days:
            reason = (
                "The relevant passages do not establish an unambiguous "
                "condition-specific duration. Human review is required."
            )

        elif claim_duration_days.isdisjoint(
            relevant_duration_days
        ):
            status = "contradicted"
            reason = (
                "The claimed waiting-period duration conflicts with the "
                "duration associated with the requested condition."
            )
            unmatched_numeric_facts = sorted(
                _format_days(days)
                for days in claim_duration_days
            )

        else:
            status = "supported"
            reason = (
                "Retrieved policy text contains a matching duration "
                "associated with the named condition and waiting period. "
                "Individual eligibility still requires human review."
            )
            matched_numeric_facts = sorted(
                _format_days(days)
                for days in (
                    claim_duration_days & relevant_duration_days
                )
            )

    elif claim_durations:
        reason = (
            "The claim contains a duration, but the retrieved passages "
            "do not establish its relevance to the named condition."
        )

    elif unmatched_terms:
        reason = (
            "Some recognized policy conditions or concepts were not found "
            "in the evidence: "
            + ", ".join(unmatched_terms)
            + "."
        )

    elif claim_conditions:
        reason = (
            "The condition appears in the evidence, but the retrieved text "
            "does not establish the requested policy rule for it."
        )

    elif claim_concepts:
        status = "supported"
        reason = (
            "The recognized policy concepts appear in the retrieved text. "
            "This lexical check does not establish the full claim or "
            "eligibility for insurance coverage."
        )

    return {
        "verified": status == "supported",
        "status": status,
        "reason": reason,
        "matched_passages": matched_passages,
        "matched_terms": matched_terms,
        "unmatched_terms": unmatched_terms,
        "matched_numeric_facts": matched_numeric_facts,
        "unmatched_numeric_facts": unmatched_numeric_facts,
    }
