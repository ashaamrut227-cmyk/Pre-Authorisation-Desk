import re
from typing import Any


# ---------------------------------------------------------
# 1. CONDITION AND POLICY CONCEPT DEFINITIONS
# ---------------------------------------------------------

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

MONEY_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:rs\.?|inr|₹)\s*"
    r"(\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)

# Explicit policy wording that says the longer of two waiting periods applies.
LONGER_PERIOD_PATTERNS = (
    r"\blonger\s+of\s+(?:the\s+)?(?:two|both)\b",
    r"\blonger\s+waiting\s+period\b",
    r"\bwhichever\s+is\s+longer\b",
    r"\bwhichever\s+period\s+is\s+longer\b",
    r"\bmaximum\s+of\s+the\s+(?:two|both)\s+waiting\s+periods\b",
)


# ---------------------------------------------------------
# 2. TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for deterministic comparisons."""
    text = str(text).lower().replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# 3. DURATION EXTRACTION
# ---------------------------------------------------------

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}

NUMBER_PATTERN = (
    r"\d+(?:\.\d+)?|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety"
)

DURATION_MULTIPLIERS = {
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
}


def _number_value(number_text: str) -> float:
    """Convert a numeric or supported word-based number to float."""
    number_text = number_text.lower()
    if number_text in NUMBER_WORDS:
        return float(NUMBER_WORDS[number_text])
    return float(number_text)


def extract_durations(text: str) -> set[tuple[float, str]]:
    """Extract numeric and word-based durations as (days, unit)."""
    normalized = normalize_text(text)
    pattern = re.compile(
        rf"\b({NUMBER_PATTERN})\s*"
        r"(months?|years?|days?|weeks?)\b"
    )
    durations: set[tuple[float, str]] = set()

    for match in pattern.finditer(normalized):
        number = _number_value(match.group(1))
        unit = match.group(2)
        if unit.endswith("s"):
            unit = unit[:-1]
        durations.add((number * DURATION_MULTIPLIERS[unit], unit))

    return durations


def extract_policy_age_days(text: str) -> float | None:
    """
    Extract an explicitly stated elapsed policy age.

    Examples:
      - Policy was issued eleven months ago.
      - Policy started 11 months ago.
      - The policy has been active for thirty months.
      - Policy active for 30 months.
    """
    normalized = normalize_text(text)
    issued_pattern = re.compile(
        rf"\bpolicy\b.{{0,60}}?"
        rf"\b(?:issued|started|commenced|began|incepted)\b"
        rf".{{0,30}}?"
        rf"({NUMBER_PATTERN})\s*"
        r"(months?|years?|days?|weeks?)\s+ago\b"
    )
    active_pattern = re.compile(
        rf"\bpolicy\b.{{0,60}}?"
        rf"\b(?:has\s+been\s+)?active\s+for\s+"
        rf"({NUMBER_PATTERN})\s*"
        r"(months?|years?|days?|weeks?)\b"
    )

    match = issued_pattern.search(normalized) or active_pattern.search(normalized)
    if not match:
        return None

    number = _number_value(match.group(1))
    unit = match.group(2)
    if unit.endswith("s"):
        unit = unit[:-1]
    return number * DURATION_MULTIPLIERS[unit]


# ---------------------------------------------------------
# 4. NUMERIC FACT EXTRACTION
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 5. CONDITION AND CONCEPT DETECTION
# ---------------------------------------------------------

def _conditions_in_text(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {
        condition
        for condition, phrases in CONDITION_PHRASES.items()
        if any(normalize_text(phrase) in normalized for phrase in phrases)
    }


def _concepts_in_text(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {
        concept
        for concept, phrases in CONCEPT_PHRASES.items()
        if any(normalize_text(phrase) in normalized for phrase in phrases)
    }


def _matches_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def _format_days(days: float) -> str:
    value = int(days) if float(days).is_integer() else days
    return f"{value} days"


def _split_sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", str(text))
        if part.strip()
    ]


# ---------------------------------------------------------
# 6. CONDITION-SPECIFIC WAITING PERIOD EXTRACTION
# ---------------------------------------------------------

def _condition_specific_durations(
    record: dict[str, Any],
    claim_conditions: set[str],
) -> set[float]:
    """Find durations associated with the requested condition."""
    text = str(record.get("text", ""))
    matched_days: set[float] = set()

    for sentence in _split_sentences(text):
        sentence_conditions = _conditions_in_text(sentence)
        sentence_concepts = _concepts_in_text(sentence)
        if claim_conditions & sentence_conditions and "waiting_period" in sentence_concepts:
            matched_days.update(days for days, _ in extract_durations(sentence))

    if matched_days:
        return matched_days

    # Some policy documents put a waiting-period heading and its condition
    # list on separate lines. Only accept a passage-level duration if unique.
    passage_conditions = _conditions_in_text(text)
    passage_concepts = _concepts_in_text(text)
    if not (
        claim_conditions & passage_conditions
        and "waiting_period" in passage_concepts
    ):
        return set()

    all_days = {days for days, _ in extract_durations(text)}
    return all_days if len(all_days) == 1 else set()


# ---------------------------------------------------------
# 7. PRE-EXISTING CONDITION AND LONGER-PERIOD RULE
# ---------------------------------------------------------

def _claim_says_pre_existing(claim: str) -> bool:
    normalized = normalize_text(claim)
    return bool(re.search(
        r"\b(?:pre[- ]existing|preexisting)\s+"
        r"(?:disease|condition|disability|ailment)s?\b"
        r"|\b(?:had|suffered from|diagnosed with|existing)\b"
        r".{0,50}\b(?:before|prior to)\b"
        r".{0,40}\b(?:policy|purchase|inception)\b",
        normalized,
    ))


def _has_longer_period_rule(evidence: list[dict[str, Any]]) -> bool:
    """Require an explicit longer-of-two rule in retrieved policy evidence."""
    combined = " ".join(str(item.get("text", "")) for item in evidence)
    return _matches_any_pattern(combined, LONGER_PERIOD_PATTERNS)


def _is_ped_waiting_passage(text: str) -> bool:
    normalized = normalize_text(text)
    has_ped = bool(re.search(
        r"\bpre[- ]existing\s+(?:disease|condition|disability)s?\b"
        r"|\bpreexisting\s+(?:disease|condition|disability)s?\b",
        normalized,
    ))
    has_wait = "waiting_period" in _concepts_in_text(normalized)
    return has_ped and has_wait


def _resolve_waiting_periods(
    claim: str,
    evidence: list[dict[str, Any]],
    claim_conditions: set[str],
    condition_periods: set[float],
) -> tuple[set[float], str]:
    """
    Resolve condition and PED waiting periods only when policy wording
    explicitly instructs that the longer period applies.
    """
    if not condition_periods:
        return set(), "no_condition_period"

    if not _claim_says_pre_existing(claim):
        return condition_periods, "condition_period_only"

    if not _has_longer_period_rule(evidence):
        return condition_periods, "ped_rule_not_explicit"

    ped_periods: set[float] = set()
    for item in evidence:
        text = str(item.get("text", ""))
        if _is_ped_waiting_passage(text):
            ped_periods.update(days for days, _ in extract_durations(text))

    if not ped_periods:
        return condition_periods, "ped_period_not_found"

    # The longer-of-two clause is explicit, so compare the condition-specific
    # period with the PED period. Do not choose a longer unrelated duration.
    longest_condition = max(condition_periods)
    longest_ped = max(ped_periods)
    selected = max(longest_condition, longest_ped)
    return {selected}, "longer_period_rule_applied"


# ---------------------------------------------------------
# 8. CATARACT AND PROCEDURE SUB-LIMIT EXTRACTION
# ---------------------------------------------------------

def _extract_sub_limits(
    evidence: list[dict[str, Any]],
    claim_conditions: set[str],
) -> list[dict[str, Any]]:
    """Extract monetary sub-limits from relevant passages."""
    limits = []

    for item in evidence:
        text = str(item.get("text", ""))
        if not (claim_conditions & _conditions_in_text(text)):
            continue

        for match in MONEY_PATTERN.finditer(text):
            raw_amount = match.group(1).replace(",", "")
            if not raw_amount:
                continue

            amount = float(raw_amount)
            context = text[max(0, match.start() - 80):min(len(text), match.end() + 60)]
            if not (claim_conditions & _conditions_in_text(context)):
                continue

            normalized_context = normalize_text(context)
            limits.append({
                "amount": int(amount) if amount.is_integer() else amount,
                "currency": "INR",
                "scope": (
                    "per_eye"
                    if re.search(r"\bper\s+(?:each\s+)?eye\b", normalized_context)
                    else "procedure_or_policy_context"
                ),
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id"),
                "source_text": text,
            })

    unique_limits = []
    seen = set()
    for limit in limits:
        key = (
            limit["amount"],
            limit["scope"],
            limit["page_number"],
            limit["chunk_id"],
        )
        if key not in seen:
            seen.add(key)
            unique_limits.append(limit)
    return unique_limits


# ---------------------------------------------------------
# 9. EXCLUSION ANALYSIS
# ---------------------------------------------------------

def _analyze_exclusions(
    evidence: list[dict[str, Any]],
    claim_conditions: set[str],
) -> dict[str, Any]:
    """Find possible exclusions only when condition and exclusion wording co-occur."""
    exclusion_patterns = (
        r"\bexcluded\b",
        r"\bexclusion\b",
        r"\bnot covered\b",
        r"\bshall not be payable\b",
        r"\bwill not be covered\b",
    )
    matches = []

    for item in evidence:
        text = str(item.get("text", ""))
        for sentence in _split_sentences(text):
            sentence_conditions = _conditions_in_text(sentence)
            if (
                claim_conditions & sentence_conditions
                and _matches_any_pattern(sentence, exclusion_patterns)
            ):
                matches.append({
                    "page_number": item.get("page_number"),
                    "chunk_id": item.get("chunk_id"),
                    "text": sentence,
                })

    return {
        "status": (
            "possible_exclusion_found"
            if matches
            else "no_exclusion_found_in_retrieved_evidence"
        ),
        "matches": matches,
        "human_review_required": True,
    }


# ---------------------------------------------------------
# 10. MAIN EVIDENCE VERIFICATION
# ---------------------------------------------------------

def verify_evidence(
    claim: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Conservatively screen a claim against retrieved policy passages.

    This is deterministic evidence analysis, not a final insurance
    coverage or approval decision. Human review is always required.
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
            "coverage_evidence": "not_established",
            "exclusion_analysis": {
                "status": "not_assessed",
                "matches": [],
                "human_review_required": True,
            },
            "sub_limits": [],
            "eligibility_assessment": {
                "status": "not_assessed",
                "policy_age_days": None,
                "required_waiting_period_days": None,
                "human_review_required": True,
            },
            "human_review_required": True,
        }

    claim_text = normalize_text(claim)
    claim_conditions = _conditions_in_text(claim_text)
    claim_concepts = _concepts_in_text(claim_text)
    claim_durations = extract_durations(claim_text)
    claim_duration_days = {days for days, _ in claim_durations}
    policy_age_days = extract_policy_age_days(claim_text)

    eligibility_assessment = {
        "status": "not_assessed",
        "policy_age_days": policy_age_days,
        "required_waiting_period_days": None,
        "human_review_required": True,
    }

    records = []
    for item in evidence:
        raw_text = str(item.get("text", ""))
        normalized = normalize_text(raw_text)
        records.append({
            "item": item,
            "text": raw_text,
            "normalized": normalized,
            "conditions": _conditions_in_text(normalized),
            "concepts": _concepts_in_text(normalized),
            "durations": extract_durations(normalized),
        })

    all_text = " ".join(record["normalized"] for record in records)
    all_conditions = _conditions_in_text(all_text)
    all_concepts = _concepts_in_text(all_text)

    matched_terms = sorted(
        (claim_conditions & all_conditions) | (claim_concepts & all_concepts)
    )
    unmatched_terms = sorted(
        (claim_conditions - all_conditions) | (claim_concepts - all_concepts)
    )
    missing_conditions = sorted(claim_conditions - all_conditions)

    matched_passages = []
    relevant_waiting_passages = []

    for record in records:
        condition_overlap = claim_conditions & record["conditions"]
        concept_overlap = claim_concepts & record["concepts"]
        passage_duration_days = {days for days, _ in record["durations"]}

        if condition_overlap and "waiting_period" in record["concepts"]:
            relevant_waiting_passages.append(record)

        if condition_overlap or concept_overlap:
            matched_passages.append({
                "page_number": record["item"].get("page_number"),
                "chunk_id": record["item"].get("chunk_id"),
                "matched_terms": sorted(condition_overlap | concept_overlap),
                "matched_numeric_facts": sorted(
                    _format_days(days) for days in passage_duration_days
                ),
                "text": record["text"],
            })

    condition_periods: set[float] = set()
    for record in relevant_waiting_passages:
        condition_periods.update(
            _condition_specific_durations(record, claim_conditions)
        )

    resolved_periods, period_resolution = _resolve_waiting_periods(
        claim,
        evidence,
        claim_conditions,
        condition_periods,
    )

    sub_limits = _extract_sub_limits(evidence, claim_conditions)
    exclusion_analysis = _analyze_exclusions(evidence, claim_conditions)

    coverage_evidence = (
        "condition_mentioned"
        if claim_conditions & all_conditions
        else "not_established"
    )

    no_waiting_claim = _matches_any_pattern(
        claim_text, NO_WAITING_PERIOD_PATTERNS
    )
    guarantee_claim = _matches_any_pattern(claim_text, GUARANTEE_PATTERNS)

    status = "unverified"
    reason = (
        "The supplied evidence does not establish the full claim. "
        "Human review is required."
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
            "The requested condition was not found in the supplied "
            "evidence: " + ", ".join(missing_conditions) + "."
        )

    elif no_waiting_claim:
        if relevant_waiting_passages:
            status = "contradicted"
            reason = (
                "The claim says there is no waiting period, but "
                "the retrieved evidence specifies one."
            )
        else:
            reason = (
                "No condition-specific evidence establishes that "
                "the named condition has no waiting period."
            )

    elif "waiting_period" in claim_concepts:
        if not relevant_waiting_passages:
            reason = (
                "No retrieved passage links the requested condition "
                "to a waiting-period rule."
            )
        elif not claim_duration_days:
            reason = (
                "A relevant waiting-period passage was found, "
                "but the claim does not specify a duration."
            )
        elif not condition_periods:
            reason = (
                "The evidence does not establish an unambiguous "
                "condition-specific duration."
            )
        elif claim_duration_days.isdisjoint(condition_periods):
            status = "contradicted"
            reason = (
                "The claimed waiting-period duration conflicts "
                "with the retrieved condition-specific policy evidence."
            )
            unmatched_numeric_facts = sorted(
                _format_days(days) for days in claim_duration_days
            )
        else:
            status = "supported"
            reason = (
                "Retrieved policy text contains a matching waiting "
                "period associated with the named condition. "
                "Individual eligibility still requires human review."
            )
            matched_numeric_facts = sorted(
                _format_days(days)
                for days in (claim_duration_days & condition_periods)
            )

    elif claim_durations:
        reason = (
            "The claim contains a duration, but the retrieved "
            "passages do not establish its relevance."
        )

    elif unmatched_terms:
        reason = (
            "Some recognized policy conditions or concepts were "
            "not found in the evidence: "
            + ", ".join(unmatched_terms) + "."
        )

    elif claim_conditions:
        reason = (
            "The condition appears in the evidence, but the retrieved "
            "text does not establish the requested coverage rule."
        )

    elif claim_concepts:
        status = "supported"
        reason = (
            "The recognized policy concepts appear in the retrieved "
            "text. This does not establish individual eligibility."
        )

    # -----------------------------------------------------
    # POLICY-AGE AND WAITING-PERIOD ASSESSMENT
    # -----------------------------------------------------
    if policy_age_days is not None and resolved_periods:
        required_periods = sorted(resolved_periods)

        if len(required_periods) == 1:
            required_period = required_periods[0]
            eligibility_assessment["required_waiting_period_days"] = required_period

            if policy_age_days < required_period:
                eligibility_assessment["status"] = "waiting_period_not_completed"
                status = "unverified"
                reason = (
                    f"The stated policy age is approximately "
                    f"{_format_days(policy_age_days)}, which is shorter than "
                    f"the applicable waiting period of {_format_days(required_period)}. "
                    "Prior continuous-coverage credits, portability, declared/accepted "
                    "PED status and applicable exceptions must be checked by a human reviewer."
                )
            else:
                eligibility_assessment["status"] = (
                    "waiting_period_elapsed_subject_to_review"
                )
                status = "unverified"
                if period_resolution == "longer_period_rule_applied":
                    reason = (
                        f"The policy age is approximately {_format_days(policy_age_days)} "
                        f"and the longer-of-two rule indicates an applicable waiting "
                        f"period of {_format_days(required_period)}. The elapsed period "
                        "does not establish coverage or approval; continuity, PED "
                        "declaration/acceptance, policy conditions and exceptions "
                        "must be checked by a human reviewer."
                    )
                else:
                    reason = (
                        f"The stated policy age is at least as long as the retrieved "
                        f"waiting period of {_format_days(required_period)}. Eligibility "
                        "still requires verification of continuity, policy conditions "
                        "and applicable exceptions."
                    )
        else:
            eligibility_assessment["status"] = (
                "multiple_waiting_periods_require_review"
            )
            status = "unverified"
            reason = (
                "Multiple possible waiting periods were retrieved and the policy "
                "does not provide enough clearly linked evidence to resolve them. "
                "A human reviewer must determine which provision applies."
            )

    elif policy_age_days is not None and condition_periods:
        eligibility_assessment["status"] = "waiting_period_rule_requires_review"
        status = "unverified"
        reason = (
            "A condition-specific waiting period was found, but the applicable "
            "period could not be resolved from the retrieved policy passages. "
            "Human review is required."
        )

    # Do not convert elapsed time or matching policy facts into approval.
    return {
        "verified": status == "supported",
        "status": status,
        "reason": reason,
        "matched_passages": matched_passages,
        "matched_terms": matched_terms,
        "unmatched_terms": unmatched_terms,
        "matched_numeric_facts": matched_numeric_facts,
        "unmatched_numeric_facts": unmatched_numeric_facts,
        "coverage_evidence": coverage_evidence,
        "exclusion_analysis": exclusion_analysis,
        "sub_limits": sub_limits,
        "eligibility_assessment": eligibility_assessment,
        "human_review_required": True,
    }