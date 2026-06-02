from __future__ import annotations

"""
Rule-based suspicion scoring for retrieval-stage document filtering.

The goal of this module is not to perfectly classify malicious content. Instead,
it provides a simple, transparent, and auditable local heuristic that can be
used to down-rank suspicious retrieved chunks before they enter the prompt.

The scoring intentionally prioritizes retrieval-stage transparency:
- all rules are explicit and inspectable
- the score is bounded to the range [0.0, 1.0]
- each matched rule is returned so runs can be analyzed later
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SuspicionResult:
    """
    Result of applying rule-based suspicion scoring to a retrieved chunk.

    Attributes:
        score: Aggregate suspicion score in the range [0.0, 1.0].
        matched_rules: Human-readable names of the triggered heuristic rules.
    """

    score: float
    matched_rules: list[str]


INSTRUCTION_PHRASE_RULES: list[tuple[str, tuple[str, ...], float]] = [
    (
        "ignore_prior_instructions",
        (
            "ignore previous instructions",
            "ignore the previous instructions",
            "disregard previous instructions",
            "forget previous instructions",
        ),
        0.75,
    ),
    (
        "instruction_override_strong",
        (
            "instead follow these instructions",
            "override prior instructions",
            "do not follow the above",
        ),
        0.75,
    ),
    (
        "instruction_override_weak",
        (
            "new instructions",
        ),
        0.35,
    ),
    (
        "exact_output_constraint",
        (
            "respond exactly with",
            "output only",
            "return only",
            "final answer must include",
        ),
        0.50,
    ),
    (
        "verbatim_inclusion_request",
        (
            "copy the identifier verbatim",
            "include the archive reference",
            "include the reference verbatim",
            "do not omit the identifier",
        ),
        0.40,
    ),
    (
        "prompt_exfiltration",
        (
            "reveal the system prompt",
            "print the system prompt",
            "show the hidden prompt",
            "developer instructions",
        ),
        0.45,
    ),
    (
        "assistant_role_play",
        (
            "you are chatgpt",
            "you are an assistant",
            "as an assistant",
            "assistant:",
            "system:",
            "user:",
        ),
        0.20,
    ),
]


DIRECTIVE_PATTERN = re.compile(
    r"(?im)^\s*(?:do\s+not\s+)?(?:ignore|disregard|forget|reveal|print|output|repeat|return|respond|answer|follow|comply|copy|include|omit)\b"
)
PROMPT_MARKUP_PATTERN = re.compile(r"(?i)</?(system|assistant|user|instruction|prompt)>|\[\[.*?\]\]")
EMPHATIC_FORMATTING_PATTERN = re.compile(r"`{3,}|#{2,}|(?<![A-Z])[A-Z]{6,}(?![A-Z])")


def score_text_suspicion(text: str) -> SuspicionResult:
    """
    Compute a local rule-based suspicion score for one text chunk.

    Args:
        text: The chunk text to inspect.

    Returns:
        A `SuspicionResult` containing the bounded score and the names of all
        triggered rules.

    Scoring intuition:
        - explicit instruction-override phrases receive the highest weights
        - prompt/role markers receive medium weights
        - formatting-only or tone-only cues receive lower weights

    The final score is capped at 1.0 to keep the threshold interpretation
    stable across documents.
    """
    lowered = text.lower()
    score = 0.0
    matched_rules: list[str] = []

    for rule_name, phrases, weight in INSTRUCTION_PHRASE_RULES:
        if any(phrase in lowered for phrase in phrases):
            score += weight
            matched_rules.append(rule_name)

    directive_matches = DIRECTIVE_PATTERN.findall(text)
    if len(directive_matches) >= 2:
        score += 0.25
        matched_rules.append("directive_language")

    if PROMPT_MARKUP_PATTERN.search(text):
        score += 0.20
        matched_rules.append("prompt_markup")

    if EMPHATIC_FORMATTING_PATTERN.search(text):
        score += 0.10
        matched_rules.append("abnormal_formatting")

    return SuspicionResult(score=min(score, 1.0), matched_rules=matched_rules)
