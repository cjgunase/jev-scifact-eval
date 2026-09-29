"""All Jev questions used in the SciFact evaluation.

Bump QUESTION_VERSION whenever any wording changes. The ledger and outputs record it,
and the dev split may only be run with the frozen version (see plan Task 9).
"""

from __future__ import annotations

QUESTION_VERSION = "v1"
MAX_SENTENCE_QUESTIONS_PER_REQUEST = 40
VERDICT_TO_SCIFACT = {"supports": "SUPPORT", "contradicts": "CONTRADICT"}

RELEVANCE_LEVELS = [
    "The abstract is about a different topic from the claim.",
    "The abstract is on the same general topic, but it does not study what the claim asserts.",
    "The abstract studies what the claim is about, but its reported findings neither confirm nor refute the claim.",
    "The abstract reports findings that directly confirm or refute the claim.",
]

VERDICT_CRITERIA = {
    "supports": "The abstract reports findings showing that the claim is true.",
    "contradicts": (
        "The abstract reports findings showing that the claim is false, including an effect in the "
        "opposite direction to the claim or no effect where the claim asserts one."
    ),
    "not_enough_info": (
        "The abstract does not report a finding that shows whether the claim is true or false, "
        "including when it is on the same topic but studies something different."
    ),
}


def relevance_questions() -> dict:
    return {
        "relevance": {
            "type": "score",
            "instructions": "How directly does the abstract in `abstract` address the scientific claim in `claim`?",
            "criteria": RELEVANCE_LEVELS,
        }
    }


def _verdict_question() -> dict:
    return {
        "type": "choice",
        "instructions": "Based only on `abstract`, does it support or contradict the scientific claim in `claim`?",
        "criteria": VERDICT_CRITERIA,
    }


def _evidence_question(i: int) -> dict:
    return {
        "type": "noul",
        "instructions": (
            f"Does sentence `abstract.sentences.s{i}` report a result or finding that, on its own or together "
            "with neighbouring sentences, helps show whether the claim in `claim` is true or false?"
        ),
    }


def verify_question_chunks(n_sentences: int) -> list[dict]:
    chunks = []
    for start in range(0, max(n_sentences, 1), MAX_SENTENCE_QUESTIONS_PER_REQUEST):
        chunk = {"verdict": _verdict_question()} if start == 0 else {}
        for i in range(start, min(start + MAX_SENTENCE_QUESTIONS_PER_REQUEST, n_sentences)):
            chunk[f"evidence_s{i}"] = _evidence_question(i)
        chunks.append(chunk)
    return chunks
