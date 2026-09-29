from lib.questions import (MAX_SENTENCE_QUESTIONS_PER_REQUEST, QUESTION_VERSION, relevance_questions,
                           verify_question_chunks)


def test_relevance_is_single_ordered_score():
    q = relevance_questions()
    assert list(q) == ["relevance"] and q["relevance"]["type"] == "score"
    assert 2 <= len(q["relevance"]["criteria"]) <= 10


def test_short_abstract_is_one_chunk_with_verdict():
    chunks = verify_question_chunks(8)
    assert len(chunks) == 1
    assert chunks[0]["verdict"]["type"] == "choice"
    assert set(chunks[0]["verdict"]["criteria"]) == {"supports", "contradicts", "not_enough_info"}
    assert [k for k in chunks[0] if k.startswith("evidence_")] == [f"evidence_s{i}" for i in range(8)]


def test_long_abstract_chunks_cover_every_sentence_once():
    n = 367
    chunks = verify_question_chunks(n)
    ids = [k for c in chunks for k in c if k.startswith("evidence_")]
    assert ids == [f"evidence_s{i}" for i in range(n)]
    assert all(sum(k.startswith("evidence_") for k in c) <= MAX_SENTENCE_QUESTIONS_PER_REQUEST for c in chunks)
    assert sum("verdict" in c for c in chunks) == 1 and "verdict" in chunks[0]


def test_evidence_question_points_at_its_sentence():
    q = verify_question_chunks(3)[0]["evidence_s2"]
    assert q["type"] == "noul" and "`abstract.sentences.s2`" in q["instructions"]


def test_version_is_set():
    assert QUESTION_VERSION.startswith("v")
