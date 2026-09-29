import json

from lib.data import N_TRAIN, load_corpus, load_split, rerank_state, verify_state


def test_split_sizes_and_determinism():
    train = load_split("train")
    assert len(train) == N_TRAIN
    assert [c.claim_id for c in train] == [c.claim_id for c in load_split("train")]
    assert len(load_split("dev")) == 300


def test_evidence_keys_are_int_and_in_corpus():
    corpus = load_corpus()
    for split in ("train", "dev"):
        for c in load_split(split):
            for doc_id, (label, rationales) in c.evidence.items():
                assert isinstance(doc_id, int) and doc_id in corpus
                assert label in {"SUPPORT", "CONTRADICT"}
                assert all(0 <= i < len(corpus[doc_id].sentences) for r in rationales for i in r)


def test_states_contain_no_gold_fields():
    corpus = load_corpus()
    claim = next(c for c in load_split("dev") if c.evidence)
    doc = corpus[next(iter(claim.evidence))]
    for state in (rerank_state(claim, doc), verify_state(claim, doc)):
        assert set(state) == {"claim", "abstract"}
        blob = json.dumps(state)
        for forbidden in ('"evidence"', '"label"', '"rationales"', '"cited_doc_ids"', '"doc_id"'):
            assert forbidden not in blob


def test_verify_state_indexes_every_sentence():
    corpus = load_corpus()
    claim = load_split("dev")[0]
    doc = max(corpus.values(), key=lambda d: len(d.sentences))
    s = verify_state(claim, doc)["abstract"]["sentences"]
    assert list(s) == [f"s{i}" for i in range(len(doc.sentences))]
