import asyncio
import csv

import pytest

from lib.budget import Budget, BudgetExceeded
from lib.jev_client import JevRunner

STATE = {"claim": "c", "abstract": {"title": "t", "text": "x"}}
QS = {"q": {"type": "noul", "instructions": "Is it?"}}


def fake_call(tokens=1000, fail=False):
    calls = []

    async def call(state, questions, model):
        calls.append((state, questions))
        if fail:
            raise RuntimeError("503")
        return {"model": model, "answers": {"q": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": tokens}}

    return call, calls


def run(coro):
    return asyncio.run(coro)


def test_cache_hit_does_not_call_or_charge(tmp_path):
    budget = Budget(tmp_path / "ledger.csv")
    call, calls = fake_call()
    r = JevRunner(budget, call, cache_dir=tmp_path / "cache")
    kw = dict(stage="t", split="train", question_version="v1")
    first = run(r.ask(STATE, QS, **kw))
    second = run(r.ask(STATE, QS, **kw))
    assert len(calls) == 1 and not first["cached"] and second["cached"]
    assert second["answers"] == first["answers"]
    assert budget.spent_usd == pytest.approx(1000 * 0.042 / 1e6)


def test_ledger_persists_total_across_instances(tmp_path):
    ledger = tmp_path / "ledger.csv"
    call, _ = fake_call(tokens=2_000_000)
    run(JevRunner(Budget(ledger), call, cache_dir=tmp_path / "c").ask(STATE, QS, stage="t", split="train", question_version="v1"))
    assert Budget(ledger).spent_usd == pytest.approx(0.084)
    rows = list(csv.DictReader(open(ledger)))
    assert rows[0]["input_tokens"] == "2000000" and rows[0]["question_version"] == "v1"


def test_budget_refuses_call_that_would_exceed_cap(tmp_path):
    budget = Budget(tmp_path / "ledger.csv", cap_usd=1e-9)
    call, calls = fake_call()
    with pytest.raises(BudgetExceeded):
        run(JevRunner(budget, call, cache_dir=tmp_path / "c").ask(STATE, QS, stage="t", split="train", question_version="v1"))
    assert calls == []


def test_budget_exceeded_keeps_cache_and_resumes(tmp_path):
    ledger, cache = tmp_path / "ledger.csv", tmp_path / "c"
    call, calls = fake_call(tokens=1000)
    kw = dict(stage="t", split="train", question_version="v1")
    run(JevRunner(Budget(ledger), call, cache_dir=cache).ask(STATE, QS, **kw))
    tight = Budget(ledger, cap_usd=Budget(ledger).spent_usd)  # no headroom left
    again = run(JevRunner(tight, call, cache_dir=cache).ask(STATE, QS, **kw))
    assert again["cached"] and len(calls) == 1
    with pytest.raises(BudgetExceeded):
        run(JevRunner(tight, call, cache_dir=cache).ask({**STATE, "claim": "new"}, QS, **kw))


def test_failed_call_releases_reservation(tmp_path):
    budget = Budget(tmp_path / "ledger.csv")
    call, _ = fake_call(fail=True)
    r = JevRunner(budget, call, cache_dir=tmp_path / "c")
    with pytest.raises(RuntimeError):
        run(r.ask(STATE, QS, stage="t", split="train", question_version="v1"))
    assert budget._reserved_usd == 0 and budget.spent_usd == 0
    assert not any((tmp_path / "c").glob("*.json"))


def test_sdk_client_tolerates_a_slow_service():
    # Dev rerank failed twice on transient disconnects/10 s timeouts with SDK defaults (2 retries).
    from lib.jev_client import CLIENT_TIMEOUT_S, RETRY_POLICY, make_client

    assert CLIENT_TIMEOUT_S >= 60
    assert RETRY_POLICY.max_retries >= 5 and RETRY_POLICY.api_timeout_error and RETRY_POLICY.api_connection_error
    assert make_client(api_key="test-key") is not None
