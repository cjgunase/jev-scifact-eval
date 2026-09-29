"""Cached, budget-guarded async access to Jev."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from lib.budget import Budget, BudgetExceeded, estimate_tokens
from lib.data import PROCESSED_DIR

MODEL_ID = "jev-1.13.0"
CACHE_DIR = PROCESSED_DIR / "jev_cache"
Call = Callable[[dict, dict, str], Awaitable[dict]]


def cache_key(model: str, state: dict, questions: dict) -> str:
    blob = json.dumps({"model": model, "state": state, "questions": questions}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class JevRunner:
    def __init__(self, budget: Budget, call: Call, cache_dir: Path = CACHE_DIR,
                 model: str = MODEL_ID, max_concurrency: int = 16):
        self.budget, self.call, self.model = budget, call, model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._stopped = False

    async def ask(self, state: dict, questions: dict, *, stage: str, split: str, question_version: str) -> dict:
        path = self.cache_dir / f"{cache_key(self.model, state, questions)}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return {"answers": data["answers"], "input_tokens": data["usage"]["input_tokens"], "cached": True}
        # Reserve inside the semaphore so that queued tasks don't all reserve at once and trip the cap.
        async with self._sem:
            if self._stopped:
                raise BudgetExceeded("runner stopped after an earlier budget refusal")
            est = estimate_tokens(json.dumps({"state": state, "questions": questions}))
            try:
                reserved = self.budget.reserve(est)
            except BudgetExceeded:
                self._stopped = True
                raise
            try:
                resp = await self.call(state, questions, self.model)
            except BaseException:
                self.budget.release(reserved)
                raise
            tokens = int(resp["usage"]["input_tokens"])
            self.budget.settle(reserved, tokens, stage, split, question_version)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(resp))
            tmp.replace(path)
            return {"answers": resp["answers"], "input_tokens": tokens, "cached": False}


def make_sdk_call(client) -> Call:
    """Wrap an AsyncTypeSafeClient so that it takes plain-dict questions and returns plain dicts."""
    from typesafe_sdk import Choice, Noul, Score

    kinds = {"noul": Noul, "choice": Choice, "score": Score}

    async def call(state: dict, questions: dict, model: str) -> dict:
        sdk_qs = {qid: kinds[q["type"]](**{k: v for k, v in q.items() if k != "type"}) for qid, q in questions.items()}
        resp = await client.system_one(state=state, questions=sdk_qs, model=model)
        return {
            "model": resp.model,
            "answers": {qid: a.model_dump() for qid, a in resp.answers.items()},
            "usage": {"input_tokens": resp.usage.input_tokens},
        }

    return call


async def gather_or_stop(coros: list[Awaitable[Any]]) -> list[Any]:
    """Run all coroutines. If any raised BudgetExceeded, re-raise it after the others finish."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    for r in results:
        if isinstance(r, BudgetExceeded):
            raise r
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return results


def run_with_budget(main_coro) -> Any:
    try:
        return asyncio.run(main_coro)
    except BudgetExceeded as e:
        print(f"STOPPED: budget guard refused a call: {e}\nCompleted calls are cached; no output file written.",
              file=sys.stderr)
        sys.exit(2)
