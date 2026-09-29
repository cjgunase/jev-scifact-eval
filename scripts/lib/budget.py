"""Persistent spend ledger with a hard USD cap on TypeSafe calls."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

from lib.data import PROCESSED_DIR

PRICE_PER_TOKEN_USD = 0.042 / 1_000_000  # jev-1.13.0, input tokens only; output is free
BUDGET_CAP_USD = 0.90                    # user limit is $1.00; keep a margin
LEDGER_PATH = PROCESSED_DIR / "spend_ledger.csv"
TOKEN_SAFETY_FACTOR = 1.5                # pre-call estimate is deliberately pessimistic
FIELDS = ["timestamp", "stage", "split", "question_version", "input_tokens", "cost_usd", "total_usd"]


class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(payload: str) -> int:
    return math.ceil(len(payload.encode()) / 4 * TOKEN_SAFETY_FACTOR)


class Budget:
    def __init__(self, ledger_path: Path = LEDGER_PATH, cap_usd: float = BUDGET_CAP_USD):
        self.ledger_path = Path(ledger_path)
        self.cap_usd = cap_usd
        self._reserved_usd = 0.0
        self.spent_usd = 0.0
        if self.ledger_path.exists():
            with open(self.ledger_path) as fh:
                self.spent_usd = sum(float(r["cost_usd"]) for r in csv.DictReader(fh))

    def reserve(self, est_tokens: int) -> float:
        cost = est_tokens * PRICE_PER_TOKEN_USD
        if self.spent_usd + self._reserved_usd + cost > self.cap_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} + reserved ${self._reserved_usd:.4f} + next ${cost:.6f} "
                f"> cap ${self.cap_usd:.2f}"
            )
        self._reserved_usd += cost
        return cost

    def release(self, reserved: float) -> None:
        self._reserved_usd = max(0.0, self._reserved_usd - reserved)

    def settle(self, reserved: float, input_tokens: int, stage: str, split: str, question_version: str) -> None:
        self.release(reserved)
        cost = input_tokens * PRICE_PER_TOKEN_USD
        self.spent_usd += cost
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.ledger_path.exists()
        with open(self.ledger_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "stage": stage, "split": split, "question_version": question_version,
                "input_tokens": input_tokens, "cost_usd": f"{cost:.8f}", "total_usd": f"{self.spent_usd:.8f}",
            })
