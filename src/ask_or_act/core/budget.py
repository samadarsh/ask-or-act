"""Budget guard: never spend more than the configured limit.

It works by reserve, then settle. Before each call the runner reserves the call's
estimated cost, and the reservation is refused if spent + reserved + estimate would
exceed the limit. When the call returns, the reservation is replaced by the actual cost.

So with N calls in flight we can never be committed to more than the limit. The
exception is a decider that underestimates its own cost; once actual spending reveals
that, no further calls start.

No lock is needed: asyncio runs one coroutine at a time and these methods never
``await``, so each check-and-update can't be interrupted part-way. Go would need a mutex
here.
"""


class BudgetExceededError(Exception):
    pass


class BudgetGuard:
    def __init__(self, limit_usd: float | None) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.reserved_usd = 0.0

    def reserve(self, usd: float) -> None:
        if self.limit_usd is None:
            self.reserved_usd += usd
            return
        projected = self.spent_usd + self.reserved_usd + usd
        # Small tolerance so float rounding on an exactly-at-budget run doesn't refuse it.
        if projected > self.limit_usd + 1e-9:
            raise BudgetExceededError(
                f"would reach ${projected:.4f}, over the ${self.limit_usd:.4f} budget "
                f"(spent ${self.spent_usd:.4f}, in flight ${self.reserved_usd:.4f})"
            )
        self.reserved_usd += usd

    def settle(self, reserved_usd: float, actual_usd: float) -> None:
        self.reserved_usd -= reserved_usd
        self.spent_usd += actual_usd
