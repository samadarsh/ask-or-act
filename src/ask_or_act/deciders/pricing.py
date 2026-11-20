"""Dated, cited price table. A real decider refuses to run on a model not listed here,
because the budget guard can't protect you from a price it doesn't know.

Prices are USD per 1M tokens, standard (interactive) tier, short context. Each entry
records where and when it was read, and whether the number was seen verbatim on the
page (``verified=True``) or only through a summary of it (``False``). Unverified prices
are printed as a warning in every cost estimate.

**Free tiers.** A provider with a free tier (Groq) still gets its published *paid*
(developer-tier) price here. On the free tier, ``Decision.cost_usd`` is the cost at that
price, so the study can report cost per 1,000 decisions, and ``charged_usd`` is 0.

Update procedure: re-read the source page, update the numbers and the ``retrieved``
date, and note it in the changelog. Never edit a price from memory.
"""

from dataclasses import dataclass

from ask_or_act.core.decider import Pricing

GEMINI_PRICING = "https://aistudio.google.com/docs/pricing"
OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing"
GROQ_MODEL_PAGE = "https://console.groq.com/docs/model/"  # + model id; also /docs/models


@dataclass(frozen=True)
class Price:
    model: str
    provider: str
    input_per_mtok: float
    output_per_mtok: float  # includes thinking/reasoning tokens for both providers
    source: str
    retrieved: str
    verified: bool
    cached_input_per_mtok: float | None = None
    valid_until: str | None = None
    notes: str = ""
    free_tier: bool = False  # the provider offers this model on a $0 tier

    @property
    def pricing(self) -> Pricing:
        return Pricing(self.input_per_mtok, self.output_per_mtok)

    def describe(self) -> str:
        flag = "" if self.verified else "  ⚠ UNVERIFIED: confirm on the source page"
        until = f", valid until {self.valid_until}" if self.valid_until else ""
        return (
            f"{self.model}: ${self.input_per_mtok}/1M in, ${self.output_per_mtok}/1M out "
            f"({self.source}, read {self.retrieved}{until}){flag}"
        )


PRICES: dict[str, Price] = {
    p.model: p
    for p in [
        Price(
            "gemini-decision-1",
            "google",
            0.10,
            0.0,
            GEMINI_PRICING,
            "2026-11-18",
            verified=False,
            notes="Zero output tokens. Free tier: free of charge, but data is used to "
            "improve Google products. $0.10 seen only via page summaries.",
        ),
        Price(
            "gemini-3.5-flash-lite",
            "google",
            0.30,
            2.50,
            GEMINI_PRICING,
            "2026-11-18",
            verified=True,
            notes="Stable. Output price includes thinking tokens.",
        ),
        Price(
            "gemini-3.8-flash",
            "google",
            0.75,
            3.75,
            GEMINI_PRICING,
            "2026-11-18",
            verified=True,
            valid_until="2026-12-31",
            notes="Stable. Rises to $1.50 / $7.50 on 2027-01-01. Docs summary says "
            "temperature is ignored on Gemini 3.6+ (unverified).",
        ),
        Price(
            "gpt-6-luna",
            "openai",
            0.10,
            0.50,
            OPENAI_PRICING,
            "2026-11-18",
            verified=True,
            cached_input_per_mtok=0.01,
            notes="'Most efficient model for focused, high-volume tasks'. Reasoning "
            "effort none..max. Short context (<=272K).",
        ),
        Price(
            "gpt-6.1-terra",
            "openai",
            1.00,
            5.00,
            OPENAI_PRICING,
            "2026-11-18",
            verified=True,
            cached_input_per_mtok=0.05,
        ),
        Price(
            "gpt-6.2-sol",
            "openai",
            2.00,
            10.00,
            OPENAI_PRICING,
            "2026-11-18",
            verified=True,
            cached_input_per_mtok=0.10,
            notes="Reasoning effort low..max (rejects 'none').",
        ),
        # Groq: developer-tier prices from each model's page; /docs/models agrees.
        # groq.com/pricing has no model prices (marketing copy only).
        Price(
            "openai/gpt-oss-120b",
            "groq",
            0.15,
            0.60,
            GROQ_MODEL_PAGE + "openai/gpt-oss-120b",
            "2026-11-20",
            verified=True,
            cached_input_per_mtok=0.075,
            free_tier=True,
            notes="Production. Free tier: 30 RPM, 1K RPD, 8K TPM, 200K TPD. "
            "Output price includes reasoning tokens.",
        ),
        Price(
            "openai/gpt-oss-20b",
            "groq",
            0.075,
            0.30,
            GROQ_MODEL_PAGE + "openai/gpt-oss-20b",
            "2026-11-20",
            verified=True,
            cached_input_per_mtok=0.037,
            free_tier=True,
            notes="Production. Free tier: 30 RPM, 1K RPD, 8K TPM, 200K TPD.",
        ),
        Price(
            "openai/gpt-oss-safeguard-20b",
            "groq",
            0.075,
            0.30,
            GROQ_MODEL_PAGE + "openai/gpt-oss-safeguard-20b",
            "2026-11-20",
            verified=True,
            cached_input_per_mtok=0.037,
            free_tier=True,
            notes="Preview; safety classifier (policy in the system message). Free tier: "
            "3 RPM, 1K RPD, 2K TPM, 200K TPD.",
        ),
        Price(
            "qwen/qwen3.8-27b",
            "groq",
            0.80,
            4.00,
            GROQ_MODEL_PAGE + "qwen/qwen3.8-27b",
            "2026-11-20",
            verified=True,
            free_tier=True,
            notes="Preview. Free tier: 30 RPM, 1K RPD, 8K TPM, 200K TPD. Max output 16,384.",
        ),
        # allam-2-7b is deliberately absent: its price is "Pending" on its model page,
        # so cost per decision can't be reported. Add it once a price is published.
    ]
}


def price_for(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError:
        raise ValueError(
            f"no price on record for model {model!r}; add a dated, cited entry to "
            f"deciders/pricing.py before running it (known: {sorted(PRICES)})"
        ) from None
