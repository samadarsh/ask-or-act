"""Minimal async JSON-over-HTTPS for provider adapters (needs the ``providers`` extra).

**API keys** are read from the environment at call time, never stored on objects,
never logged, and never put in error messages. Error messages only quote the
response *body*, cut to 300 characters; provider error bodies don't echo request
headers.

Error mapping, which drives the runner's retry policy:

* 429 (rate limited) and 5xx → ``RetryableError``, carrying the ``retry-after`` header
  (seconds) when the provider sends one.
* A 429 that names a **per-day** limit → ``QuotaExhaustedError``. The runner stops
  cleanly and a later re-run resumes from the cache. Recognised wording: Groq's
  "tokens per day (TPD)" / "requests per day (RPD)", and Google's "..._per_day" /
  "PerDay" quota metric names.
* A 429 with ``insufficient_quota`` → ``ProviderError``. The spend limit is reached, so
  retrying can't help.
* Timeouts and connection failures → ``RetryableError``.
* Other 4xx (bad request, safety block, auth) → ``ProviderError``, which is permanent.
"""

import json
import re
from collections.abc import Mapping
from typing import Any

from ask_or_act.core.decider import QuotaExhaustedError, RetryableError

_PER_DAY = re.compile(r"per day|\((?:RPD|TPD)\)|per_day|PerDay", re.IGNORECASE)
_ORG_ID = re.compile(r"\borg_[A-Za-z0-9]+")


class ProviderError(Exception):
    """A permanent provider failure: retrying the same request won't help."""


def api_key(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ProviderError(
            f"{name} is not set. Put it in .env and run with `uv run --env-file .env ...`"
        )
    return value


def _excerpt(text: str) -> str:
    """Error-body excerpt for messages: whitespace collapsed, account/org ids redacted
    (Groq's 429 bodies name the organization), cut to 300 characters."""
    return _ORG_ID.sub("org_[redacted]", " ".join(text.split()))[:300]


def _retry_after(headers: Mapping[str, str]) -> float | None:
    try:
        return float(headers["retry-after"])
    except (KeyError, ValueError):
        return None


async def post_json(
    url: str,
    body: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
    timeout_s: float = 120.0,
    transport: Any = None,
) -> dict[str, Any]:
    data, _ = await post_json_full(
        url, body, headers=headers, timeout_s=timeout_s, transport=transport
    )
    return data


async def post_json_full(
    url: str,
    body: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
    timeout_s: float = 120.0,
    transport: Any = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Like ``post_json``, but also returns the response headers (lower-cased names),
    e.g. for recording ``x-ratelimit-*``."""
    import httpx

    try:
        async with httpx.AsyncClient(transport=transport, timeout=timeout_s) as client:
            response = await client.post(url, json=body, headers=dict(headers))
    except httpx.TimeoutException as exc:
        raise RetryableError(f"timeout: {type(exc).__name__}") from None
    except httpx.TransportError as exc:
        raise RetryableError(f"network error: {type(exc).__name__}") from None

    status = response.status_code
    resp_headers = {k.lower(): v for k, v in response.headers.items()}
    if status >= 400:
        text = response.text
        wait = _retry_after(resp_headers)
        if status == 429 and "insufficient_quota" in text:
            raise ProviderError(f"HTTP 429 (quota exhausted): {_excerpt(text)}")
        if status == 429 and _PER_DAY.search(text):
            raise QuotaExhaustedError(
                f"HTTP 429 (daily limit): {_excerpt(text)}", retry_after_s=wait
            )
        if status == 429 or status >= 500:
            raise RetryableError(f"HTTP {status}: {_excerpt(text)}", retry_after_s=wait)
        raise ProviderError(f"HTTP {status}: {_excerpt(text)}")
    try:
        data = response.json()
    except json.JSONDecodeError:
        raise RetryableError(f"invalid JSON from provider: {_excerpt(response.text)}") from None
    if not isinstance(data, dict):
        raise ProviderError("provider returned a non-object JSON response")
    return data, resp_headers
