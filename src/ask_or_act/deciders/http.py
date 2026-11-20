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
* A 400 whose error code is ``json_validate_failed`` or ``output_parse_failed`` →
  ``InvalidOutputError``. Groq sends these when the model's output doesn't match the
  requested JSON format or can't be parsed. It is a *model* failure, not a bad request.
  The decider records it as an invalid answer.
* **``failed_generation`` never goes into an error message.** It can hold the model's
  reasoning, which may restate the input. On Aegis that would be unsafe text. It is
  removed from every excerpt and passed separately on ``InvalidOutputError``; the
  decider decides whether to keep it (see ``SENSITIVE_TASKS``).
* Other 4xx (bad request, safety block, auth) → ``ProviderError``, which is permanent.
* On SENSITIVE_TASKS the deciders pass every error message through
  ``public_message``, which keeps only our own wording and the error code.
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


class InvalidOutputError(ProviderError):
    """The model's output failed the provider's JSON validation (Groq
    ``json_validate_failed``). ``generation`` is the start of what it produced."""

    def __init__(self, message: str, *, code: str, generation: str) -> None:
        super().__init__(message)
        self.code = code
        self.generation = generation


def _error_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    err = data.get("error") if isinstance(data, dict) else None
    return err if isinstance(err, dict) else {}


def api_key(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ProviderError(
            f"{name} is not set. Put it in .env and run with `uv run --env-file .env ...`"
        )
    return value


INVALID_OUTPUT_CODES = ("json_validate_failed", "output_parse_failed")


def _excerpt(text: str) -> str:
    """Error-body excerpt for messages: model output (``failed_generation``) removed,
    whitespace collapsed, account/org ids redacted (Groq's 429 bodies name the
    organization), cut to 300 characters."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        if "failed_generation" in err:
            err["failed_generation"] = "[removed]"
        text = json.dumps(data)
    elif "failed_generation" in text:
        text = text.split("failed_generation", 1)[0] + "failed_generation [removed]"
    return _ORG_ID.sub("org_[redacted]", " ".join(text.split()))[:300]


_SAFE_HEAD = re.compile(
    r"^(?:HTTP \d{3}(?: \((?:daily limit|quota exhausted)\)| [a-z_]+(?=:))?"
    r"|timeout: \w+|network error: \w+|generation stopped: \w+=\w+"
    r"|no choices in response|model refused|no candidates|blocked or empty answer"
    r"|response \w+|invalid JSON from provider|provider returned a non-object JSON response"
    r"|[A-Z][A-Z0-9_]* is not set)"
)
_CODE = re.compile(r'"code"\s*:\s*"?([A-Za-z0-9_.-]{1,64})')


def public_message(message: str) -> str:
    """An error message with every provider- or model-supplied text removed.

    For SENSITIVE_TASKS (Aegis). Provider messages can quote the model's output, e.g.
    a schema-validation message names the invalid value, and that output can restate
    unsafe input. What survives is our own fixed wording (``HTTP 400``, ``timeout:
    ReadTimeout``, ...) plus the provider's error *code*, which is an identifier."""
    head = m.group(0) if (m := _SAFE_HEAD.match(message)) else "provider error"
    code = _CODE.search(message)
    if code and code.group(1) not in head:
        head = f"{head} {code.group(1)}"
    return f"{head} [details removed: sensitive task]"


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
        err = _error_object(text)
        code = str(err.get("code", ""))
        if code in INVALID_OUTPUT_CODES:
            raise InvalidOutputError(
                f"HTTP {status} {code}: {str(err.get('message', ''))[:200]}",
                code=code,
                generation=str(err.get("failed_generation", "")),
            )
        raise ProviderError(f"HTTP {status}: {_excerpt(text)}")
    try:
        data = response.json()
    except json.JSONDecodeError:
        raise RetryableError(f"invalid JSON from provider: {_excerpt(response.text)}") from None
    if not isinstance(data, dict):
        raise ProviderError("provider returned a non-object JSON response")
    return data, resp_headers
