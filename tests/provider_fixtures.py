"""Provider responses in the documented shapes, served by httpx.MockTransport.

These are hand-built from the docs and SDK types (read 2026-11-18), not recordings.
After the approved smoke test they are joined by sanitized real recordings, with input
text removed. No test ever opens a network connection: MockTransport answers in-process.
"""

import itertools
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import httpx

Handler = Callable[[httpx.Request], httpx.Response]
_IDS = itertools.count()


def openai_response(
    answer: str,
    confidence: int,
    *,
    answer_p: float = 0.9,
    in_tokens: int = 120,
    out_tokens: int = 12,
    reasoning: int = 0,
    logprobs: bool = True,
) -> dict[str, Any]:
    text = json.dumps({"answer": answer, "confidence": confidence}, separators=(",", ":"))
    head, tail = '{"answer":"', f'","confidence":{confidence}' + "}"
    lps: list[dict[str, Any]] = [
        {"token": head, "logprob": 0.0, "top_logprobs": []},
        {"token": answer, "logprob": math.log(answer_p), "top_logprobs": []},
        {"token": '","confidence":', "logprob": 0.0, "top_logprobs": []},
        {
            "token": str(confidence),
            "logprob": math.log(0.6),
            "top_logprobs": [
                {"token": str(confidence), "logprob": math.log(0.6)},
                {"token": "50", "logprob": math.log(0.4)},
            ],
        },
        {"token": "}", "logprob": 0.0, "top_logprobs": []},
    ]
    assert "".join(t["token"] for t in lps) == head + answer + tail == text
    content: dict[str, Any] = {"type": "output_text", "text": text, "annotations": []}
    if logprobs:
        content["logprobs"] = lps
    return {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "model": "gpt-6-luna",
        "output": [{"type": "message", "role": "assistant", "content": [content]}],
        "usage": {
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": in_tokens + out_tokens,
        },
    }


def gemini_response(
    answer: str,
    confidence: int,
    *,
    in_tokens: int = 150,
    out_tokens: int = 10,
    thoughts: int = 0,
    finish: str = "STOP",
) -> dict[str, Any]:
    text = json.dumps({"answer": answer, "confidence": confidence}, separators=(",", ":"))
    pieces = ['{"answer":"', answer, '","confidence":', str(confidence), "}"]
    chosen = [{"token": p, "logProbability": 0.0} for p in pieces]
    chosen[1]["logProbability"] = math.log(0.8)
    top = [{"candidates": [c]} for c in chosen]
    top[3] = {
        "candidates": [
            {"token": str(confidence), "logProbability": math.log(0.5)},
            {"token": "40", "logProbability": math.log(0.5)},
        ]
    }
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": finish,
                "logprobsResult": {"chosenCandidates": chosen, "topCandidates": top},
            }
        ],
        "usageMetadata": {
            "promptTokenCount": in_tokens,
            "candidatesTokenCount": out_tokens,
            "thoughtsTokenCount": thoughts,
            "totalTokenCount": in_tokens + out_tokens + thoughts,
        },
        "modelVersion": "gemini-3.5-flash-lite",
    }


def decisions_choice(probs: dict[str, float], tokens: int = 300) -> dict[str, Any]:
    choice = max(probs, key=lambda k: probs[k])
    return {
        "answers": {
            "answer": {
                "type": "choice",
                "choice": choice,
                "probabilities": probs,
                "confidence": probs[choice],
            }
        },
        "usage": {"total_input_tokens": tokens, "total_output_tokens": 0, "total_tokens": tokens},
    }


def decisions_noul(p_true: float, tokens: int = 80) -> dict[str, Any]:
    return {
        "answers": {
            "answer": {"type": "noul", "noul": p_true, "confidence": max(p_true, 1 - p_true)}
        },
        "usage": {"total_input_tokens": tokens, "total_output_tokens": 0, "total_tokens": tokens},
    }


class Recorder:
    """A MockTransport handler that returns queued responses and records requests."""

    def __init__(self, responses: Sequence[httpx.Response | dict[str, Any]]) -> None:
        self.queue = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.queue:
            raise AssertionError("unexpected extra HTTP request")
        item = self.queue.pop(0)
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def never_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"network call attempted: {request.url}")


GROQ_HEADERS = {
    # Documented meanings: *-requests = per DAY, *-tokens = per MINUTE.
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-remaining-requests": "999",
    "x-ratelimit-reset-requests": "1m26.4s",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "7400",
    "x-ratelimit-reset-tokens": "4.5s",
    "x-request-id": "req_test",
}


def groq_response(
    answer: str,
    confidence: int,
    *,
    model: str = "openai/gpt-oss-120b",
    in_tokens: int = 400,
    out_tokens: int = 90,
    reasoning: str | None = None,
    finish: str = "stop",
) -> dict[str, Any]:
    """Chat Completions response as in Groq's API reference (no logprobs)."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": json.dumps({"answer": answer, "confidence": confidence}),
    }
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": f"chatcmpl-test-{next(_IDS)}",  # unique, like real Groq ids
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}],
        "usage": {
            "queue_time": 0.01,
            "prompt_tokens": in_tokens,
            "prompt_time": 0.02,
            "completion_tokens": out_tokens,
            "completion_time": 0.05,
            "total_tokens": in_tokens + out_tokens,
            "total_time": 0.07,
        },
        "system_fingerprint": "fp_test",
        "x_groq": {"id": "req_test"},
    }


def groq_ok(answer: str, confidence: int, **kw: Any) -> httpx.Response:
    return httpx.Response(200, json=groq_response(answer, confidence, **kw), headers=GROQ_HEADERS)


def groq_429(window: str, *, retry_after: str = "7") -> httpx.Response:
    """A rate-limit error. The wording follows Groq's observed 429 messages, which name
    the limit ("tokens per minute (TPM)", "requests per day (RPD)", ...). Groq's docs
    don't specify the body, so the smoke test checks this against a real one."""
    message = (
        "Rate limit reached for model `openai/gpt-oss-120b` in organization "
        f"`org_01abcDEF` service tier `on_demand` on {window}: Limit 1000, Used 1000, "
        "Requested 1. Please try again in 1m26.4s."
    )
    return httpx.Response(
        429,
        json={"error": {"message": message, "type": "requests", "code": "rate_limit_exceeded"}},
        headers={"retry-after": retry_after},
    )


def groq_json_failed(generation: str = '{"token') -> httpx.Response:
    """Groq's 400 when output fails JSON validation (wording from the 2026-11-20 smoke
    test, qwen/qwen3.8-27b with strict json_schema)."""
    return httpx.Response(
        400,
        json={
            "error": {
                "message": "Failed to generate JSON. Please adjust your prompt. See "
                "'failed_generation' for more details.",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
                "failed_generation": generation,
            }
        },
    )


def groq_parse_failed(generation: str) -> httpx.Response:
    """Groq's 400 when the output can't be parsed at all. Seen 2026-11-20 on
    gpt-oss-safeguard-20b: ``failed_generation`` held the model's *reasoning*, which can
    restate the input. Tests pass a harmless placeholder."""
    return httpx.Response(
        400,
        json={
            "error": {
                "message": "Parsing failed. The model generated output that could not be "
                "parsed. Please adjust your prompt. See 'failed_generation' for more details.",
                "type": "invalid_request_error",
                "code": "output_parse_failed",
                "failed_generation": generation,
            }
        },
    )
