"""Confidence signals from an LLM's JSON answer and its token logprobs.

The model answers ``{"answer": "<option>", "confidence": <0-100>}``. From one call we get:

* **verbalized:** the stated confidence divided by 100.
* **token_logprob:** the probability the model gave its own answer string, i.e.
  exp(Σ logprob) over the tokens of the answer value.
* **verbalized_logprob:** the hybrid from Merdjanovska et al., arXiv:2608.04899 §4
  (Findings of ACL 2026). Instead of reading off the sampled confidence digits, take
  their *expected value* under the token distribution:
  conf = Σ_d 10·d·P(tens = d) + Σ_d d·P(units = d). The two positions are treated as
  independent, which the paper calls an approximation (footnote 6). Each position's
  distribution is renormalized over its digit candidates within the top-k.

  **Generalization (ours).** The paper assumes one token per digit (Limitations).
  Tokenizers that encode "87" as one token are handled by taking the expectation over
  that position's numeric candidates directly (e.g. "87", "90", "85" weighted by their
  probabilities). We record which mode was used in ``hybrid_mode``.

Self-consistency needs several calls, so it is computed by the decider, not here.

Tokens are matched to text through their concatenation, so the parsing works the same
for any provider's tokenization.
"""

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    text: str
    logprob: float
    top: tuple[tuple[str, float], ...] = ()  # (token, logprob) alternatives, incl. itself


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str | None
    confidence: int | None  # 0-100 as stated


def parse_answer(text: str) -> ParsedAnswer:
    """Strict JSON first, then a lenient regex (for models that wrap JSON in prose)."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            ans = data.get("answer")
            conf = data.get("confidence")
            return ParsedAnswer(
                str(ans) if ans is not None else None,
                int(conf) if isinstance(conf, int | float) and not isinstance(conf, bool) else None,
            )
    except json.JSONDecodeError:
        pass
    ans_m = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    conf_m = re.search(r'"confidence"\s*:\s*(\d{1,3})', text)
    return ParsedAnswer(ans_m.group(1) if ans_m else None, int(conf_m.group(1)) if conf_m else None)


def _spans(tokens: Sequence[Token]) -> list[tuple[int, int]]:
    out, pos = [], 0
    for t in tokens:
        out.append((pos, pos + len(t.text)))
        pos += len(t.text)
    return out


def _overlapping(tokens: Sequence[Token], start: int, end: int) -> list[int]:
    return [i for i, (a, b) in enumerate(_spans(tokens)) if a < end and b > start]


def answer_probability(tokens: Sequence[Token]) -> float | None:
    """exp(Σ logprob) over the tokens covering the answer string's characters."""
    text = "".join(t.text for t in tokens)
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if not m or m.start(1) == m.end(1):
        return None
    idx = _overlapping(tokens, m.start(1), m.end(1))
    return math.exp(sum(tokens[i].logprob for i in idx)) if idx else None


def _number_candidates(top: Sequence[tuple[str, float]], prefix: str) -> list[tuple[int, float]]:
    """Alternatives at one position that are ``prefix`` + digits, as (value, prob)."""
    out = []
    for tok, lp in top:
        if tok.startswith(prefix) and tok[len(prefix) :].isdigit():
            out.append((int(tok[len(prefix) :]), math.exp(lp)))
    return out


def _expected(cands: Sequence[tuple[int, float]]) -> float | None:
    total = sum(p for _, p in cands)
    return None if total <= 0 else sum(v * p for v, p in cands) / total


@dataclass(frozen=True)
class Hybrid:
    value: float  # in [0, 1]
    mode: str  # "digits" (paper's per-digit formula) or "number" (whole-number tokens)


def verbalized_logprob(tokens: Sequence[Token]) -> Hybrid | None:
    """Expected stated confidence under the token distribution (arXiv 2608.04899 §4)."""
    text = "".join(t.text for t in tokens)
    m = re.search(r'"confidence"\s*:\s*(\d{1,3})', text)
    if not m:
        return None
    start, end = m.start(1), m.end(1)
    idx = _overlapping(tokens, start, end)
    if not idx:
        return None
    spans = _spans(tokens)
    if len(idx) == 1:
        # One token holds the whole number, possibly with a non-digit prefix (": 87").
        i = idx[0]
        tok_start = spans[i][0]
        prefix = tokens[i].text[: start - tok_start] if start > tok_start else ""
        if tokens[i].text[len(prefix) :] != text[start:end]:
            return None  # token also holds characters after the number
        top = tokens[i].top or ((tokens[i].text, tokens[i].logprob),)
        cands = [(v, p) for v, p in _number_candidates(top, prefix) if v <= 100]
        e = _expected(cands)
        return None if e is None else Hybrid(min(1.0, e / 100), "number")
    # Several tokens: the paper's case. Each must hold exactly one digit.
    digits = [tokens[i] for i in idx]
    if any(len(t.text.strip()) != 1 or not t.text.strip().isdigit() for t in digits):
        return None
    value = 0.0
    for place, t in enumerate(digits):
        prefix = t.text[: len(t.text) - len(t.text.lstrip())]
        top = t.top or ((t.text, t.logprob),)
        cands = [(v, p) for v, p in _number_candidates(top, prefix) if v <= 9]
        e = _expected(cands)
        if e is None:
            return None
        value += 10 ** (len(digits) - 1 - place) * e
    return Hybrid(min(1.0, value / 100), "digits")
