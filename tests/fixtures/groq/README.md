# Real Groq responses (recorded 2026-11-20, free tier)

Main-call responses from the approved 20-example smoke tests, and the rate-limit
headers that came with them.

**Sanitized:**
- `id` and `x_groq.id` are replaced.
- No provider echoes the input, and reasoning text is never stored. The safeguard
  response is an Aegis decision, but it contains only `{"answer", "confidence"}`, never
  the prompt.

| file | model | dataset |
|---|---|---|
| `gpt-oss-120b-*` | openai/gpt-oss-120b, `reasoning_effort=low`, strict json_schema | Banking77 |
| `qwen3.8-27b-*` | qwen/qwen3.8-27b, `reasoning_effort=none`, strict json_schema | Banking77 |
| `gpt-oss-safeguard-20b-*` | openai/gpt-oss-safeguard-20b, system-message policy, best-effort json_schema | Aegis |
