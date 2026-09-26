"""Claude models for the LLM harnesses, on the official Anthropic SDK (kept apart from the OpenAI-compatible
providers in providers.py).

Same interface as the other providers: complete(system, messages, max_tokens, schema) -> text, plus `usage`
and last_reasoning / last_tokens / last_seconds.

Found by probing claude-opus-5-5 on 2026-09-23 with a real Clean Up decision prompt:

  - Structured output (output_config.format with the action schema) is REFUSED, every time, with
    stop_reason "refusal" and category "reasoning_extraction": forcing a "reasoning" field into a
    schema-constrained reply reads to the classifier as extracting the model's reasoning. Without the format
    the same prompt answers normally with the requested JSON, so the schema is not sent; the prompt already
    says to reply with one JSON object and the harness parses and validates it. (The schema would also have
    needed trimming: minItems other than 0/1 is unsupported.)
  - Thinking cannot be switched off on Opus 5.5; effort is the only lever (default medium). The experiments
    run it at `low`, the setting the OpenAI models run at.
  - The system prompt is identical on every call and is marked for caching; the minimum cacheable prefix on
    Opus 5.5 is 512 tokens and it hit from the second call on (2851 tokens read from cache).
  - A refusal can still happen; it is returned as empty text, which the harness counts as an invalid reply
    (a stay), and the stop reason is kept in last_stop for the log.
"""
from __future__ import annotations

import os
import time


class AnthropicProvider:
    def __init__(self, model, key_file="~/.config/anthropic/key", *, effort="low", timeout=600.0, max_retries=6):
        import anthropic
        key = open(os.path.expanduser(key_file)).read().strip()
        self.client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=max_retries)
        self.model, self.effort = model, effort
        self.usage = {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0,
                      "refusals": 0, "seconds": 0.0}
        self.last_reasoning, self.last_tokens, self.last_seconds, self.last_stop = "", 0, 0.0, None
        self.last_usage = {}

    def complete(self, system, messages, max_tokens=400, schema=None):
        t0 = time.time()
        from llm_policy.providers import patient
        resp = patient(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=min(max(max_tokens, 4000), 16000),   # thinking counts against it; 16k keeps a non-streaming call under the SDK's long-request guard
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={"effort": self.effort},
        ), self.model)
        dt = time.time() - t0
        u = resp.usage
        self.last_usage = {"input_tokens": int(u.input_tokens or 0),
                           "cached_tokens": int(u.cache_read_input_tokens or 0),
                           "cache_write_tokens": int(u.cache_creation_input_tokens or 0),
                           "output_tokens": int(u.output_tokens or 0)}
        for k, v in self.last_usage.items():
            self.usage[k] += v
        self.usage["calls"] += 1; self.usage["seconds"] += dt
        self.last_stop = resp.stop_reason
        if resp.stop_reason == "refusal":
            self.usage["refusals"] += 1
        self.last_tokens, self.last_seconds = self.last_usage["output_tokens"], round(dt, 2)
        self.last_reasoning = ""                       # Opus 5.5 returns no chain of thought
        return next((b.text for b in resp.content if b.type == "text"), "")
