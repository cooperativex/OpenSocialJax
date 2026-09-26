"""Hosted-API providers for the LLM harnesses, beside the vLLM ChatProvider in open_cleanup_policy.

Every provider has the interface the harness uses: complete(system, messages, max_tokens, schema) -> text,
a running `usage` dict, and last_reasoning / last_tokens / last_seconds after each call.

APIProvider covers the OpenAI-compatible endpoints (OpenAI itself, DeepSeek, Zhipu GLM). They differ in three
ways, all found by probing the real endpoints on 2026-09-23 with a real decision prompt:

  - how a JSON reply is enforced: OpenAI takes a strict json_schema; DeepSeek rejects json_schema ("This
    response_format type is unavailable now") and takes json_object; GLM takes json_object.
  - thinking: deepseek-flash thinks by default and, left alone, spent the whole 6000-token budget reasoning and
    returned no JSON -- with thinking disabled it answers in ~3 s and ~200 tokens. glm-5.3-flash cannot stop
    thinking ("use low, high or max"); at reasoning_effort=low it answers in ~10 s and ~210 tokens, against
    ~3 minutes and a blown budget at its default.
  - which token-limit parameter the endpoint takes (OpenAI reasoning models want max_completion_tokens).

Keys are read from a file outside the repository (~/.config/<vendor>/key), never from the code or the logs.

The Anthropic provider lives in anthropic_provider.py, on the official Anthropic SDK.
"""
from __future__ import annotations

import os
import random
import time


def patient(call, what, max_wait=900.0, give_up_after=6 * 3600.0):
    """Keep retrying a request through rate limits, timeouts, dropped connections and server errors, backing off
    up to max_wait seconds between tries, for as long as give_up_after allows. The SDKs' own retries (a handful,
    seconds apart) are not enough for a long experiment: on 2026-09-23 fifteen concurrent GLM agents hit the
    account's rate limit (429, code 1302) and a run died on its first minute. Errors that retrying cannot fix
    (a bad request, a bad key) are raised at once."""
    t0, wait, n = time.time(), 5.0, 0
    while True:
        try:
            return call()
        except Exception as e:                                         # classify by SDK type name, for both SDKs
            name = type(e).__name__
            transient = name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
                                 "OverloadedError", "ServiceUnavailableError") or getattr(e, "status_code", 0) in (408, 409, 429, 500, 502, 503, 504, 529)
            if not transient or time.time() - t0 > give_up_after:
                raise
            n += 1
            sleep = min(max_wait, wait) * random.uniform(0.7, 1.3)
            print(f"[retry] {what}: {name} ({str(e)[:100]}); try {n}, sleeping {sleep:.0f}s", flush=True)
            time.sleep(sleep)
            wait = min(max_wait, wait * 2)


class APIProvider:
    def __init__(self, model, base_url, key_file, *, temperature=None, reasoning_effort=None, extra_body=None,
                 schema_mode="json_schema", token_param="max_tokens", timeout=600.0, max_retries=6):
        from openai import OpenAI
        key = open(os.path.expanduser(key_file)).read().strip()
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.temperature, self.reasoning_effort = temperature, reasoning_effort
        self.extra_body = extra_body or {}
        assert schema_mode in ("json_schema", "json_object", "none"), schema_mode
        self.schema_mode, self.token_param = schema_mode, token_param
        self.usage = {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "seconds": 0.0}
        self.last_reasoning, self.last_tokens, self.last_seconds, self.last_reasoning_tokens = "", 0, 0.0, 0
        self.last_usage = {}

    def complete(self, system, messages, max_tokens=400, schema=None):
        kw = {self.token_param: max_tokens}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        if schema is not None and self.schema_mode == "json_schema":
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "action", "schema": schema, "strict": True}}
        elif schema is not None and self.schema_mode == "json_object":
            kw["response_format"] = {"type": "json_object"}          # the prompt itself says "reply with ONE JSON object"
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        t0 = time.time()
        resp = patient(lambda: self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}] + messages, **kw), self.model)
        dt = time.time() - t0
        u = resp.usage
        inp = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
        out = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
        pdet = getattr(u, "prompt_tokens_details", None) if u else None
        cached = int(getattr(pdet, "cached_tokens", 0) or 0) if pdet else 0
        cached = cached or int(getattr(u, "prompt_cache_hit_tokens", 0) or 0) if u else cached   # DeepSeek reports it here
        det = getattr(u, "completion_tokens_details", None) if u else None
        rtok = int(getattr(det, "reasoning_tokens", 0) or 0) if det else 0
        self.last_usage = {"input_tokens": inp, "cached_tokens": cached, "output_tokens": out, "reasoning_tokens": rtok}
        for k, v in self.last_usage.items():
            self.usage[k] += v
        self.usage["calls"] += 1; self.usage["seconds"] += dt
        msg = resp.choices[0].message
        self.last_reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        self.last_tokens, self.last_seconds, self.last_reasoning_tokens = out, round(dt, 2), rtok
        return msg.content or ""
