"""Thin LLM wrapper so the rest of the code never cares which provider is live.

Supports:
  * Groq   – openai/gpt-oss-120b (default), gpt-oss-20b, groq/compound-mini
  * Gemini – gemini-2.5-flash (default), 2.0-flash, 2.5-flash-lite

Model availability on Groq's free tier changes often (the Llama chat models
moved to enterprise-only in 2026). So instead of dying on a 404, this client
walks a fallback list, sticks to the first model that answers, and tells you
what it switched to. Rate limits get backoff; 404s and bad keys fail fast.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from config import SETTINGS
from utils import log, warn

# Ordered fallbacks. First entry that the key can actually reach wins.
GROQ_FALLBACKS = [
    "openai/gpt-oss-120b",   # free tier, best quality of the OSS pair
    "openai/gpt-oss-20b",    # faster, cheaper, still fine for formatting
    "groq/compound-mini",    # last resort system model
]
GEMINI_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]


class LLMError(RuntimeError):
    pass


class NoModelAvailable(LLMError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


def classify_error(exc: Exception) -> str:
    """Bucket an SDK exception: 'model' | 'auth' | 'rate' | 'params' | 'transient'."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc).lower()

    if status == 404 or "model_not_found" in text or "does not exist or you do not have access" in text:
        return "model"
    if status in (401, 403) or "invalid api key" in text or "unauthorized" in text or "permission_denied" in text:
        return "auth"
    if status == 429 or "rate limit" in text or "quota" in text or "resource_exhausted" in text:
        return "rate"
    if status == 400 and ("unsupported" in text or "unrecognized" in text or "invalid_request" in text
                          or "not supported" in text or "unknown field" in text):
        return "params"
    return "transient"


class LLMClient:
    """One object, two backends, one `complete()` method."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or SETTINGS.resolve_provider()).lower()
        if self.provider == "groq":
            self.model = model or SETTINGS.groq_model
            self.fallbacks = GROQ_FALLBACKS
            self._client = self._init_groq()
        elif self.provider == "gemini":
            self.model = model or SETTINGS.gemini_model
            self.fallbacks = GEMINI_FALLBACKS
            self._client = self._init_gemini()
        else:
            raise LLMError(f"Unknown LLM provider: {self.provider!r}")

        self._pinned = model is not None  # explicit --model means "don't wander"
        self._dead: set[str] = set()      # models this key can't reach
        self._no_json = False             # provider rejected response_format
        self._no_reasoning = False        # provider rejected reasoning_effort
        self._last_call = 0.0
        # Free tier is 30 req/min -> keep ~2.1s between calls to stay polite.
        self._min_interval = float(os.getenv("LLM_MIN_INTERVAL", "2.1"))

    # ------------------------------------------------------------------
    # provider init
    # ------------------------------------------------------------------
    def _init_groq(self):
        if not SETTINGS.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set (add it to .env).")
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise LLMError("pip install groq") from exc
        return Groq(api_key=SETTINGS.groq_api_key)

    def _init_gemini(self):
        if not SETTINGS.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set (add it to .env).")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError("pip install google-genai") from exc
        return genai.Client(api_key=SETTINGS.gemini_api_key)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def available_models(self) -> list[str]:
        """Ask the provider what this key can actually use."""
        if self.provider == "groq":
            return sorted(m.id for m in self._client.models.list().data)
        return sorted(
            m.name.replace("models/", "")
            for m in self._client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])
        )

    def candidates(self) -> list[str]:
        if self._pinned:
            return [self.model]
        ordered = [self.model] + [m for m in self.fallbacks if m != self.model]
        return [m for m in ordered if m not in self._dead]

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        temperature = SETTINGS.temperature if temperature is None else temperature
        candidates = self.candidates()
        if not candidates:
            raise NoModelAvailable(self._no_model_message())

        last_error: Exception | None = None
        for model in candidates:
            for attempt in range(1, SETTINGS.max_retries + 1):
                self._throttle()
                try:
                    text = self._dispatch(model, prompt, system, json_mode, temperature, max_tokens)
                    if not text or not text.strip():
                        raise LLMError("model returned an empty response")
                    if model != self.model:
                        warn(f"switched to {model} (previous model unavailable on this key)")
                        self.model = model
                    return LLMResponse(text=text.strip(), provider=self.provider, model=model)

                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    kind = classify_error(exc)

                    if kind == "auth":
                        raise LLMError(
                            f"{self.provider.upper()} rejected your API key. Check the key in .env "
                            f"(get a new one at {'https://console.groq.com/keys' if self.provider == 'groq' else 'https://aistudio.google.com/apikey'})."
                        ) from exc

                    if kind == "model":
                        self._dead.add(model)
                        warn(f"'{model}' is not available on your key — trying the next model")
                        break  # next candidate, no point retrying

                    if kind == "params":
                        # Drop the optional knobs and retry the same model once.
                        if json_mode and not self._no_json:
                            self._no_json = True
                            warn(f"'{model}' rejected JSON mode — falling back to prompt-enforced JSON")
                            continue
                        if not self._no_reasoning:
                            self._no_reasoning = True
                            continue
                        self._dead.add(model)
                        break

                    if attempt == SETTINGS.max_retries:
                        break
                    delay = self._backoff(kind, attempt)
                    warn(f"{type(exc).__name__}: {str(exc)[:160]} — retry {attempt}/{SETTINGS.max_retries - 1} in {delay:.1f}s")
                    time.sleep(delay)

        if last_error is not None and classify_error(last_error) == "model":
            raise NoModelAvailable(self._no_model_message()) from last_error
        raise LLMError(f"LLM call failed ({self.provider}): {last_error}")

    def _no_model_message(self) -> str:
        tried = ", ".join(sorted(self._dead)) or self.model
        listing = "python main.py --list-models"
        if self.provider == "groq":
            hint = (
                "Groq moved the Llama chat models to enterprise-only in 2026. "
                "Free-tier text models are openai/gpt-oss-120b and openai/gpt-oss-20b."
            )
        else:
            hint = "Try gemini-2.5-flash or gemini-2.0-flash."
        return f"None of these models are available on your key: {tried}. {hint} Run `{listing}` to see what is."

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if self._last_call and elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    @staticmethod
    def _backoff(kind: str, attempt: int) -> float:
        base = 12.0 if kind == "rate" else 2.0
        return base * attempt + random.uniform(0, 1.5)

    def _dispatch(self, model, prompt, system, json_mode, temperature, max_tokens) -> str:
        if self.provider == "groq":
            return self._call_groq(model, prompt, system, json_mode, temperature, max_tokens)
        return self._call_gemini(model, prompt, system, json_mode, temperature, max_tokens)

    def _call_groq(self, model, prompt, system, json_mode, temperature, max_tokens) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if json_mode and self._no_json:
            prompt += "\n\nRespond with valid JSON only. No prose, no code fences."
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_mode and not self._no_json:
            kwargs["response_format"] = {"type": "json_object"}
        # gpt-oss models are reasoning models: keep the thinking short and hidden,
        # otherwise they burn the token budget before writing anything.
        if "gpt-oss" in model and not self._no_reasoning:
            kwargs["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low")

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _call_gemini(self, model, prompt, system, json_mode, temperature, max_tokens) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system or None,
            response_mime_type="application/json" if (json_mode and not self._no_json) else "text/plain",
        )
        if json_mode and self._no_json:
            prompt += "\n\nRespond with valid JSON only. No prose, no code fences."
        resp = self._client.models.generate_content(model=model, contents=prompt, config=cfg)
        return resp.text


_DEFAULT: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide singleton so throttling actually works across modules."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMClient()
        log(f"  · LLM: {_DEFAULT.provider} / {_DEFAULT.model}")
    return _DEFAULT


def set_client(client) -> None:
    """Inject a client (used by tests / offline mode)."""
    global _DEFAULT
    _DEFAULT = client