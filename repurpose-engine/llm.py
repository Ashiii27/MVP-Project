"""Thin LLM wrapper so the rest of the code never cares which provider is live.

Supports:
  * Groq  – llama-3.3-70b-versatile (default)
  * Gemini – gemini-2.0-flash

Handles retries, rate limits (Groq free tier = 30 req/min), and JSON mode.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from config import SETTINGS
from utils import log, warn


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMClient:
    """One object, two backends, one `complete()` method."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or SETTINGS.resolve_provider()).lower()
        if self.provider == "groq":
            self.model = model or SETTINGS.groq_model
            self._client = self._init_groq()
        elif self.provider == "gemini":
            self.model = model or SETTINGS.gemini_model
            self._client = self._init_gemini()
        else:
            raise LLMError(f"Unknown LLM provider: {self.provider!r}")
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
        self._throttle()

        last_error: Exception | None = None
        for attempt in range(1, SETTINGS.max_retries + 1):
            try:
                if self.provider == "groq":
                    text = self._call_groq(prompt, system, json_mode, temperature, max_tokens)
                else:
                    text = self._call_gemini(prompt, system, json_mode, temperature, max_tokens)
                if not text or not text.strip():
                    raise LLMError("model returned an empty response")
                return LLMResponse(text=text.strip(), provider=self.provider, model=self.model)
            except Exception as exc:  # noqa: BLE001 - we retry on anything transient
                last_error = exc
                if attempt == SETTINGS.max_retries:
                    break
                delay = self._backoff(exc, attempt)
                warn(f"{type(exc).__name__}: {exc} — retry {attempt}/{SETTINGS.max_retries - 1} in {delay:.1f}s")
                time.sleep(delay)

        raise LLMError(f"LLM call failed after {SETTINGS.max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if self._last_call and elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    @staticmethod
    def _backoff(exc: Exception, attempt: int) -> float:
        text = str(exc).lower()
        base = 12.0 if ("rate" in text or "429" in text or "quota" in text) else 2.0
        return base * attempt + random.uniform(0, 1.5)

    def _call_groq(self, prompt, system, json_mode, temperature, max_tokens) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _call_gemini(self, prompt, system, json_mode, temperature, max_tokens) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system or None,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = self._client.models.generate_content(
            model=self.model, contents=prompt, config=cfg
        )
        return resp.text


_DEFAULT: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide singleton so throttling actually works across modules."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMClient()
        log(f"  · LLM: {_DEFAULT.provider} / {_DEFAULT.model}")
    return _DEFAULT


def set_client(client: LLMClient) -> None:
    """Inject a client (used by tests / offline mode)."""
    global _DEFAULT
    _DEFAULT = client
