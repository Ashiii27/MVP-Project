"""Small shared helpers: logging, slugs, timestamps, JSON salvage."""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from contextlib import contextmanager

from config import SETTINGS

_STEP_COLORS = {
    "step": "\033[1;36m",
    "ok": "\033[1;32m",
    "warn": "\033[1;33m",
    "err": "\033[1;31m",
    "dim": "\033[2m",
}
_RESET = "\033[0m"


def _color(kind: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{_STEP_COLORS.get(kind, '')}{text}{_RESET}"


def log(msg: str, kind: str = "dim") -> None:
    if SETTINGS.verbose:
        print(_color(kind, msg), flush=True)


def step(msg: str) -> None:
    log(f"\n▶ {msg}", "step")


def ok(msg: str) -> None:
    log(f"  ✓ {msg}", "ok")


def warn(msg: str) -> None:
    print(_color("warn", f"  ! {msg}"), flush=True)


def err(msg: str) -> None:
    print(_color("err", f"  ✗ {msg}"), file=sys.stderr, flush=True)


@contextmanager
def timed(label: str):
    start = time.time()
    yield
    log(f"  ⏱ {label} in {time.time() - start:.1f}s")


def slugify(value: str, max_len: int = 60) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[-\s]+", "-", value)
    return value[:max_len].strip("-") or "untitled"


def hhmmss(seconds: float | int | None) -> str:
    """Format seconds as HH:MM:SS (or MM:SS for short clips)."""
    if seconds is None:
        return "00:00"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_timestamp(value) -> float:
    """Parse '01:02:03' / '02:03' / 123 / '123.4' into seconds."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return 0.0
        total = 0.0
        for n in nums:
            total = total * 60 + n
        return total
    try:
        return float(text)
    except ValueError:
        return 0.0


def extract_json(raw: str):
    """Pull a JSON object/array out of an LLM response.

    Handles ```json fences, leading chatter, and trailing commas.
    Raises ValueError if nothing parseable is found.
    """
    if raw is None:
        raise ValueError("empty LLM response")
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the widest bracketed span and try to repair it.
    starts = [i for i, c in enumerate(text) if c in "[{"]
    ends = [i for i, c in enumerate(text) if c in "]}"]
    if starts and ends:
        candidate = text[starts[0] : ends[-1] + 1]
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"could not parse JSON from model output:\n{raw[:500]}")


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text or ""))
