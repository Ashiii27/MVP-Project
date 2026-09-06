"""chunker.py — full transcript in, 5–7 standalone ideas out.

Long transcripts are chunked by token budget, mined separately, then merged
and re-ranked, because a 90-minute podcast does not fit in one context window
comfortably (and quality drops well before the hard limit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from config import PROMPTS_DIR, SETTINGS
from llm import get_client
from transcriber import Transcript
from utils import extract_json, hhmmss, log, ok, parse_timestamp, step, warn

# Rough budget: ~4 chars per token. 40k chars ≈ 10k tokens of transcript per pass.
MAX_CHARS_PER_PASS = 40_000


@dataclass
class Idea:
    title: str
    summary: str
    start_time: str
    end_time: str
    hook: str
    quote: str = ""
    score: int = 5

    @property
    def start_seconds(self) -> float:
        return parse_timestamp(self.start_time)

    @property
    def end_seconds(self) -> float:
        return parse_timestamp(self.end_time)

    def as_dict(self) -> dict:
        return asdict(self)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _split_for_context(text: str, max_chars: int = MAX_CHARS_PER_PASS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    passes, current, size = [], [], 0
    for line in lines:
        if size + len(line) > max_chars and current:
            passes.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        passes.append("\n".join(current))
    return passes


def _coerce_ideas(payload) -> list[Idea]:
    if isinstance(payload, dict):
        for key in ("ideas", "chunks", "results", "items", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        return []

    ideas: list[Idea] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not title or not summary:
            continue
        try:
            score = int(float(item.get("score", 5)))
        except (TypeError, ValueError):
            score = 5
        ideas.append(
            Idea(
                title=title,
                summary=summary,
                start_time=hhmmss(parse_timestamp(item.get("start_time"))),
                end_time=hhmmss(parse_timestamp(item.get("end_time"))),
                hook=str(item.get("hook") or "").strip(),
                quote=str(item.get("quote") or "").strip(),
                score=max(1, min(10, score)),
            )
        )
    return ideas


def _dedupe(ideas: list[Idea]) -> list[Idea]:
    seen: list[set[str]] = []
    unique: list[Idea] = []
    for idea in ideas:
        words = {w for w in idea.title.lower().split() if len(w) > 3}
        if any(words and len(words & prev) / max(1, len(words)) > 0.6 for prev in seen):
            continue
        seen.append(words)
        unique.append(idea)
    return unique


def chunk_transcript(
    transcript: Transcript,
    *,
    title: str = "",
    creator: str = "",
    cache_path: str | Path | None = None,
) -> list[Idea]:
    """Extract and rank standalone ideas from a transcript."""
    if cache_path and Path(cache_path).exists():
        cached = _coerce_ideas(json.loads(Path(cache_path).read_text()))
        if cached:
            ok(f"Reusing cached ideas ({len(cached)})")
            return cached

    step("Extracting standalone ideas")
    body = transcript.timestamped()
    passes = _split_for_context(body)
    if len(passes) > 1:
        log(f"  · transcript is long — mining it in {len(passes)} passes")

    template = load_prompt("chunker.txt")
    client = get_client()
    per_pass = max(3, SETTINGS.max_ideas if len(passes) == 1 else SETTINGS.max_ideas - 2)

    all_ideas: list[Idea] = []
    for i, part in enumerate(passes, 1):
        prompt = template.format(
            title=title or "(unknown)",
            creator=creator or "the speaker",
            duration=hhmmss(transcript.duration),
            transcript=part,
            min_ideas=min(SETTINGS.min_ideas, per_pass),
            max_ideas=per_pass,
        )
        resp = client.complete(prompt, json_mode=True, temperature=0.4, max_tokens=4096)
        try:
            ideas = _coerce_ideas(extract_json(resp.text))
        except ValueError as exc:
            warn(f"pass {i}: {exc}")
            continue
        log(f"  · pass {i}/{len(passes)}: {len(ideas)} ideas")
        all_ideas.extend(ideas)

    if not all_ideas:
        raise RuntimeError("The model returned no usable ideas. Try a different video or model.")

    all_ideas = _dedupe(all_ideas)
    all_ideas.sort(key=lambda i: (-i.score, i.start_seconds))
    ok(f"{len(all_ideas)} standalone ideas (top score {all_ideas[0].score}/10)")
    for idea in all_ideas[:7]:
        log(f"    {idea.score:>2}/10  [{idea.start_time}] {idea.title}")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(
            json.dumps([i.as_dict() for i in all_ideas], indent=2), encoding="utf-8"
        )
    return all_ideas


def excerpt_for(transcript: Transcript, idea: Idea, max_chars: int = 6000) -> str:
    """Pull the raw transcript text covering an idea's time range.

    The formatter writes from the speaker's real words, not just a summary —
    that's most of the difference between 'human' and 'AI slop'.
    """
    start, end = idea.start_seconds, idea.end_seconds
    if end <= start:
        end = start + 180
    if transcript.segments:
        picked = [s.text for s in transcript.segments if s.end >= start - 15 and s.start <= end + 15]
        text = " ".join(t.strip() for t in picked if t.strip())
        if len(text) > 200:
            return text[:max_chars]
    # No usable timestamps: fall back to a proportional slice of the raw text.
    if transcript.duration > 0 and transcript.text:
        ratio_start = max(0.0, start / transcript.duration)
        n = len(transcript.text)
        begin = int(ratio_start * n)
        return transcript.text[begin : begin + max_chars] or transcript.text[:max_chars]
    return transcript.text[:max_chars]