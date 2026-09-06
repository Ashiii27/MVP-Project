"""mock_llm.py — a deterministic offline stand-in for the LLM.

Used by `python main.py --mock ...` and by the test suite so the whole
pipeline can be exercised (JSON parsing, validation, repair loop, file
writing) without burning API quota or needing keys.

It is NOT a content generator. Output is intentionally plain.
"""

from __future__ import annotations

import json
import re

from llm import LLMResponse


def _sentences(text: str, n: int) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
    return parts[:n] or ["The speaker makes a specific, testable claim about the topic."]


class MockLLM:
    provider = "mock"
    model = "mock-1"

    def complete(self, prompt: str, *, system=None, json_mode=False, temperature=None, max_tokens=4096):
        source = _extract_block(prompt)
        if '"ideas"' in prompt:
            payload = self._ideas(source)
        elif '"tweets"' in prompt:
            payload = self._tweets(source)
        elif '"post"' in prompt:
            payload = self._post(source)
        else:
            payload = self._email(prompt, source)
        return LLMResponse(text=json.dumps(payload), provider=self.provider, model=self.model)

    # -- generators -----------------------------------------------------
    def _ideas(self, source: str) -> dict:
        marks = re.findall(r"\[(\d{2}:\d{2}(?::\d{2})?)\]", source) or ["00:00"]
        sents = _sentences(re.sub(r"\[[\d:]+\]", "", source), 12)
        ideas = []
        for i in range(min(6, max(1, len(sents)))):
            start = marks[min(i * 2, len(marks) - 1)]
            end = marks[min(i * 2 + 2, len(marks) - 1)]
            body = sents[i]
            ideas.append({
                "title": " ".join(body.split()[:6]).rstrip(".,"),
                "summary": " ".join(sents[i : i + 3]),
                "start_time": start if start.count(":") == 2 else f"00:{start}",
                "end_time": end if end.count(":") == 2 else f"00:{end}",
                "hook": body[:110],
                "quote": body,
                "score": 10 - i,
            })
        return {"ideas": ideas}

    def _tweets(self, source: str) -> dict:
        sents = _sentences(source, 6)
        while len(sents) < 6:
            sents.append(f"Point {len(sents) + 1}. The detail matters more than the theory.")
        return {"tweets": [s[:260] for s in sents[:6]]}

    def _post(self, source: str) -> dict:
        sents = _sentences(source, 8)
        lines = [s[:120] for s in sents]
        while sum(len(l.split()) for l in lines) < 155:
            lines.append("The specifics decide whether the idea survives contact with reality.")
        lines.append("What would change your mind about this? 🙂")
        return {"post": "\n".join(lines), "word_count": sum(len(l.split()) for l in lines)}

    def _email(self, prompt: str, source: str) -> dict:
        url_match = re.search(r"VIDEO URL: (\S+)", prompt)
        url = url_match.group(1) if url_match else "https://youtube.com"
        ts = re.search(r"TIMESTAMP WHERE THIS COMES UP: (\S+)", prompt)
        sents = _sentences(source, 10)
        body = "\n\n".join(sents)
        while len(body.split()) < 235:
            body += "\n\nOne more concrete detail from the conversation, kept short and plain."
        body += f"\n\nWatch the full breakdown here: {url} (starts around {ts.group(1) if ts else '00:00'})."
        return {
            "subject": "A plainer way to think about this",
            "preview_text": "One idea, pulled straight from the episode.",
            "body": body,
        }


def _extract_block(prompt: str) -> str:
    blocks = re.findall(r"---\n(.*?)\n---", prompt, re.S)
    return blocks[0] if blocks else prompt