"""formatter.py — one idea in, three pieces of publishable text out.

Three separate LLM calls (Twitter / LinkedIn / Email), each with its own
prompt template. Every output is validated against the rules in its prompt
(length, banned phrases, formatting) and gets one automatic repair pass if it
fails. That repair loop is what keeps the output from drifting into AI slop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from chunker import Idea, excerpt_for, load_prompt
from config import SETTINGS
from llm import get_client
from transcriber import Transcript
from utils import extract_json, log, ok, step, warn, word_count

BANNED = [
    "delve", "unlock", "game-changer", "game changer", "leverage", "harness",
    "in today's fast-paced world", "let that sink in", "supercharge",
    "revolutionize", "revolutionise", "myriad", "testament to",
    "navigate the landscape", "without further ado", "buckle up",
    "hope this finds you well", "circle back", "moving the needle",
    "at the end of the day", "i wanted to reach out", "dive in", "synergy",
    "in this thread", "a thread 🧵", "smash the like",
]

EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002700-\U000027BF" "\U00002600-\U000026FF"
    "\U0001F1E6-\U0001F1FF" "\u2190-\u21FF" "\u2B00-\u2BFF" "]",
    flags=re.UNICODE,
)


@dataclass
class Outputs:
    idea_title: str
    twitter: str = ""
    linkedin: str = ""
    email: str = ""
    issues: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# validators
# ----------------------------------------------------------------------
def _banned_hits(text: str) -> list[str]:
    low = text.lower()
    return [p for p in BANNED if p in low]


def validate_twitter(data: dict) -> list[str]:
    issues: list[str] = []
    tweets = data.get("tweets") or []
    if not isinstance(tweets, list) or len(tweets) != 6:
        issues.append(f"Return exactly 6 tweets (got {len(tweets) if isinstance(tweets, list) else 0}).")
    for i, t in enumerate(tweets if isinstance(tweets, list) else [], 1):
        t = str(t)
        if len(t) > 275:
            issues.append(f"Tweet {i} is {len(t)} chars; cut it under 270.")
        if "#" in t:
            issues.append(f"Tweet {i} contains a hashtag; remove it.")
        if EMOJI_RE.search(t):
            issues.append(f"Tweet {i} contains an emoji; remove it.")
        if "—" in t:
            issues.append(f"Tweet {i} uses an em-dash; rewrite without it.")
    if tweets and len(str(tweets[0])) < 40:
        issues.append("Tweet 1 is too thin to stop a scroll; make the hook a specific claim.")
    hits = _banned_hits(" ".join(str(t) for t in tweets))
    if hits:
        issues.append(f"Banned phrases used: {', '.join(sorted(set(hits)))}. Rewrite those lines.")
    return issues


def validate_linkedin(data: dict) -> list[str]:
    issues: list[str] = []
    post = str(data.get("post") or "")
    wc = word_count(post)
    if not post.strip():
        return ["The post is empty."]
    if not 140 <= wc <= 215:
        issues.append(f"Post is {wc} words; it must be 150-200.")
    lines = [l for l in post.split("\n") if l.strip()]
    if len(lines) < 6:
        issues.append("Put a line break between every sentence (at least 6 lines).")
    long_lines = [l for l in lines if len(re.findall(r"[.!?]", l)) > 1 and word_count(l) > 25]
    if long_lines:
        issues.append("Some lines hold multiple sentences; split them onto their own lines.")
    if "#" in post:
        issues.append("Remove all hashtags.")
    if "—" in post:
        issues.append("Remove em-dashes.")
    emojis = EMOJI_RE.findall(post)
    if len(emojis) > 1:
        issues.append(f"Use exactly one emoji at the very end (found {len(emojis)}).")
    stripped = post.rstrip()
    if not (stripped.endswith("?") or (emojis and "?" in stripped[-40:])):
        issues.append("End with a real question, then one emoji after it.")
    hits = _banned_hits(post)
    if hits:
        issues.append(f"Banned phrases used: {', '.join(sorted(set(hits)))}. Rewrite those lines.")
    return issues


def validate_email(data: dict, video_url: str = "") -> list[str]:
    issues: list[str] = []
    subject = str(data.get("subject") or "")
    body = str(data.get("body") or "")
    if not subject.strip():
        issues.append("Missing subject line.")
    elif len(subject) > 60:
        issues.append(f"Subject is {len(subject)} chars; keep it under 55.")
    wc = word_count(body)
    if not 210 <= wc <= 300:
        issues.append(f"Body is {wc} words; target 230-280.")
    if video_url and video_url.split("?")[0] not in body and video_url not in body:
        issues.append("The CTA must include the full video URL.")
    if EMOJI_RE.search(body):
        issues.append("Remove emojis from the email.")
    if "—" in body:
        issues.append("Remove em-dashes.")
    if "[your name]" in body.lower() or "[name]" in body.lower():
        issues.append("Remove placeholder brackets like [Your Name].")
    hits = _banned_hits(subject + " " + body)
    if hits:
        issues.append(f"Banned phrases used: {', '.join(sorted(set(hits)))}. Rewrite those lines.")
    return issues


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------
def _generate(prompt: str, validator, *, label: str, max_tokens: int = 2048) -> tuple[dict, list[str]]:
    """One generation call plus, if needed, one targeted repair call."""
    client = get_client()
    resp = client.complete(prompt, json_mode=True, temperature=SETTINGS.temperature, max_tokens=max_tokens)
    try:
        data = extract_json(resp.text)
    except ValueError as exc:
        warn(f"{label}: {exc}")
        data = {}
    if not isinstance(data, dict):
        data = {"tweets": data} if isinstance(data, list) else {}

    issues = validator(data)
    if not issues:
        return data, []

    log(f"  · {label}: fixing {len(issues)} issue(s)")
    repair = (
        f"{prompt}\n\n---\nYou already produced this draft:\n\n{resp.text}\n\n"
        "It breaks these rules:\n- " + "\n- ".join(issues) +
        "\n\nRewrite it so every rule is satisfied. Keep what already works, "
        "keep the same specifics from the source, do not pad with filler to hit "
        "a word count. Return ONLY the same JSON shape."
    )
    resp2 = client.complete(repair, json_mode=True, temperature=0.5, max_tokens=max_tokens)
    try:
        data2 = extract_json(resp2.text)
    except ValueError:
        return data, issues
    if not isinstance(data2, dict):
        return data, issues
    issues2 = validator(data2)
    if len(issues2) <= len(issues):
        return data2, issues2
    return data, issues


def _render_twitter(data: dict) -> str:
    tweets = [str(t).strip() for t in (data.get("tweets") or []) if str(t).strip()]
    return "\n\n".join(f"{i}/{len(tweets)}\n{t}" for i, t in enumerate(tweets, 1))


def _render_email(data: dict) -> str:
    subject = str(data.get("subject") or "").strip()
    preview = str(data.get("preview_text") or "").strip()
    body = str(data.get("body") or "").strip()
    head = f"Subject: {subject}"
    if preview:
        head += f"\nPreview: {preview}"
    return f"{head}\n\n{body}"


def format_idea(
    idea: Idea,
    transcript: Transcript,
    meta,
    *,
    creator: str | None = None,
    tone: str | None = None,
) -> Outputs:
    """Run the three formatting prompts for a single idea."""
    creator = creator or SETTINGS.creator_name or getattr(meta, "channel", "") or "the creator"
    tone = tone or SETTINGS.tone
    excerpt = excerpt_for(transcript, idea)

    common = {
        "creator": creator,
        "tone": tone,
        "title": idea.title,
        "hook": idea.hook,
        "summary": idea.summary,
        "quote": idea.quote or "(none)",
        "excerpt": excerpt,
        "video_title": getattr(meta, "title", ""),
        "video_url": getattr(meta, "url", ""),
        "start_time": idea.start_time,
    }

    step(f"Formatting: {idea.title}")
    out = Outputs(idea_title=idea.title)

    tw_data, tw_issues = _generate(
        load_prompt("twitter.txt").format(**common), validate_twitter, label="twitter"
    )
    out.twitter = _render_twitter(tw_data)
    out.issues["twitter"] = tw_issues

    li_data, li_issues = _generate(
        load_prompt("linkedin.txt").format(**common), validate_linkedin, label="linkedin"
    )
    out.linkedin = str(li_data.get("post") or "").strip()
    out.issues["linkedin"] = li_issues

    em_data, em_issues = _generate(
        load_prompt("email.txt").format(**common),
        lambda d: validate_email(d, common["video_url"]),
        label="email",
        max_tokens=2048,
    )
    out.email = _render_email(em_data)
    out.issues["email"] = em_issues

    remaining = sum(len(v) for v in out.issues.values())
    if remaining:
        warn(f"{remaining} soft rule violation(s) left after repair (see report).")
    else:
        ok("All three formats pass the style checks")
    return out