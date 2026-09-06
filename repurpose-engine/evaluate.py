"""evaluate.py — a slop detector for Task 1.7.

Testing 5 videos by vibes is slow and inconsistent. This scores every piece
of generated content on the things that actually separate "I'd post this"
from "generic AI slop", then prints a table so you can see which prompt
change helped.

    python evaluate.py                      # score everything in output/
    python evaluate.py output/some-video    # score one run

Metrics per piece (0-100 overall):
  specificity   numbers, names and concrete nouns lifted from the source
  concision     average sentence length (short = human, long = essay-bot)
  freshness     absence of banned/AI-tell phrases and cliché openers
  rules         the format's own hard rules (length, hashtags, emojis, CTA)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from config import OUTPUT_DIR
from formatter import BANNED, EMOJI_RE, validate_email, validate_linkedin, validate_twitter
from utils import word_count

AI_TELLS = [
    "in a world where", "it's important to note", "as we all know", "the key takeaway",
    "in conclusion", "furthermore", "moreover", "additionally,", "this begs the question",
    "when it comes to", "one of the most", "in essence", "ultimately,", "that being said",
    "not only", "but also", "elevate", "empower", "seamless", "robust", "cutting-edge",
    "transformative", "actionable insights", "key insights", "unpack", "landscape of",
]

CLICHE_OPENERS = [
    "have you ever", "imagine", "what if i told you", "let me tell you", "picture this",
    "here's the thing", "we've all been there", "in today's",
]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def score_text(text: str, *, source: str = "") -> dict:
    words = word_count(text)
    sents = sentences(text)
    avg_len = words / max(1, len(sents))
    low = text.lower()

    banned_hits = [p for p in BANNED if p in low]
    tell_hits = [p for p in AI_TELLS if p in low]
    opener = sentences(text)[0].lower() if sents else ""
    cliche = [c for c in CLICHE_OPENERS if opener.startswith(c)]

    numbers = len(re.findall(r"\b\d[\d,.]*\s?(?:%|percent|dollars?|x|k|m|hours?|days?|weeks?|months?|years?)?\b", text))
    proper = len({w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text)})
    # Did it actually reuse the speaker's words? (5-gram overlap with source)
    overlap = 0.0
    if source:
        src_grams = _ngrams(source.lower(), 5)
        out_grams = _ngrams(low, 5)
        if out_grams:
            overlap = len(src_grams & out_grams) / len(out_grams)

    specificity = min(100, (numbers * 9) + (proper * 4) + int(overlap * 120))
    concision = max(0, 100 - max(0, avg_len - 14) * 7)
    freshness = max(0, 100 - len(banned_hits) * 25 - len(tell_hits) * 12 - len(cliche) * 20)

    return {
        "words": words,
        "avg_sentence": round(avg_len, 1),
        "numbers": numbers,
        "source_overlap": round(overlap, 3),
        "banned": banned_hits,
        "ai_tells": tell_hits,
        "cliche_opener": cliche,
        "specificity": specificity,
        "concision": round(concision),
        "freshness": freshness,
    }


def _ngrams(text: str, n: int) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text)
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def _rule_score(kind: str, text: str, video_url: str) -> tuple[int, list[str]]:
    if kind == "twitter":
        tweets = [t.strip() for t in re.split(r"\n?\d+/\d+\n", text) if t.strip()]
        issues = validate_twitter({"tweets": tweets})
    elif kind == "linkedin":
        issues = validate_linkedin({"post": text})
    else:
        subject = ""
        body = text
        m = re.match(r"Subject:\s*(.+)", text)
        if m:
            subject = m.group(1).strip()
            body = text.split("\n\n", 1)[-1]
        issues = validate_email({"subject": subject, "body": body}, video_url)
    return max(0, 100 - 20 * len(issues)), issues


def evaluate_run(run_dir: Path) -> list[dict]:
    source = ""
    transcript = run_dir / "transcript.txt"
    if transcript.exists():
        source = transcript.read_text(encoding="utf-8")
    report = run_dir / "report.md"
    video_url = ""
    if report.exists():
        m = re.search(r"- Source: (\S+)", report.read_text(encoding="utf-8"))
        if m and m.group(1) != "n/a":
            video_url = m.group(1)

    rows = []
    for folder in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for kind in ("twitter", "linkedin", "email"):
            path = folder / f"{kind}.md"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8").strip()
            s = score_text(text, source=source)
            rule, issues = _rule_score(kind, text, video_url)
            overall = round(0.35 * s["specificity"] + 0.2 * s["concision"] + 0.25 * s["freshness"] + 0.2 * rule)
            rows.append({
                "run": run_dir.name,
                "idea": folder.name,
                "format": kind,
                "overall": overall,
                "rules": rule,
                **s,
                "rule_issues": issues,
            })
    return rows


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv] or [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")]
    rows: list[dict] = []
    for t in targets:
        if t.is_dir():
            rows.extend(evaluate_run(t))
    if not rows:
        print("Nothing to evaluate. Run main.py first.")
        return 1

    print(f"\n{'format':10}{'overall':>8}{'spec':>6}{'concis':>8}{'fresh':>7}{'rules':>7}{'words':>7}  idea")
    print("─" * 100)
    for r in rows:
        print(f"{r['format']:10}{r['overall']:>8}{r['specificity']:>6}{r['concision']:>8}"
              f"{r['freshness']:>7}{r['rules']:>7}{r['words']:>7}  {r['idea'][:38]}")

    print("\nWeak spots")
    print("─" * 100)
    flagged = 0
    for r in rows:
        problems = []
        if r["banned"]:
            problems.append("banned: " + ", ".join(r["banned"]))
        if r["ai_tells"]:
            problems.append("ai-tells: " + ", ".join(r["ai_tells"][:4]))
        if r["cliche_opener"]:
            problems.append("cliché opener")
        if r["specificity"] < 40:
            problems.append(f"thin on specifics (numbers={r['numbers']}, overlap={r['source_overlap']})")
        if r["rule_issues"]:
            problems.append("rules: " + "; ".join(r["rule_issues"][:3]))
        if problems:
            flagged += 1
            print(f"  {r['idea'][:30]:32}{r['format']:10}{' | '.join(problems)[:110]}")
    if not flagged:
        print("  none — every piece passed the checks")

    avg = sum(r["overall"] for r in rows) / len(rows)
    print(f"\nAverage score: {avg:.0f}/100 across {len(rows)} pieces")
    print("  ≥75  GO. Ship it, move to Phase 2.")
    print("  60-74 borderline: tighten the prompt rules that keep failing above.")
    print("  <60  ITERATE on prompts before building anything else.\n")

    (OUTPUT_DIR / "eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Detail written to {OUTPUT_DIR / 'eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
