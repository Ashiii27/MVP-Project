#!/usr/bin/env python3
"""main.py — the orchestrator.

    python main.py "https://youtube.com/watch?v=xyz"

Downloads audio -> transcribes -> extracts standalone ideas -> writes a
Twitter thread, a LinkedIn post and an email for the top ideas, prints them
to the terminal, and saves everything under output/.

Useful flags:
    --top-n 3            how many ideas to format (default 3)
    --creator "Name"     whose voice to ghostwrite in
    --tone authoritative casual | authoritative | irreverent ...
    --provider gemini    force groq | gemini
    --transcript FILE    skip download+transcription, use a .txt/.json transcript
    --audio FILE         skip download, transcribe a local audio file
    --no-cache           ignore cached transcript/ideas
    --mock               run the whole pipeline with a fake LLM (no API key)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import OUTPUT_DIR, SETTINGS
from utils import err, hhmmss, log, ok, slugify, step


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="repurpose",
        description="Turn one YouTube video into a Twitter thread, a LinkedIn post and an email.",
    )
    p.add_argument("url", nargs="?", help="YouTube video URL")
    p.add_argument("--top-n", type=int, default=SETTINGS.top_n, help="ideas to format (default 3)")
    p.add_argument("--creator", default=SETTINGS.creator_name, help="creator name for the ghostwriter voice")
    p.add_argument("--tone", default=SETTINGS.tone, help="casual | authoritative | irreverent | warm")
    p.add_argument("--provider", choices=["groq", "gemini"], help="force an LLM provider")
    p.add_argument("--model", help="override the model name")
    p.add_argument("--transcript", help="use an existing transcript (.txt or .json) and skip audio")
    p.add_argument("--audio", help="use a local audio file and skip the download")
    p.add_argument("--title", default="", help="video title when using --transcript")
    p.add_argument("--no-cache", action="store_true", help="ignore cached transcript/ideas")
    p.add_argument("--keep-audio", action="store_true", help="don't delete the downloaded audio")
    p.add_argument("--quiet", action="store_true", help="less logging")
    p.add_argument("--mock", action="store_true", help="run offline with a fake LLM (no API key)")
    return p.parse_args(argv)


def _load_transcript_file(path: Path):
    from transcriber import Transcript

    if path.suffix.lower() == ".json":
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return Transcript(text=path.read_text(encoding="utf-8").strip(), segments=[], duration=0.0)


class _Meta:
    """Stand-in for VideoMeta when there's no YouTube download."""

    def __init__(self, title="", channel="", url="", video_id="", duration=0):
        self.title, self.channel, self.url = title, channel, url
        self.video_id, self.duration = video_id, duration


def run(args: argparse.Namespace) -> int:
    if args.quiet:
        SETTINGS.verbose = False
    SETTINGS.top_n = max(1, args.top_n)
    SETTINGS.creator_name = args.creator
    SETTINGS.tone = args.tone
    SETTINGS.keep_audio = SETTINGS.keep_audio or args.keep_audio
    if args.provider:
        SETTINGS.llm_provider = args.provider

    if not args.url and not args.transcript and not args.audio:
        err("Give me a YouTube URL (or --transcript / --audio).")
        return 2

    # Wire up the LLM (real or mock) before anything imports it.
    import llm

    if args.mock:
        from mock_llm import MockLLM

        llm.set_client(MockLLM())
        log("  · LLM: mock (offline)")
    else:
        try:
            llm.set_client(llm.LLMClient(provider=args.provider, model=args.model))
            log(f"  · LLM: {llm.get_client().provider} / {llm.get_client().model}")
        except Exception as exc:  # noqa: BLE001
            err(str(exc))
            return 2

    from chunker import chunk_transcript
    from formatter import format_idea
    from transcriber import transcribe

    started = datetime.now(timezone.utc)

    # ---------------- 1. audio + metadata ----------------
    if args.transcript:
        meta = _Meta(title=args.title or Path(args.transcript).stem, url=args.url or "")
        transcript = _load_transcript_file(Path(args.transcript))
        ok(f"Loaded transcript: {len(transcript.text.split())} words")
        slug = slugify(meta.title)
    elif args.audio:
        meta = _Meta(title=args.title or Path(args.audio).stem, url=args.url or "")
        slug = slugify(meta.title)
        cache = OUTPUT_DIR / slug / "transcript.json"
        transcript = transcribe(args.audio, cache_path=None if args.no_cache else cache)
    else:
        from downloader import download_audio

        meta = download_audio(args.url)
        slug = f"{slugify(meta.title)}-{meta.video_id}"
        cache = OUTPUT_DIR / slug / "transcript.json"
        transcript = transcribe(
            meta.audio_path,
            cache_path=None if args.no_cache else cache,
            context_prompt=f"{meta.title}. {meta.channel}.",
        )
        if not SETTINGS.keep_audio:
            Path(meta.audio_path).unlink(missing_ok=True)

    if not transcript.text.strip():
        err("Empty transcript — nothing to work with.")
        return 1

    run_dir = OUTPUT_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "transcript.txt").write_text(transcript.text, encoding="utf-8")
    if transcript.segments:
        (run_dir / "transcript.json").write_text(
            json.dumps(transcript.as_dict(), indent=2), encoding="utf-8"
        )

    # ---------------- 2. ideas ----------------
    ideas = chunk_transcript(
        transcript,
        title=meta.title,
        creator=args.creator or getattr(meta, "channel", ""),
        cache_path=None if args.no_cache else run_dir / "ideas.json",
    )

    # ---------------- 3. formats ----------------
    results = []
    for idea in ideas[: SETTINGS.top_n]:
        results.append((idea, format_idea(idea, transcript, meta, creator=args.creator, tone=args.tone)))

    # ---------------- 4. save + print ----------------
    written = write_outputs(run_dir, meta, ideas, results, started)
    print_outputs(meta, results)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    step("Done")
    ok(f"{len(results)} ideas x 3 formats in {elapsed:.0f}s")
    ok(f"Saved to {run_dir}")
    for path in written:
        log(f"    {path.relative_to(OUTPUT_DIR.parent)}")
    return 0


def write_outputs(run_dir: Path, meta, ideas, results, started) -> list[Path]:
    written: list[Path] = []
    (run_dir / "ideas.json").write_text(
        json.dumps([i.as_dict() for i in ideas], indent=2), encoding="utf-8"
    )
    written.append(run_dir / "ideas.json")

    lines = [
        f"# {meta.title or 'Repurposed video'}",
        "",
        f"- Source: {meta.url or 'n/a'}",
        f"- Creator: {getattr(meta, 'channel', '') or SETTINGS.creator_name or 'n/a'}",
        f"- Generated: {started.strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Ideas found: {len(ideas)} (formatted top {len(results)})",
        "",
    ]

    for n, (idea, out) in enumerate(results, 1):
        folder = run_dir / f"{n:02d}-{slugify(idea.title, 40)}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "twitter.md").write_text(out.twitter + "\n", encoding="utf-8")
        (folder / "linkedin.md").write_text(out.linkedin + "\n", encoding="utf-8")
        (folder / "email.md").write_text(out.email + "\n", encoding="utf-8")
        written += [folder / "twitter.md", folder / "linkedin.md", folder / "email.md"]

        flags = [f"{k}: {'; '.join(v)}" for k, v in out.issues.items() if v]
        lines += [
            f"## {n}. {idea.title}",
            "",
            f"*{idea.start_time}–{idea.end_time} · score {idea.score}/10*",
            "",
            f"**Hook.** {idea.hook}",
            "",
            f"{idea.summary}",
            "",
            "### Twitter thread", "", out.twitter, "",
            "### LinkedIn post", "", out.linkedin, "",
            "### Email", "", "```", out.email, "```", "",
        ]
        if flags:
            lines += ["> Style checks still failing: " + " | ".join(flags), ""]

    if len(ideas) > len(results):
        lines += ["## Ideas not yet formatted", ""]
        lines += [
            f"- [{i.start_time}] **{i.title}** ({i.score}/10) — {i.hook}" for i in ideas[len(results):]
        ]
        lines.append("")

    report = run_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    written.append(report)
    return written


def print_outputs(meta, results) -> None:
    bar = "═" * 72
    print(f"\n{bar}\n  {meta.title or 'Repurposed video'}\n{bar}")
    for n, (idea, out) in enumerate(results, 1):
        print(f"\n\n### IDEA {n}: {idea.title}  [{idea.start_time}–{idea.end_time}]")
        print(f"    {idea.hook}\n")
        print("─" * 72 + "\n  TWITTER THREAD\n" + "─" * 72)
        print(out.twitter)
        print("\n" + "─" * 72 + "\n  LINKEDIN POST\n" + "─" * 72)
        print(out.linkedin)
        print("\n" + "─" * 72 + "\n  EMAIL\n" + "─" * 72)
        print(out.email)
    print()


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        err("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001
        err(f"{type(exc).__name__}: {exc}")
        if os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes"}:
            import traceback

            traceback.print_exc()
        else:
            log("  (set DEBUG=1 for the full traceback)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
