"""batch.py — Task 1.7: run the pipeline over a list of videos, then score them.

    python batch.py videos.txt
    python batch.py videos.txt --top-n 2 --tone authoritative

videos.txt format (one per line, blank lines and # comments ignored):

    https://youtube.com/watch?v=aaa | Lenny Rachitsky | authoritative
    https://youtube.com/watch?v=bbb | Ali Abdaal
    https://youtube.com/watch?v=ccc
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import evaluate
import main as orchestrator
from utils import err, ok, step


def parse_list(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        url = parts[0]
        creator = parts[1] if len(parts) > 1 else ""
        tone = parts[2] if len(parts) > 2 else ""
        rows.append((url, creator, tone))
    return rows


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the repurpose pipeline over a list of videos.")
    ap.add_argument("list_file", help="text file with one YouTube URL per line")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--tone", default="casual")
    ap.add_argument("--provider", choices=["groq", "gemini"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="skip the scoring pass at the end")
    args = ap.parse_args(argv)

    videos = parse_list(Path(args.list_file))
    if not videos:
        err("No URLs found in that file.")
        return 2

    results = []
    for i, (url, creator, tone) in enumerate(videos, 1):
        step(f"[{i}/{len(videos)}] {url}")
        cli = [url, "--top-n", str(args.top_n), "--tone", tone or args.tone]
        if creator:
            cli += ["--creator", creator]
        if args.provider:
            cli += ["--provider", args.provider]
        if args.mock:
            cli.append("--mock")
        started = time.time()
        code = orchestrator.main(cli)
        results.append((url, code, time.time() - started))
        if code != 0:
            err(f"failed: {url}")

    step("Batch summary")
    for url, code, secs in results:
        print(f"  {'ok ' if code == 0 else 'FAIL'}  {secs:6.0f}s  {url}")
    ok(f"{sum(1 for _, c, _ in results if c == 0)}/{len(results)} videos processed")

    if not args.no_eval and any(c == 0 for _, c, _ in results):
        step("Scoring output (go / no-go)")
        evaluate.main([])
    return 0 if all(c == 0 for _, c, _ in results) else 1


if __name__ == "__main__":
    sys.exit(run())
