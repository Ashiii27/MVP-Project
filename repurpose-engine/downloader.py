"""downloader.py — YouTube URL in, audio file + metadata out.

Audio only (bestaudio) so we never pull a 500 MB video for a 10 MB voice track.
If ffmpeg is available we transcode to a small mono 16 kHz mp3 (Whisper's sweet
spot, ~1 MB per 8 minutes). If it isn't, we keep the raw downloaded audio and
let the transcriber deal with it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

from yt_dlp import YoutubeDL

from config import AUDIO_DIR
from utils import log, ok, slugify, step, warn

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@dataclass
class VideoMeta:
    video_id: str
    title: str
    channel: str
    duration: int
    url: str
    description: str
    audio_path: str

    def as_dict(self) -> dict:
        return asdict(self)


class DownloadError(RuntimeError):
    pass


def _base_opts(outtmpl: str) -> dict:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
        "fragment_retries": 5,
        "ignoreerrors": False,
        # Podcasts/livestreams sometimes only expose m4a; that's fine.
        "postprocessors": [],
    }
    if FFMPEG_AVAILABLE:
        opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ]
        # Mono 16 kHz keeps files tiny without hurting Whisper accuracy.
        opts["postprocessor_args"] = {"extractaudio": ["-ac", "1", "-ar", "16000"]}
    return opts


def fetch_metadata(url: str) -> dict:
    """Cheap metadata-only call (no download)."""
    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        return ydl.extract_info(url, download=False)


def download_audio(url: str, output_dir: Path | None = None) -> VideoMeta:
    """Download audio for `url` and return the file path plus metadata."""
    step(f"Downloading audio: {url}")
    output_dir = Path(output_dir or AUDIO_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not FFMPEG_AVAILABLE:
        warn("ffmpeg not found — keeping the original audio stream (install ffmpeg for smaller files).")

    probe = fetch_metadata(url)
    if probe.get("_type") == "playlist":
        raise DownloadError("That URL is a playlist. Pass a single video URL.")

    video_id = probe.get("id", "video")
    title = probe.get("title", "Untitled")
    slug = slugify(title)
    outtmpl = str(output_dir / f"{slug}-{video_id}.%(ext)s")

    # Reuse an already-downloaded file (re-running on the same video is common).
    existing = sorted(output_dir.glob(f"{slug}-{video_id}.*"))
    existing = [p for p in existing if p.suffix.lower() in {".mp3", ".m4a", ".webm", ".opus", ".wav"}]
    if existing:
        audio_path = existing[0]
        ok(f"Reusing cached audio: {audio_path.name}")
    else:
        with YoutubeDL(_base_opts(outtmpl)) as ydl:
            info = ydl.extract_info(url, download=True)
        audio_path = _resolve_downloaded_path(info, output_dir, slug, video_id)
        ok(f"Audio saved: {audio_path.name} ({audio_path.stat().st_size / 1e6:.1f} MB)")

    meta = VideoMeta(
        video_id=video_id,
        title=title,
        channel=probe.get("uploader") or probe.get("channel") or "",
        duration=int(probe.get("duration") or 0),
        url=probe.get("webpage_url") or url,
        description=(probe.get("description") or "")[:2000],
        audio_path=str(audio_path),
    )
    log(f"  · {meta.title} — {meta.channel} ({meta.duration // 60} min)")
    return meta


def _resolve_downloaded_path(info: dict, output_dir: Path, slug: str, video_id: str) -> Path:
    """yt-dlp renames files after post-processing; find whatever landed on disk."""
    candidates = []
    requested = info.get("requested_downloads") or []
    for item in requested:
        for key in ("filepath", "_filename", "filename"):
            if item.get(key):
                candidates.append(Path(item[key]))
    for key in ("filepath", "_filename"):
        if info.get(key):
            candidates.append(Path(info[key]))

    for path in candidates:
        if path.exists():
            return path
        mp3 = path.with_suffix(".mp3")
        if mp3.exists():
            return mp3

    matches = sorted(output_dir.glob(f"{slug}-{video_id}.*"))
    if matches:
        # Prefer mp3 when the post-processor produced one.
        matches.sort(key=lambda p: 0 if p.suffix == ".mp3" else 1)
        return matches[0]

    raise DownloadError("Download finished but no audio file was found on disk.")


if __name__ == "__main__":  # manual smoke test
    import sys

    if len(sys.argv) < 2:
        print('usage: python downloader.py "https://youtube.com/watch?v=..."')
        raise SystemExit(1)
    print(download_audio(sys.argv[1]).as_dict())
