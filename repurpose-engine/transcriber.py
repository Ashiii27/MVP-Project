"""transcriber.py — audio file in, timestamped transcript out (Groq Whisper).

Groq caps uploads at 25 MB, so anything bigger gets split first:
  * with ffmpeg  -> clean time-based segments (works for any codec)
  * without it   -> pure-Python MP3 frame splitting (no external deps)

Every chunk's segment timestamps are shifted by that chunk's offset, so the
final transcript timeline matches the original video.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path

from config import MAX_UPLOAD_BYTES, SETTINGS
from utils import hhmmss, log, ok, step, warn

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


class TranscriptionError(RuntimeError):
    pass


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: list[Segment] = field(default_factory=list)
    duration: float = 0.0
    language: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "duration": self.duration,
            "language": self.language,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(
            text=data.get("text", ""),
            segments=[Segment(**s) for s in data.get("segments", [])],
            duration=float(data.get("duration") or 0.0),
            language=data.get("language", ""),
        )

    def timestamped(self, every: float = 30.0) -> str:
        """Transcript with [MM:SS] markers every ~`every` seconds.

        This is what the chunker sees — it lets the LLM cite real start/end
        times instead of hallucinating them.
        """
        if not self.segments:
            return self.text
        lines: list[str] = []
        buffer: list[str] = []
        bucket_start = self.segments[0].start
        for seg in self.segments:
            if seg.start - bucket_start >= every and buffer:
                lines.append(f"[{hhmmss(bucket_start)}] " + " ".join(buffer).strip())
                buffer = []
                bucket_start = seg.start
            buffer.append(seg.text.strip())
        if buffer:
            lines.append(f"[{hhmmss(bucket_start)}] " + " ".join(buffer).strip())
        return "\n".join(lines)


# ----------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------
def _probe_duration(path: Path) -> float:
    if FFPROBE:
        try:
            out = subprocess.run(
                [FFPROBE, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(path)],
                capture_output=True, text=True, check=True, timeout=120,
            )
            return float(out.stdout.strip())
        except Exception:  # noqa: BLE001
            pass
    if path.suffix.lower() == ".mp3":
        frames = _mp3_frames(path)
        if frames:
            return sum(f[2] for f in frames)
    return 0.0


def _split_with_ffmpeg(path: Path, chunk_seconds: int, workdir: Path) -> list[tuple[Path, float]]:
    pattern = str(workdir / "chunk_%04d.mp3")
    cmd = [
        FFMPEG, "-v", "error", "-i", str(path),
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-ac", "1", "-ar", "16000", "-b:a", "64k",
        "-reset_timestamps", "1", pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    chunks = sorted(workdir.glob("chunk_*.mp3"))
    return [(c, i * chunk_seconds) for i, c in enumerate(chunks)]


# --- pure-python MP3 splitter (used when ffmpeg is missing) ------------
_BITRATES_V1_L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
_BITRATES_V2_L3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
_RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def _mp3_frames(path: Path) -> list[tuple[int, int, float]]:
    """Return [(offset, size, duration_seconds), ...] for each MPEG audio frame."""
    data = path.read_bytes()
    i = 0
    n = len(data)
    # Skip ID3v2 tag if present.
    if data[:3] == b"ID3" and n > 10:
        size = 0
        for byte in data[6:10]:  # syncsafe 28-bit integer
            size = (size << 7) | (byte & 0x7F)
        i = 10 + size
    frames: list[tuple[int, int, float]] = []
    while i + 4 <= n:
        if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
            i += 1
            continue
        version_bits = (data[i + 1] >> 3) & 0x03
        layer_bits = (data[i + 1] >> 1) & 0x03
        bitrate_idx = (data[i + 2] >> 4) & 0x0F
        rate_idx = (data[i + 2] >> 2) & 0x03
        padding = (data[i + 2] >> 1) & 0x01
        if layer_bits != 0x01 or rate_idx == 3 or bitrate_idx in (0, 15) or version_bits == 1:
            i += 1
            continue
        table = _BITRATES_V1_L3 if version_bits == 3 else _BITRATES_V2_L3
        bitrate = table[bitrate_idx] * 1000
        sample_rate = _RATES[version_bits][rate_idx]
        samples = 1152 if version_bits == 3 else 576
        size = int((samples // 8) * bitrate / sample_rate) + padding
        if size <= 4:
            i += 1
            continue
        frames.append((i, size, samples / sample_rate))
        i += size
    return frames


def _split_mp3_python(path: Path, max_bytes: int, workdir: Path) -> list[tuple[Path, float]]:
    frames = _mp3_frames(path)
    if not frames:
        raise TranscriptionError(
            f"{path.name} is larger than {max_bytes / 1e6:.0f} MB and cannot be split without "
            "ffmpeg. Install ffmpeg (brew install ffmpeg / apt install ffmpeg) and retry."
        )
    data = path.read_bytes()
    chunks: list[tuple[Path, float]] = []
    cur_start_byte = frames[0][0]
    cur_bytes = 0
    cur_offset = 0.0
    elapsed = 0.0
    idx = 0
    for offset, size, dur in frames:
        if cur_bytes + size > max_bytes and cur_bytes > 0:
            out = workdir / f"chunk_{idx:04d}.mp3"
            out.write_bytes(data[cur_start_byte : cur_start_byte + cur_bytes])
            chunks.append((out, cur_offset))
            idx += 1
            cur_start_byte = offset
            cur_bytes = 0
            cur_offset = elapsed
        cur_bytes += size
        elapsed += dur
    if cur_bytes:
        out = workdir / f"chunk_{idx:04d}.mp3"
        out.write_bytes(data[cur_start_byte : cur_start_byte + cur_bytes])
        chunks.append((out, cur_offset))
    return chunks


def split_audio(path: Path, workdir: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> list[tuple[Path, float]]:
    """Return [(chunk_path, start_offset_seconds)]; single item if no split needed."""
    size = path.stat().st_size
    if size <= max_bytes:
        return [(path, 0.0)]

    duration = _probe_duration(path)
    n_chunks = math.ceil(size / max_bytes) + 1
    step(f"Audio is {size / 1e6:.1f} MB — splitting into ~{n_chunks} chunks")

    if FFMPEG and duration:
        chunk_seconds = max(60, int(duration / n_chunks))
        chunks = _split_with_ffmpeg(path, chunk_seconds, workdir)
    elif path.suffix.lower() == ".mp3":
        warn("ffmpeg missing — using the built-in MP3 frame splitter.")
        chunks = _split_mp3_python(path, max_bytes, workdir)
    else:
        raise TranscriptionError(
            f"Cannot split {path.suffix} files without ffmpeg. Install ffmpeg and retry."
        )
    ok(f"{len(chunks)} chunks ready")
    return chunks


# ----------------------------------------------------------------------
# transcription
# ----------------------------------------------------------------------
def _groq_client():
    if not SETTINGS.groq_api_key:
        raise TranscriptionError("GROQ_API_KEY is not set (add it to .env).")
    from groq import Groq

    return Groq(api_key=SETTINGS.groq_api_key)


def _transcribe_one(client, path: Path, offset: float, prompt: str | None) -> tuple[str, list[Segment], float]:
    with path.open("rb") as fh:
        resp = client.audio.transcriptions.create(
            file=(path.name, fh.read()),
            model=SETTINGS.whisper_model,
            response_format="verbose_json",
            temperature=0.0,
            prompt=prompt or None,
        )
    payload = resp if isinstance(resp, dict) else json.loads(resp.model_dump_json())
    text = (payload.get("text") or "").strip()
    segments = [
        Segment(
            start=float(s.get("start", 0.0)) + offset,
            end=float(s.get("end", 0.0)) + offset,
            text=(s.get("text") or "").strip(),
        )
        for s in payload.get("segments") or []
    ]
    duration = float(payload.get("duration") or (segments[-1].end - offset if segments else 0.0))
    return text, segments, duration


def transcribe(
    audio_path: str | Path,
    *,
    cache_path: str | Path | None = None,
    context_prompt: str | None = None,
) -> Transcript:
    """Transcribe an audio file, splitting it first if it exceeds the upload cap."""
    audio_path = Path(audio_path)
    if cache_path and Path(cache_path).exists():
        cached = Transcript.from_dict(json.loads(Path(cache_path).read_text()))
        if cached.text:
            ok(f"Reusing cached transcript ({len(cached.text.split())} words)")
            return cached

    step(f"Transcribing with {SETTINGS.whisper_model}")
    client = _groq_client()

    with tempfile.TemporaryDirectory(prefix="repurpose-chunks-") as tmp:
        chunks = split_audio(audio_path, Path(tmp))
        texts: list[str] = []
        segments: list[Segment] = []
        total = 0.0
        for i, (chunk_path, offset) in enumerate(chunks, 1):
            log(f"  · chunk {i}/{len(chunks)} ({chunk_path.stat().st_size / 1e6:.1f} MB)")
            # Feed the tail of the previous chunk as context to keep names/jargon consistent.
            prompt = context_prompt if i == 1 else " ".join(texts[-1].split()[-60:])
            text, segs, dur = _transcribe_one(client, chunk_path, offset, prompt)
            texts.append(text)
            segments.extend(segs)
            total = max(total, offset + dur)

    transcript = Transcript(
        text=" ".join(t for t in texts if t).strip(),
        segments=segments,
        duration=total,
        language="",
    )
    ok(f"Transcript: {len(transcript.text.split())} words, {hhmmss(transcript.duration)}")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(transcript.as_dict(), indent=2))
    return transcript


if __name__ == "__main__":  # manual smoke test
    import sys

    if len(sys.argv) < 2:
        print("usage: python transcriber.py path/to/audio.mp3")
        raise SystemExit(1)
    print(transcribe(sys.argv[1]).text[:2000])
