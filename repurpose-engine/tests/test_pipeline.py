"""Offline test suite — no API keys, no network.

    python -m unittest discover -s tests -v
    (or: ./venv/bin/python tests/test_pipeline.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
from chunker import Idea, _coerce_ideas, _dedupe, excerpt_for  # noqa: E402
from formatter import validate_email, validate_linkedin, validate_twitter  # noqa: E402
from mock_llm import MockLLM  # noqa: E402
from transcriber import Segment, Transcript, _mp3_frames, _split_mp3_python, split_audio  # noqa: E402
from utils import extract_json, hhmmss, parse_timestamp, slugify, word_count  # noqa: E402


def fake_mp3(path: Path, n_frames: int = 500) -> Path:
    """A structurally valid MPEG1 Layer III 128kbps/44.1kHz file of silence."""
    header = bytes([0xFF, 0xFB, 0x90, 0x00])
    frame = header + b"\x00" * (417 - 4)
    path.write_bytes(b"ID3" + b"\x03\x00\x00\x00\x00\x00\x00" + frame * n_frames)
    return path


class TestUtils(unittest.TestCase):
    def test_parse_timestamp(self):
        self.assertEqual(parse_timestamp("01:02:03"), 3723.0)
        self.assertEqual(parse_timestamp("02:03"), 123.0)
        self.assertEqual(parse_timestamp(90), 90.0)
        self.assertEqual(parse_timestamp("90.5"), 90.5)
        self.assertEqual(parse_timestamp(None), 0.0)
        self.assertEqual(parse_timestamp("garbage"), 0.0)

    def test_hhmmss(self):
        self.assertEqual(hhmmss(65), "01:05")
        self.assertEqual(hhmmss(3725), "01:02:05")

    def test_slugify(self):
        self.assertEqual(slugify("Why $99 Beats $29 — Really!"), "why-99-beats-29-really")

    def test_extract_json_variants(self):
        self.assertEqual(extract_json('{"a": 1}')["a"], 1)
        self.assertEqual(extract_json('```json\n{"a": 2}\n```')["a"], 2)
        self.assertEqual(extract_json('Sure! Here you go:\n{"a": 3}\nHope that helps')["a"], 3)
        self.assertEqual(extract_json('{"a": [1,2,],}')["a"], [1, 2])
        with self.assertRaises(ValueError):
            extract_json("no json here at all")

    def test_word_count(self):
        self.assertEqual(word_count("one two three-four don't"), 4)


class TestTranscript(unittest.TestCase):
    def setUp(self):
        self.t = Transcript(
            text="a b c",
            segments=[Segment(0, 20, "first bit"), Segment(20, 45, "second bit"), Segment(45, 70, "third bit")],
            duration=70,
        )

    def test_timestamped_buckets(self):
        out = self.t.timestamped(every=30)
        self.assertIn("[00:00]", out)
        self.assertIn("[00:45]", out)
        self.assertEqual(len(out.strip().split("\n")), 2)

    def test_roundtrip(self):
        again = Transcript.from_dict(json.loads(json.dumps(self.t.as_dict())))
        self.assertEqual(again.segments[1].text, "second bit")
        self.assertEqual(again.duration, 70)


class TestAudioSplitting(unittest.TestCase):
    def test_frames_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fake_mp3(Path(tmp) / "a.mp3", 300)
            frames = _mp3_frames(path)
            self.assertEqual(len(frames), 300)
            self.assertAlmostEqual(sum(f[2] for f in frames), 300 * 1152 / 44100, places=3)

    def test_no_split_when_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fake_mp3(Path(tmp) / "a.mp3", 100)
            chunks = split_audio(path, Path(tmp), max_bytes=10_000_000)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0][1], 0.0)

    def test_python_splitter_offsets_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = fake_mp3(Path(tmp) / "big.mp3", 2000)
            work = Path(tmp) / "chunks"
            work.mkdir()
            limit = 200_000
            chunks = _split_mp3_python(src, limit, work)
            self.assertGreater(len(chunks), 3)
            for path, offset in chunks:
                self.assertLessEqual(path.stat().st_size, limit)
                self.assertEqual(len(_mp3_frames(path)) > 0, True)
            offsets = [o for _, o in chunks]
            self.assertEqual(offsets, sorted(offsets))
            self.assertEqual(offsets[0], 0.0)
            # every frame accounted for
            total = sum(len(_mp3_frames(p)) for p, _ in chunks)
            self.assertEqual(total, 2000)


class TestChunker(unittest.TestCase):
    def test_coerce_handles_wrappers_and_junk(self):
        payload = {"ideas": [
            {"title": "A real idea", "summary": "Something specific.", "start_time": "62", "end_time": "02:10", "hook": "h", "score": "9"},
            {"title": "", "summary": "no title"},
            "garbage",
        ]}
        ideas = _coerce_ideas(payload)
        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0].start_time, "01:02")
        self.assertEqual(ideas[0].score, 9)

    def test_score_clamped(self):
        ideas = _coerce_ideas([{"title": "t", "summary": "s", "score": 99}])
        self.assertEqual(ideas[0].score, 10)

    def test_dedupe_near_duplicates(self):
        ideas = [
            Idea("Pricing filters your customers", "s", "00:00", "01:00", "h"),
            Idea("Pricing filters your customers today", "s", "05:00", "06:00", "h"),
            Idea("Read cancellation notes aloud", "s", "10:00", "11:00", "h"),
        ]
        self.assertEqual(len(_dedupe(ideas)), 2)

    def test_excerpt_uses_time_range(self):
        t = Transcript(
            text="x",
            segments=[Segment(i * 10, i * 10 + 10, f"sentence number {i} with enough words to matter") for i in range(30)],
            duration=300,
        )
        idea = Idea("t", "s", "00:50", "01:40", "h")
        excerpt = excerpt_for(t, idea)
        self.assertIn("sentence number 6", excerpt)
        self.assertNotIn("sentence number 25", excerpt)

    def test_excerpt_fallback_without_segments(self):
        t = Transcript(text="word " * 5000, segments=[], duration=600)
        excerpt = excerpt_for(t, Idea("t", "s", "05:00", "06:00", "h"))
        self.assertTrue(len(excerpt) > 100)


class TestValidators(unittest.TestCase):
    def test_twitter_catches_everything(self):
        bad = {"tweets": [
            "Short",
            "x" * 300,
            "Use #hashtags now",
            "Emoji 🚀 here",
            "We must leverage this",
            "Fine tweet with an em-dash — right here",
        ]}
        issues = " ".join(validate_twitter(bad))
        self.assertIn("chars", issues)
        self.assertIn("hashtag", issues)
        self.assertIn("emoji", issues)
        self.assertIn("leverage", issues)
        self.assertIn("em-dash", issues)

    def test_twitter_accepts_good_thread(self):
        good = {"tweets": [
            "Dropping your price does not fix a demand problem. It changes who shows up.",
            "We ran the same page at 29, 49 and 99 dollars for eleven months.",
            "The 29 dollar page converted better. Revenue per visitor was three times higher at 99.",
            "Refunds on the cheap plan hit 9 percent. On the expensive plan, under 2.",
            "The customers who paid least filed the most support tickets.",
            "Price is a filter. Set it where the customers you want walk through the door.",
        ]}
        self.assertEqual(validate_twitter(good), [])

    def test_linkedin_word_count_and_ending(self):
        post = {"post": "\n".join(["Line one." for _ in range(8)])}
        issues = " ".join(validate_linkedin(post))
        self.assertIn("words", issues)
        self.assertIn("question", issues)

    def test_linkedin_accepts_good_post(self):
        sentences = [
            "Everyone says lower prices win more customers.",
            "That was not true for us.",
            "We ran the same landing page at 29 dollars and at 99 dollars for eleven months.",
            "The cheap page converted slightly better at 2.1 percent versus 1.8 percent.",
            "Revenue per visitor was almost three times higher on the expensive page.",
            "Refund rate on the cheap plan was 9 percent, against under 2 percent on the expensive one.",
            "The customers paying us the least also filed the most support tickets.",
            "Price is a filter, and it sorts people before they ever speak to you.",
            "So we stopped asking what the market will bear and started asking who we want to serve.",
            "We raised the floor price twice in a year and lost about four percent of signups.",
            "Support load dropped by a third in the same period.",
            "Nobody on the team wants to go back to the cheaper plan.",
            "The cheap tier was not a growth lever, it was a filter pointed the wrong way.",
            "Which customers would you lose if you doubled your price tomorrow, and would you miss them? 🙂",
        ]
        self.assertEqual(validate_linkedin({"post": "\n".join(sentences)}), [])

    def test_email_requires_url_and_length(self):
        issues = " ".join(validate_email({"subject": "s", "body": "too short"}, "https://youtu.be/abc"))
        self.assertIn("words", issues)
        self.assertIn("URL", issues)

    def test_email_flags_long_subject(self):
        body = ("word " * 250) + "https://youtu.be/abc"
        issues = " ".join(validate_email({"subject": "x" * 80, "body": body}, "https://youtu.be/abc"))
        self.assertIn("Subject", issues)


class TestEndToEndMock(unittest.TestCase):
    def test_full_pipeline_writes_three_formats(self):
        llm.set_client(MockLLM())
        import main
        from config import OUTPUT_DIR

        with tempfile.TemporaryDirectory() as tmp:
            import config

            original = config.OUTPUT_DIR
            config.OUTPUT_DIR = Path(tmp)
            main.OUTPUT_DIR = Path(tmp)
            try:
                code = main.main([
                    "--mock",
                    "--transcript", str(ROOT / "samples" / "sample_transcript.json"),
                    "--title", "Test video",
                    "--creator", "Tester",
                    "--top-n", "2",
                    "--no-cache",
                    "--quiet",
                ])
                self.assertEqual(code, 0)
                run_dir = Path(tmp) / "test-video"
                self.assertTrue((run_dir / "report.md").exists())
                self.assertTrue((run_dir / "ideas.json").exists())
                folders = sorted(p for p in run_dir.iterdir() if p.is_dir())
                self.assertEqual(len(folders), 2)
                for folder in folders:
                    for name in ("twitter.md", "linkedin.md", "email.md"):
                        content = (folder / name).read_text()
                        self.assertGreater(len(content.strip()), 80, f"{folder.name}/{name} too short")
            finally:
                config.OUTPUT_DIR = original
                main.OUTPUT_DIR = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
