# repurpose-engine — Phase 1: the core pipeline

One YouTube URL in. A Twitter thread, a LinkedIn post and an email newsletter segment out, for each of the top ideas in the video.

```
python main.py "https://youtube.com/watch?v=xyz"
```

```
download audio (yt-dlp)  →  transcribe (Groq Whisper)  →  extract 5-7 standalone ideas (LLM)
                                                       →  format top 3 × 3 channels (3 LLM calls each)
                                                       →  print + save to output/
```

Cost: $0 on Groq's free tier (20 hrs/day Whisper, 30 req/min Llama 3.3 70B).

---

## 1. Setup (5 minutes)

```bash
cd repurpose-engine
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste your key into .env
```

Get a free Groq key at <https://console.groq.com/keys> and put it in `.env`:

```
GROQ_API_KEY=gsk_...
```

Check what your key can actually reach (model availability changes — see §6):

```bash
python main.py --list-models
```

**Install ffmpeg too** (strongly recommended — it shrinks a 2-hour podcast from ~200 MB to ~15 MB):

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
winget install ffmpeg        # Windows
```

Without ffmpeg the pipeline still runs: it keeps the raw audio stream and, for MP3s over the 25 MB
upload cap, falls back to a built-in pure-Python frame splitter.

Try it offline first, no key needed:

```bash
python main.py --mock --transcript samples/sample_transcript.json --title "Pricing and churn"
```

## 2. Usage

```bash
# the normal case
python main.py "https://youtube.com/watch?v=xyz"

# ghostwrite in a specific voice, format the top 5 ideas
python main.py "https://youtu.be/xyz" --creator "Lenny Rachitsky" --tone authoritative --top-n 5

# use Gemini 2.0 Flash instead of Llama 3.3
python main.py "https://youtu.be/xyz" --provider gemini

# skip stages you already ran
python main.py --audio ~/Downloads/episode.mp3 --title "Episode 42"
python main.py --transcript output/some-video/transcript.json --title "Episode 42"

# force fresh work (ignore cached transcript/ideas)
python main.py "https://youtu.be/xyz" --no-cache
```

Everything lands in `output/<video-slug>/`:

```
output/how-pricing-works-abc123/
├── transcript.txt / transcript.json     # cached; re-runs are free
├── ideas.json                           # all 5-7 ideas, ranked
├── report.md                            # everything in one readable file
├── 01-price-is-a-filter/
│   ├── twitter.md
│   ├── linkedin.md
│   └── email.md
├── 02-.../
└── 03-.../
```

## 3. Files

| File | What it does |
| --- | --- |
| `main.py` | Orchestrator + CLI. Chains the stages, saves and prints. |
| `downloader.py` | yt-dlp wrapper. Audio only, mono 16 kHz mp3 when ffmpeg exists. Caches by video id. |
| `transcriber.py` | Groq Whisper. Splits files over 25 MB (ffmpeg, or a pure-Python MP3 splitter), stitches timestamps back onto one timeline. |
| `chunker.py` | Sends the timestamped transcript to the LLM, gets ranked standalone ideas, dedupes them. Long transcripts are mined in multiple passes. |
| `formatter.py` | Three separate LLM calls per idea + validators + one automatic repair pass. |
| `llm.py` | Groq / Gemini behind one `complete()`. Retries, backoff, free-tier throttling, JSON mode. |
| `evaluate.py` | Slop detector. Scores every output so Task 1.7 isn't a vibes exercise. |
| `batch.py` | Runs a list of videos, then scores them all. |
| `prompts/*.txt` | The actual prompts. **This is where output quality lives.** |
| `mock_llm.py` | Offline fake LLM for testing the plumbing without API calls. |
| `tests/` | 27 offline tests (`python tests/test_pipeline.py`). |

## 4. Task 1.7 — testing 5 videos without kidding yourself

```bash
# edit videos.txt, put in 5 videos you know well (podcast, tutorial, talk, vlog, solo essay)
python batch.py videos.txt
```

That runs all five and then scores every piece of output:

```
format     overall  spec  concis  fresh  rules  words  idea
twitter         82    88      95     75    100    104  01-price-is-a-filter
linkedin        68    52      90     75     80    182  01-price-is-a-filter
...
Average score: 74/100 across 15 pieces
```

* **specificity** — real numbers, names and phrases carried over from the transcript. Low score = the model is summarising vaguely instead of using what the speaker said. This is the single biggest driver of "sounds human".
* **concision** — average sentence length. Above ~14 words and it starts reading like an essay bot.
* **freshness** — banned phrases and AI tells (`delve`, `unlock`, `it's important to note`, `seamless`, cliché openers).
* **rules** — the format's own hard rules: 6 tweets under 270 chars, no hashtags; LinkedIn 150-200 words, line break per sentence, one emoji at the end, ends on a question; email 230-280 words, subject under 55 chars, CTA contains the video URL.

Run `python evaluate.py` any time to re-score what's in `output/`.

### Go / No-Go

* **≥ 75 average, and you'd actually post at least one piece unedited** → GO to Phase 2.
* **60-74** → tighten the specific rules that keep failing (the "Weak spots" table names them).
* **< 60** → ITERATE on `prompts/`. Do not build anything else yet.

### How to iterate on prompts (in order of impact)

1. **Feed more raw speech.** The formatters already get the verbatim transcript slice for each idea (`excerpt_for`). If output feels generic, widen it or raise `max_chars`.
2. **Ban what you hate.** Add phrases to `BANNED` in `formatter.py` and to the prompt's banned list. The validator forces one rewrite pass when they show up.
3. **Give the voice a reference.** Paste 2-3 of the creator's real posts into the top of `prompts/twitter.txt` under a "Here's how they actually write" heading. Few-shot beats adjectives every time.
4. **Change one thing at a time**, re-run `python batch.py videos.txt`, compare the average. `output/eval.json` keeps the detail.
5. **Temperature.** `LLM_TEMPERATURE=0.8` for more voice, `0.5` when it starts inventing facts.

## 5. Config (`.env`)

| Var | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | — | required (Whisper + Llama) |
| `GEMINI_API_KEY` | — | optional alternative for the writing steps |
| `LLM_PROVIDER` | `auto` | `auto` \| `groq` \| `gemini` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | free tier; Llama chat models are enterprise-only now |
| `GEMINI_MODEL` | `gemini-2.5-flash` | |
| `WHISPER_MODEL` | `whisper-large-v3` | `whisper-large-v3-turbo` is faster, slightly worse |
| `GROQ_REASONING_EFFORT` | `low` | gpt-oss models are reasoning models; `low` keeps them writing instead of thinking |
| `CREATOR_NAME` | — | whose voice to write in |
| `TONE` | `casual` | `casual` \| `authoritative` \| `irreverent` \| `warm` |
| `TOP_N` | `3` | ideas to format |
| `MIN_IDEAS` / `MAX_IDEAS` | `5` / `7` | ideas to extract |
| `LLM_TEMPERATURE` | `0.7` | |
| `LLM_MIN_INTERVAL` | `2.1` | seconds between LLM calls (30 req/min free tier) |
| `KEEP_AUDIO` | `false` | keep the mp3 after transcription |

## 6. Which model should I use?

Groq's catalogue moves. In 2026 the Llama chat models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)
became **enterprise-only**, so a free-tier key calling them gets `404 model_not_found`. Current free-tier
text models:

| Model | Speed | Good for |
| --- | --- | --- |
| `openai/gpt-oss-120b` **(default)** | ~500 t/s | Best quality of the free pair. Use it for chunking and formatting. |
| `openai/gpt-oss-20b` | ~1000 t/s | Faster and cheaper; noticeably blander prose. Fine for the chunking pass. |
| `groq/compound-mini` | ~450 t/s | Last-resort system model. |
| `whisper-large-v3` | — | Transcription. Unchanged, still free-tier. |

Or switch the writing steps to Gemini, which has a separate free quota:

```bash
python main.py "https://youtu.be/xyz" --provider gemini      # gemini-2.5-flash
```

You don't have to babysit this. If the configured model returns a 404 the client **fails fast on that
model** (no pointless retries), walks its fallback list, tells you what it switched to, and remembers
the dead model for the rest of the run:

```
! 'llama-3.3-70b-versatile' is not available on your key — trying the next model
! switched to openai/gpt-oss-120b (previous model unavailable on this key)
```

`--model X` pins a model and disables that wandering. `python main.py --list-models` prints exactly
what your key can reach, for both providers, and recommends what to put in `.env`.

A tip for quality: mixing providers is cheap insurance against rate limits and against one model's
house style. `GROQ_MODEL=openai/gpt-oss-120b` for chunking and `--provider gemini` for a second
opinion on the writing is a reasonable A/B when you're tuning prompts.

## 7. Costs and limits

| Step | Calls per video | Free-tier reality |
| --- | --- | --- |
| Whisper | 1 per 25 MB chunk | 20 hrs audio/day |
| Chunking | 1 (+1 per extra 40k chars) | 30 req/min |
| Formatting | 3 per idea, +1 per repair | 3 ideas ≈ 9-12 calls |

A 60-minute podcast is roughly 12-15 LLM calls and about 2-4 minutes wall clock, mostly transcription. `llm.py` spaces calls ~2.1s apart so you don't hit the rate limit.

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Sign in to confirm you're not a bot` | yt-dlp needs cookies: `yt-dlp --cookies-from-browser chrome`, or update yt-dlp (`pip install -U yt-dlp`). |
| `413` / file too large | Install ffmpeg. The 25 MB cap is Groq's. |
| `404 ... model_not_found` | That model isn't on your key (Groq made the Llama models enterprise-only). The client auto-falls back; to silence it, run `python main.py --list-models` and set `GROQ_MODEL` in `.env`. |
| `401 invalid api key` | Fails immediately with a link to regenerate the key. Check for a stray quote or space in `.env`. |
| `rate_limit_exceeded` | Raise `LLM_MIN_INTERVAL` to 3.0. Retries with backoff already happen automatically. |
| Output is short or empty on gpt-oss | It spent the budget on reasoning. Keep `GROQ_REASONING_EFFORT=low`. |
| Model returns prose instead of JSON | Already handled (`extract_json` strips fences and repairs trailing commas). If it persists, drop to `--provider gemini`. |
| Ideas have wrong timestamps | The transcript needs segments. Check `transcript.json` has a non-empty `segments` array. |
| Output sounds robotic | See "How to iterate on prompts" above. That's the make-or-break work, not the code. |

## 9. Tests

```bash
python tests/test_pipeline.py          # 27 tests, no network, no API key
```

Covers timestamp parsing, JSON salvage from messy LLM output, the MP3 frame splitter (offsets and
frame conservation), idea coercion/dedupe, transcript excerpting, all three validators, model
fallback on 404 / auth / rate-limit errors, and a full end-to-end run against the mock LLM.