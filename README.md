# MVP-Project

Turning long-form video into publishable text.

## Phase 1 — the core pipeline (`repurpose-engine/`)

A local Python pipeline: YouTube URL → audio → transcript → standalone ideas → a Twitter thread,
a LinkedIn post and an email newsletter segment for each of the top ideas.

```bash
cd repurpose-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your free Groq API key
python main.py "https://youtube.com/watch?v=xyz"
```

Full docs, prompt-tuning guide and the go/no-go criteria: [`repurpose-engine/README.md`](repurpose-engine/README.md).

Stack: Python 3.11, yt-dlp, Groq Whisper `whisper-large-v3`, Groq Llama 3.3 70B (or Gemini 2.0 Flash). Runs on the free tier.
