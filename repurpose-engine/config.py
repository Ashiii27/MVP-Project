"""Central configuration for the repurpose engine.

Everything tunable lives here so the other modules stay boring.
Values come from the environment (.env) with sane defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root no matter where the script is invoked from.
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

PROMPTS_DIR = ROOT / "prompts"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "output"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", OUTPUT_DIR / "_audio"))

# Groq hard-limits Whisper uploads at 25 MB. Stay under it with margin.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 24 * 1024 * 1024))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime settings resolved from the environment."""

    # --- providers -------------------------------------------------
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    # "auto" -> groq if key present, else gemini
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").lower())

    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    whisper_model: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL", "whisper-large-v3")
    )

    # --- generation knobs ------------------------------------------
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "4")))

    # --- content knobs ---------------------------------------------
    creator_name: str = field(default_factory=lambda: os.getenv("CREATOR_NAME", ""))
    tone: str = field(default_factory=lambda: os.getenv("TONE", "casual"))
    min_ideas: int = field(default_factory=lambda: int(os.getenv("MIN_IDEAS", "5")))
    max_ideas: int = field(default_factory=lambda: int(os.getenv("MAX_IDEAS", "7")))
    top_n: int = field(default_factory=lambda: int(os.getenv("TOP_N", "3")))

    # --- behaviour --------------------------------------------------
    keep_audio: bool = field(default_factory=lambda: _env_flag("KEEP_AUDIO", False))
    verbose: bool = field(default_factory=lambda: _env_flag("VERBOSE", True))

    def resolve_provider(self) -> str:
        """Return the concrete provider name to use ('groq' or 'gemini')."""
        if self.llm_provider in {"groq", "gemini"}:
            return self.llm_provider
        if self.groq_api_key:
            return "groq"
        if self.gemini_api_key:
            return "gemini"
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY (or GEMINI_API_KEY) in your .env file."
        )


SETTINGS = Settings()