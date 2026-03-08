"""Two-tier ASL gloss to English sentence refinement."""

from __future__ import annotations

from collections import deque
import asyncio
import logging
from typing import Optional

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional runtime dependency
    AsyncOpenAI = None


LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an ASL interpreter. Convert ASL gloss sequences to natural English. "
    "ASL omits articles, uses topic-comment structure, and signs may be in a different "
    "order than English. Preserve meaning, not word order. Respond with only the English sentence."
)

COMMON_GLOSS_MAP = {
    "FINISH": "done",
    "WANT": "want to",
    "GO": "go",
    "ME": "I",
    "YOU": "you",
    "PLEASE": "please",
    "THANK YOU": "thank you",
    "QUESTION": "question",
    "HELLO": "hello",
    "WAIT": "wait",
    "YES": "yes",
    "NO": "no",
}

PRONOUN_BE_VERB = {
    "I": "am",
    "YOU": "are",
    "WE": "are",
    "THEY": "are",
    "HE": "is",
    "SHE": "is",
    "IT": "is",
    "ME": "am",
}


class TextRefiner:
    """Two-tier text refiner: local rules + optional async LLM."""

    def __init__(
        self,
        enabled: bool = True,
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.enabled = enabled
        self.openai_model = openai_model
        self._client: Optional[AsyncOpenAI] = None
        self._cache: dict[str, str] = {}
        self._context_window: deque[str] = deque(maxlen=2)
        self._cache_lock = asyncio.Lock()

        if enabled and openai_api_key and AsyncOpenAI is not None:
            self._client = AsyncOpenAI(api_key=openai_api_key)
        elif enabled:
            LOGGER.info("Text refiner running in local rule-based mode.")

    def _normalize_gloss_sequence(self, raw_text: str) -> str:
        return " ".join(raw_text.strip().upper().split())

    def _apply_common_mappings(self, tokens: list[str]) -> list[str]:
        mapped: list[str] = []
        for token in tokens:
            replacement = COMMON_GLOSS_MAP.get(token, token.lower())
            mapped.extend(replacement.split())
        return mapped

    def _insert_be_verbs(self, tokens: list[str]) -> list[str]:
        out: list[str] = []
        for index, token in enumerate(tokens):
            out.append(token)
            upper = token.upper()
            if upper in PRONOUN_BE_VERB:
                has_next = index + 1 < len(tokens)
                next_token = tokens[index + 1].lower() if has_next else ""
                if next_token not in {"am", "is", "are"}:
                    out.append(PRONOUN_BE_VERB[upper])
        return out

    def refine_tier1(self, raw_text: str) -> str:
        """Tier 1: deterministic low-latency rule-based refinement."""
        cleaned = self._normalize_gloss_sequence(raw_text)
        if not cleaned:
            return ""

        mapped_tokens = self._apply_common_mappings(cleaned.split())
        grammar_tokens = self._insert_be_verbs(mapped_tokens)
        sentence = " ".join(grammar_tokens).strip()
        if not sentence:
            return ""

        sentence = sentence[:1].upper() + sentence[1:]
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

    def refine(self, raw_text: str) -> str:
        """Compatibility sync API returning Tier 1 result."""
        if not self.enabled:
            return raw_text
        return self.refine_tier1(raw_text)

    def add_context(self, sentence: str) -> None:
        sentence = sentence.strip()
        if sentence:
            self._context_window.append(sentence)

    async def refine_tier2_async(self, raw_text: str, tier1_text: str) -> str:
        """
        Tier 2: async LLM refinement with cache + timeout + context.

        Returns tier1 fallback on timeout/error/unavailable key.
        """
        if not self.enabled:
            return raw_text
        if self._client is None:
            return tier1_text

        normalized = self._normalize_gloss_sequence(raw_text)
        if not normalized:
            return tier1_text

        async with self._cache_lock:
            cached = self._cache.get(normalized)
        if cached:
            return cached

        context = list(self._context_window)[-2:]
        user_prompt = (
            f"Previous context (last two sentences): {context}\n"
            f"ASL gloss sequence: {normalized}\n"
            f"Baseline rule-based sentence: {tier1_text}\n"
            "Rewrite as natural English sentence:"
        )
        try:
            response = await asyncio.wait_for(
                self._client.responses.create(
                    model=self.openai_model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_output_tokens=80,
                ),
                timeout=2.0,
            )
            text = (getattr(response, "output_text", "") or "").strip()
            if text:
                if not text.endswith((".", "!", "?")):
                    text += "."
                async with self._cache_lock:
                    self._cache[normalized] = text
                self.add_context(text)
                return text
        except TimeoutError:
            LOGGER.warning("Tier-2 refinement timed out. Falling back to Tier-1 text.")
        except Exception:
            LOGGER.exception("Tier-2 refinement failed.")
        self.add_context(tier1_text)
        return tier1_text

