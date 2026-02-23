"""Raw gesture phrase refinement into natural language text."""

from __future__ import annotations

import logging
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenAI = None


LOGGER = logging.getLogger(__name__)

RULE_BASED_MAP = {
    "ME QUESTION": "I have a question.",
    "HELLO": "Hello.",
    "THANK YOU": "Thank you.",
    "SLOW DOWN": "Could you please slow down?",
    "YES": "Yes.",
    "NO": "No.",
    "WAIT": "Please wait.",
    "AGREE": "I agree.",
    "DISAGREE": "I disagree.",
}


class TextRefiner:
    """Optional LLM text refinement for gesture output."""

    def __init__(
        self,
        enabled: bool = True,
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        self.enabled = enabled
        self.openai_model = openai_model
        self._client: Optional[OpenAI] = None
        if enabled and openai_api_key and OpenAI is not None:
            self._client = OpenAI(api_key=openai_api_key)
        elif enabled:
            LOGGER.info("Text refiner is enabled with local fallback mode.")

    def _rule_based_refine(self, raw_text: str) -> str:
        cleaned = " ".join(raw_text.strip().upper().split())
        if not cleaned:
            return ""
        if cleaned in RULE_BASED_MAP:
            return RULE_BASED_MAP[cleaned]

        # Basic fallback: convert all-caps token stream into sentence case.
        transformed = cleaned.lower()
        transformed = transformed.replace(" i ", " I ")
        sentence = transformed[:1].upper() + transformed[1:]
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        return sentence

    def refine(self, raw_text: str) -> str:
        """Refine raw gesture words into a readable sentence."""
        if not self.enabled:
            return raw_text

        if not raw_text.strip():
            return ""

        if self._client is None:
            return self._rule_based_refine(raw_text)

        try:
            # Placeholder production integration for OpenAI API.
            response = self._client.responses.create(
                model=self.openai_model,
                input=(
                    "Rewrite the following gesture keywords as one short grammatically "
                    f"correct first-person sentence: {raw_text}"
                ),
                max_output_tokens=60,
            )
            text = getattr(response, "output_text", "")
            if text:
                return text.strip()
        except Exception:
            LOGGER.exception("OpenAI refinement failed. Falling back to rule-based text.")

        return self._rule_based_refine(raw_text)

