"""Gemini adapter for the :class:`~app.llm.base.LLM` port.

The only module that knows about google-genai. Uses Gemini's native structured
output (JSON mime type + Pydantic ``response_schema``) and the SDK's async
interface (``client.aio``) so the event loop is never blocked.
"""

from __future__ import annotations

from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_JSON_MIME = "application/json"


class LLMError(RuntimeError):
    """Raised when the model does not return a valid schema instance."""


class GeminiLLM:
    """LLM adapter backed by google-genai."""

    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    async def complete_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """Call Gemini with native structured output and return the parsed model."""
        # google-genai's generate_content is only partially typed at the boundary.
        response = await self._client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type=_JSON_MIME,
                response_schema=schema,
            ),
        )
        parsed = response.parsed
        if not isinstance(parsed, schema):
            raise LLMError("Gemini did not return a parsed schema instance")
        return parsed


def build_gemini_llm(api_key: str, model: str) -> GeminiLLM:
    """Construct a :class:`GeminiLLM` (composition-root helper)."""
    return GeminiLLM(genai.Client(api_key=api_key), model)
