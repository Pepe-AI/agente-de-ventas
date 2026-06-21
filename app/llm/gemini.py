"""Gemini adapter for the :class:`~app.llm.base.LLM` port.

The only module that knows about google-genai. Uses Gemini's native structured
output (JSON mime type + Pydantic ``response_schema``) and the SDK's async
interface (``client.aio``) so the event loop is never blocked.
"""

from __future__ import annotations

from typing import TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_JSON_MIME = "application/json"
_RATE_LIMITED = 429
# A non-conforming structured output is stochastic; re-roll once (1 try + 1 retry).
_PARSE_ATTEMPTS = 2


class LLMError(RuntimeError):
    """Raised when the model does not return a valid schema instance."""


def is_transient_gemini_error(exc: BaseException) -> bool:
    """Whether a Gemini failure is transient and worth retrying.

    Transient: a 5xx ``ServerError`` (e.g. 503 overloaded), a 429 rate-limit /
    resource-exhausted ``ClientError``, or any httpx transport error — timeouts
    AND network/connection failures (connection refused, reset, read/write,
    server closed the connection), which google-genai lets propagate raw.
    Permanent errors (400, auth) and parse failures are NOT transient.
    """
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return getattr(exc, "code", None) == _RATE_LIMITED
    return isinstance(exc, httpx.TransportError)


class GeminiLLM:
    """LLM adapter backed by google-genai."""

    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    async def complete_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """Call Gemini with native structured output and return the parsed model.

        If the response arrives but does not conform to ``schema`` (the SDK leaves
        ``.parsed`` as None), re-roll once with no backoff — it is stochastic, not
        a load problem. A network/API error from ``generate_content`` propagates
        (it is not a parse failure) for the retry layer to handle.
        """
        for _ in range(_PARSE_ATTEMPTS):
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
            if isinstance(parsed, schema):
                return parsed
        raise LLMError("Gemini did not return a parsed schema instance")


def build_gemini_llm(api_key: str, model: str) -> GeminiLLM:
    """Construct a :class:`GeminiLLM` (composition-root helper)."""
    return GeminiLLM(genai.Client(api_key=api_key), model)
