"""Generic, mockable LLM port.

The core depends on this abstraction; only the concrete adapter (GeminiLLM)
knows about a specific SDK. One call returns a validated Pydantic instance of
the requested schema (native structured output).
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """The LLM was unavailable after exhausting transient retries.

    A port-level signal (provider-agnostic) so callers can degrade gracefully
    on a transient outage without knowing about any specific SDK's errors.
    """


class LLM(Protocol):
    """A language model that returns structured, schema-validated output."""

    async def complete_structured(self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """Run ``prompt`` and parse the model's reply into ``schema``."""
        ...
