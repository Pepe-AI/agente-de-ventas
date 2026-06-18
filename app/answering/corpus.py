"""Corpus loading for the CAG answerer.

The knowledge corpus is a markdown file read once at the composition root and
passed to the answerer; it is never re-read per request.
"""

from __future__ import annotations

from pathlib import Path


def load_corpus(path: str) -> str:
    """Read the knowledge corpus markdown file at ``path``."""
    return Path(path).read_text(encoding="utf-8")
