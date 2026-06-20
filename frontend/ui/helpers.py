"""Small, dependency-light utility functions shared across the app."""

from __future__ import annotations

import html
import re

from frontend.api_client import IRAPIClient
from frontend.ui.constants import DEFAULT_API_URL


def _normalize_url(raw: str) -> str:
    """
    Coerce a user-typed URL into something actually connectable.

    ``0.0.0.0`` is a bind address — servers use it to mean "all interfaces",
    but it is NOT a valid destination for outgoing HTTP requests.
    Replace it with the loopback address so the client can reach the server.
    """
    url = raw.strip().rstrip("/")
    if not url:
        return DEFAULT_API_URL
    # 0.0.0.0 → 127.0.0.1
    url = url.replace("0.0.0.0", "127.0.0.1")
    # add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _client(url: str) -> IRAPIClient:
    return IRAPIClient(_normalize_url(url))


def _token_chips(tokens: list[str]) -> str:
    if not tokens:
        return "<em style='color:var(--text-muted)'>—</em>"
    return "".join(f'<span class="ir-token">{html.escape(t)}</span>' for t in tokens)


def _expansion_chips(terms: list[str]) -> str:
    return "".join(
        f'<span class="ir-expand-term">{html.escape(t)}</span>' for t in terms
    )


def _clean_display_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace for safe UI display."""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _mode_theme(mode: str) -> tuple[str, str, str]:
    """Return (chip_css_class, icon_name, family_label) for a retrieval mode."""
    if mode.startswith("hybrid"):
        return "ir-chip-fuchsia", "venn", "hybrid"
    if mode in ("sbert", "word2vec"):
        return "ir-chip-cyan", "cpu", "dense"
    return "ir-chip-primary", "cpu", "sparse"
