"""
IR 2026 Search Engine — Streamlit frontend.

Run from the project root::

    streamlit run frontend/app.py

Requirements:
    - API Gateway running at the configured URL (default http://127.0.0.1:8000)
    - ``pip install streamlit requests`` (or see frontend/requirements.txt)
"""

from __future__ import annotations

import html
import re
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# make sibling package importable when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.api_client import (  # noqa: E402
    IRAPIClient,
    IRAPIError,
    QueryProcessingInfo,
    SearchResponse,
    SearchResultItem,
)

# ─────────────────────────────────────── Constants ──────────────────────────

DEFAULT_API_URL = "http://127.0.0.1:8000"

RETRIEVAL_MODES: dict[str, str] = {
    "bm25": "BM25  (Probabilistic)",
    "tfidf": "TF-IDF  (Sparse VSM)",
    "sbert": "SBERT  (Dense Embedding)",
    "word2vec": "Word2Vec  (Dense Embedding)",
    "hybrid_parallel": "Hybrid — Parallel Fusion",
    "hybrid_serial": "Hybrid — Serial Cascade",
}

# Always fetch the maximum from the gateway; UI preview is trimmed client-side
# so changing the slider never requires a new network request.
_FETCH_SNIPPET_CHARS = 2000

# ─────────────────────────────────────── Icon system ────────────────────────
# Hand-built, geometry-only line icons (no emoji, no external icon fonts).
# Every shape below is composed of plain SVG primitives so nothing can render
# as a broken/empty glyph.

_ICONS: dict[str, str] = {
    "search": '<circle cx="11" cy="11" r="6"></circle>'
    '<line x1="20" y1="20" x2="15.4" y2="15.4"></line>',
    "target": '<circle cx="12" cy="12" r="8"></circle>'
    '<circle cx="12" cy="12" r="4"></circle>'
    '<circle cx="12" cy="12" r="0.6" fill="currentColor" stroke="none"></circle>',
    "doc": '<rect x="5" y="3" width="14" height="18" rx="2"></rect>'
    '<line x1="8" y1="8" x2="16" y2="8"></line>'
    '<line x1="8" y1="12" x2="16" y2="12"></line>'
    '<line x1="8" y1="16" x2="13" y2="16"></line>',
    "sliders": '<line x1="4" y1="6" x2="20" y2="6"></line>'
    '<circle cx="9" cy="6" r="2" fill="currentColor"></circle>'
    '<line x1="4" y1="12" x2="20" y2="12"></line>'
    '<circle cx="16" cy="12" r="2" fill="currentColor"></circle>'
    '<line x1="4" y1="18" x2="20" y2="18"></line>'
    '<circle cx="11" cy="18" r="2" fill="currentColor"></circle>',
    "layers": '<ellipse cx="12" cy="6" rx="7" ry="2.6"></ellipse>'
    '<ellipse cx="12" cy="12" rx="7" ry="2.6"></ellipse>'
    '<ellipse cx="12" cy="18" rx="7" ry="2.6"></ellipse>',
    "list": '<line x1="9" y1="6" x2="20" y2="6"></line>'
    '<line x1="9" y1="12" x2="20" y2="12"></line>'
    '<line x1="9" y1="18" x2="20" y2="18"></line>'
    '<circle cx="4.5" cy="6" r="1" fill="currentColor"></circle>'
    '<circle cx="4.5" cy="12" r="1" fill="currentColor"></circle>'
    '<circle cx="4.5" cy="18" r="1" fill="currentColor"></circle>',
    "clock": '<circle cx="12" cy="12" r="9"></circle>'
    '<line x1="12" y1="7.5" x2="12" y2="12"></line>'
    '<line x1="12" y1="12" x2="15.2" y2="14"></line>',
    "bars": '<rect x="4" y="13" width="3.2" height="7" rx="1" fill="currentColor" stroke="none"></rect>'
    '<rect x="10.4" y="8" width="3.2" height="12" rx="1" fill="currentColor" stroke="none"></rect>'
    '<rect x="16.8" y="3" width="3.2" height="17" rx="1" fill="currentColor" stroke="none"></rect>',
    "venn": '<circle cx="9" cy="12" r="6.2"></circle>'
    '<circle cx="15" cy="12" r="6.2"></circle>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"></rect>'
    '<line x1="9" y1="2" x2="9" y2="6"></line><line x1="15" y1="2" x2="15" y2="6"></line>'
    '<line x1="9" y1="18" x2="9" y2="22"></line><line x1="15" y1="18" x2="15" y2="22"></line>'
    '<line x1="2" y1="9" x2="6" y2="9"></line><line x1="2" y1="15" x2="6" y2="15"></line>'
    '<line x1="18" y1="9" x2="22" y2="9"></line><line x1="18" y1="15" x2="22" y2="15"></line>',
    "zap": '<polygon points="13,2 3,14 12,14 11,22 21,10 12,10" fill="currentColor" stroke="none"></polygon>',
    "sparkle": '<polygon points="12,2 14.12,9.88 22,12 14.12,14.12 12,22 9.88,14.12 2,12 9.88,9.88" '
    'fill="currentColor" stroke="none"></polygon>',
    "check": '<circle cx="12" cy="12" r="9"></circle>'
    '<polyline points="8,12 11,15 16,9"></polyline>',
    "server": '<rect x="3" y="4" width="18" height="6" rx="1.5"></rect>'
    '<rect x="3" y="14" width="18" height="6" rx="1.5"></rect>'
    '<circle cx="7" cy="7" r="0.8" fill="currentColor" stroke="none"></circle>'
    '<circle cx="7" cy="17" r="0.8" fill="currentColor" stroke="none"></circle>',
    "alert": '<circle cx="12" cy="12" r="9"></circle>'
    '<line x1="12" y1="8" x2="12" y2="13"></line>'
    '<circle cx="12" cy="16.2" r="0.6" fill="currentColor" stroke="none"></circle>',
}


def _icon(name: str, size: int = 15, cls: str = "") -> str:
    """Render a small inline SVG icon. Falls back to a plain circle if unknown."""
    inner = _ICONS.get(name, '<circle cx="12" cy="12" r="9"></circle>')
    return (
        f'<svg class="ir-icon {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.75" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>'
    )


def _section_label(text: str, icon_name: str) -> str:
    return f'<div class="ir-sec">{_icon(icon_name, 13)}<span>{text}</span></div>'


def _eyebrow(text: str, icon_name: str) -> str:
    return f'<div class="ir-eyebrow">{_icon(icon_name, 13)}<span>{text}</span></div>'


# ─────────────────────────────────────── CSS ────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Sora:wght@500;600;700;800&display=swap');

:root {
    --bg: #0a0c12;
    --bg-elev: #0e111a;
    --surface: #131826;
    --surface-2: #171d2e;
    --border: rgba(255,255,255,0.08);
    --border-strong: rgba(255,255,255,0.16);

    --text-primary: #edf0f7;
    --text-secondary: #99a2b8;
    --text-muted: #5b6378;

    --primary: #818cf8;
    --primary-strong: #6366f1;
    --primary-soft: rgba(129,140,248,0.12);
    --primary-border: rgba(129,140,248,0.30);

    --cyan: #38d9e8;
    --cyan-soft: rgba(56,217,232,0.12);
    --cyan-border: rgba(56,217,232,0.30);

    --fuchsia: #e879f9;
    --fuchsia-soft: rgba(232,121,249,0.12);
    --fuchsia-border: rgba(232,121,249,0.30);

    --success: #34d399;
    --success-soft: rgba(52,211,153,0.12);
    --success-border: rgba(52,211,153,0.30);

    --warning: #fbbf24;
    --warning-soft: rgba(251,191,36,0.12);
    --warning-border: rgba(251,191,36,0.30);

    --danger: #fb7185;
    --danger-soft: rgba(251,113,133,0.12);
    --danger-border: rgba(251,113,133,0.30);

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --shadow-card: 0 1px 2px rgba(0,0,0,0.45), 0 12px 28px -12px rgba(0,0,0,0.55);
}

/* ── Base app shell ──────────────────────────────────────────────────── */
.stApp { background: var(--bg) !important; }
[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stDecoration"] {
    background: linear-gradient(90deg, var(--primary-strong), var(--cyan), var(--fuchsia)) !important;
    height: 3px !important;
}
/* Removed .stApp span and .stApp div to protect native icon fonts */
[data-testid="stAppViewContainer"], .stApp, .stApp p, .stApp label {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--bg-elev) !important;
    border-right: 1px solid var(--border);
}
/* Collapse the empty native header and force zero top padding */
[data-testid="stSidebarHeader"] {
    padding: 0 !important;
    min-height: 0 !important;
    height: 0 !important;
}
[data-testid="stSidebarContent"], 
[data-testid="stSidebarUserContent"] { 
    padding-top: 1rem !important; /* Adjust this value to nudge it exactly where you want */
}
/* Reduce the flexbox gap between Streamlit widgets in the sidebar */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.75rem !important; /* Streamlit's default is usually 1rem or 1.5rem */
}

/* Shrink the massive default margins on st.divider() */
[data-testid="stSidebar"] hr {
    margin: 1rem 0 !important; /* Default is often over 2em */
}
[data-testid="stCaptionContainer"], .stCaption { color: var(--text-muted) !important; }
.ir-icon { display: inline-block; vertical-align: -3px; flex-shrink: 0; }

/* ── Native widget skinning ──────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-soft) !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 4px var(--primary-soft) !important;
}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--text-muted) !important; }
[data-testid="stWidgetLabel"] p {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
[data-testid="baseButton-secondary"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color .15s ease, background .15s ease;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: var(--primary-border) !important;
    background: var(--primary-soft) !important;
    color: var(--primary) !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--primary-strong), #5457e0) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 4px 16px -4px rgba(99,102,241,0.55) !important;
}
[data-testid="baseButton-primary"]:hover { filter: brightness(1.08); }
[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
}
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stStatusWidget"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 100px !important;
    box-shadow: var(--shadow-card) !important;
}
[data-testid="stStatusWidget"] svg { display: none !important; }
[data-testid="stStatusWidget"]::before {
    content: "";
    width: 13px; height: 13px;
    border-radius: 50%;
    border: 2px solid var(--primary-border);
    border-top-color: var(--primary);
    display: inline-block;
    margin-right: 6px;
    animation: ir-spin .7s linear infinite;
}
[data-testid="stSpinner"] > div { color: var(--text-secondary) !important; }
@keyframes ir-spin { to { transform: rotate(360deg); } }

/* ── Brand header ─────────────────────────────────────────────────────── */
.ir-brand { display: flex; align-items: center; gap: 10px; }
.ir-brand-badge {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--primary-strong), var(--cyan));
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    box-shadow: 0 4px 14px -4px rgba(99,102,241,0.6);
    flex-shrink: 0;
}
.ir-brand-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 16.5px;
    color: var(--text-primary);
    letter-spacing: -0.2px;
}

.ir-page-header { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.ir-page-header .ir-brand-badge { width: 42px; height: 42px; border-radius: 12px; }
.ir-page-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 26px;
    color: var(--text-primary);
    letter-spacing: -0.4px;
    line-height: 1.15;
}
.ir-page-subtitle {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 1px;
}

/* ── Chips (meta pills + stats row) ──────────────────────────────────── */
.ir-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 18px; }
.ir-chip {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12.5px; font-weight: 600;
    padding: 6px 13px;
    border-radius: 100px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-secondary);
    line-height: 1;
}
.ir-chip b { color: var(--text-primary); font-weight: 700; }
.ir-chip-primary { border-color: var(--primary-border); background: var(--primary-soft); color: var(--primary); }
.ir-chip-cyan { border-color: var(--cyan-border); background: var(--cyan-soft); color: var(--cyan); }
.ir-chip-fuchsia { border-color: var(--fuchsia-border); background: var(--fuchsia-soft); color: var(--fuchsia); }

/* ── Section / eyebrow labels (sidebar) ──────────────────────────────── */
.ir-sec {
    display: flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px; font-weight: 700;
    letter-spacing: 1.3px; text-transform: uppercase;
    color: var(--text-muted);
    margin: 18px 0 10px;
}
.ir-sec .ir-icon { color: var(--primary); }
.ir-eyebrow {
    display: flex; align-items: center; gap: 6px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12px; font-weight: 600;
    color: var(--text-secondary);
    margin: 10px 0 8px;
    padding-left: 10px;
    border-left: 2px solid var(--primary-border);
}
.ir-eyebrow .ir-icon { color: var(--primary); }

/* ── Connection status badge ─────────────────────────────────────────── */
.ir-status {
    display: flex; align-items: center; gap: 7px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px; font-weight: 600;
    padding: 7px 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}
.ir-status-ok { background: var(--success-soft); border-color: var(--success-border); color: var(--success); }
.ir-status-bad { background: var(--danger-soft); border-color: var(--danger-border); color: var(--danger); }
.ir-status-sub { color: var(--text-secondary); font-weight: 500; }

/* ── Query insights panel ────────────────────────────────────────────── */
.ir-qp {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 16px 20px;
    margin-bottom: 18px;
}
.ir-qp-head {
    display: flex; align-items: center; gap: 7px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12.5px; font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.6px;
    margin-bottom: 10px;
}
.ir-qp-head .ir-icon { color: var(--primary); }
.ir-qp-row { display: flex; align-items: baseline; gap: 8px; margin-bottom: 7px; flex-wrap: wrap; }
.ir-qp-label { font-size: 12px; color: var(--text-muted); font-weight: 600; min-width: 72px; }
.ir-qp-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--text-primary);
}
.ir-spell {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10.5px; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase;
    color: var(--warning);
    background: var(--warning-soft);
    border: 1px solid var(--warning-border);
    padding: 2px 9px;
    border-radius: 100px;
    margin-left: 8px;
    vertical-align: middle;
}
.ir-token {
    display: inline-flex; align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--surface-2);
    border: 1px solid var(--border);
    padding: 2px 9px;
    border-radius: 100px;
    margin: 2px 3px 2px 0;
}
.ir-expand-term {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--cyan);
    background: var(--cyan-soft);
    border: 1px solid var(--cyan-border);
    padding: 2px 9px;
    border-radius: 100px;
    margin: 2px 3px 2px 0;
    display: inline-flex;
}

/* ── Result card ──────────────────────────────────────────────────────── */
.ir-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 18px 22px 16px;
    margin-bottom: 14px;
    box-shadow: var(--shadow-card);
    transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease;
}
.ir-card:hover {
    border-color: var(--primary-border);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.4), 0 20px 36px -16px rgba(99,102,241,0.28);
}
.ir-card-header {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px;
    margin-bottom: 12px;
}
.ir-card-left { display: flex; align-items: center; gap: 9px; }

.ir-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 700;
    color: var(--text-secondary);
    background: var(--surface-2);
    border: 1px solid var(--border);
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
}
.ir-rank-1 { color: #f5c451; background: rgba(245,196,81,0.12); border-color: rgba(245,196,81,0.4); }
.ir-rank-2 { color: #c7cedb; background: rgba(199,206,219,0.10); border-color: rgba(199,206,219,0.32); }
.ir-rank-3 { color: #e0a37a; background: rgba(224,163,122,0.12); border-color: rgba(224,163,122,0.36); }

.ir-doc-id {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px; font-weight: 600;
    color: var(--primary);
    background: var(--primary-soft);
    border: 1px solid var(--primary-border);
    padding: 4px 11px;
    border-radius: var(--radius-sm);
    letter-spacing: 0.2px;
    user-select: all;
    cursor: text;
}
.ir-doc-id .ir-icon { opacity: 0.75; }

.ir-score-block { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; min-width: 110px; }
.ir-score {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px; font-weight: 600;
    color: var(--success);
    background: var(--success-soft);
    border: 1px solid var(--success-border);
    padding: 4px 11px;
    border-radius: var(--radius-sm);
}
.ir-score-bar {
    width: 100%; height: 4px;
    background: var(--surface-2);
    border-radius: 100px;
    overflow: hidden;
}
.ir-score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--success), #6ee7b7);
    border-radius: 100px;
}

.ir-title {
    font-family: 'Sora', sans-serif;
    font-size: 15.5px; font-weight: 600;
    color: var(--text-primary);
    line-height: 1.5;
    margin-bottom: 7px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.ir-snippet {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13.5px;
    color: var(--text-secondary);
    line-height: 1.75;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    -webkit-mask-image: linear-gradient(to bottom, black 75%, transparent 100%);
    mask-image: linear-gradient(to bottom, black 75%, transparent 100%);
}
.ir-snippet-muted { color: var(--text-muted) !important; font-style: italic; -webkit-line-clamp: 2; }

/* Hide the transient "Missing Submit Button" warning inside forms */
[data-testid="stForm"] [data-testid="stAlert"] {
    display: none !important;
}
/* Target the main block container using both class and modern test-ids */
.block-container, 
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
}

/* Optional: If the empty top header bar is also pushing things down, shrink it */
[data-testid="stHeader"] {
    height: 2.5rem !important; 
    min-height: 2.5rem !important;
}
</style>
"""


# ─────────────────────────────────────── Helpers ────────────────────────────


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


# ─────────────────────────────────────── Component renderers ────────────────


def render_query_processing(qp: QueryProcessingInfo) -> None:
    """Render the query-processing info banner above the results list."""
    spell_badge = (
        f'<span class="ir-spell">{_icon("check", 10)}spell corrected</span>'
        if qp.spell_corrected
        else ""
    )

    html_content = (
        '<div class="ir-qp">'
        f'<div class="ir-qp-head">{_icon("search", 13)}<span>Query Insights</span></div>'
        '<div class="ir-qp-row">'
        '<span class="ir-qp-label">Original</span>'
        f'<span class="ir-qp-value">{html.escape(qp.original_query)}</span>'
        "</div>"
        '<div class="ir-qp-row">'
        '<span class="ir-qp-label">Processed</span>'
        f'<span class="ir-qp-value">{html.escape(qp.processed_query)}</span>{spell_badge}'
        "</div>"
        '<div class="ir-qp-row" style="margin-top:2px">'
        '<span class="ir-qp-label">Tokens</span>'
        f"<span>{_token_chips(qp.query_tokens)}</span>"
        "</div>"
    )

    if qp.expanded_terms:
        html_content += (
            '<div class="ir-qp-row" style="margin-top:2px">'
            '<span class="ir-qp-label">Expansions</span>'
            f"<span>{_expansion_chips(qp.expanded_terms)}</span>"
            "</div>"
        )

    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)


def render_result_card(
    item: SearchResultItem, preview_chars: int, max_score: float = 1.0
) -> None:
    """Render one search result card."""

    # 1. Determine snippet HTML
    if item.text and preview_chars > 0:
        display_text = _clean_display_text(item.text)
        preview = html.escape(display_text[:preview_chars])
        ellipsis = "…" if len(display_text) > preview_chars else ""
        snippet_html = f'<div class="ir-snippet">{preview}{ellipsis}</div>'
    elif item.text and preview_chars == 0:
        snippet_html = (
            '<div class="ir-snippet ir-snippet-muted">'
            "Preview hidden — open the panel below to read the full document."
            "</div>"
        )
    else:
        snippet_html = '<div class="ir-snippet ir-snippet-muted">No text returned from server.</div>'

    # 2. Score bar width, purely a presentational scaling of the existing score
    safe_max = max_score if max_score and max_score > 0 else 1.0
    bar_pct = max(4, min(100, round((item.score / safe_max) * 100)))

    rank_cls = f"ir-rank ir-rank-{item.rank}" if item.rank in (1, 2, 3) else "ir-rank"

    card_html = (
        '<div class="ir-card">'
        '<div class="ir-card-header">'
        '<div class="ir-card-left">'
        f'<span class="{rank_cls}">{item.rank}</span>'
        f'<span class="ir-doc-id">{_icon("doc", 12)}{html.escape(item.doc_id)}</span>'
        "</div>"
        '<div class="ir-score-block">'
        f'<span class="ir-score">{_icon("target", 12)}{item.score:.4f}</span>'
        f'<div class="ir-score-bar"><div class="ir-score-bar-fill" style="width:{bar_pct}%"></div></div>'
        "</div>"
        "</div>"
    )

    if item.title:
        card_html += f'<div class="ir-title">{html.escape(item.title)}</div>'

    card_html += f"{snippet_html}</div>"
    st.markdown(card_html, unsafe_allow_html=True)

    # 3. Full unprocessed original text (straight from MongoDB)
    if item.text:
        with st.expander("Full original document text", icon=":material/description:"):
            st.code(item.doc_id, language=None)  # easy click-to-copy the ID
            st.text(_clean_display_text(item.text))


# ─────────────────────────────────────── Sidebar ────────────────────────────


def render_sidebar() -> dict[str, Any]:
    """
    Build sidebar controls and return the current configuration.

    Returns
    -------
    dict
        Keys: api_url, dataset, execution_mode, retrieval_mode,
        sparse_method, dense_method, bm25_k1, bm25_b,
        alpha, cascade_top_n, top_k, preview_chars.
    """
    with st.sidebar:
        st.markdown(
            f"""
                <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 16px;">
                    <div class="ir-brand-badge">{_icon("search", 20)}</div>
                        <div style="display: flex; flex-direction: column;">
                            <div class="ir-brand-title" style="line-height: 1.2;">IR Search Engine</div>
                            <div style="color: #5f6470; font-size: 13.5px;">Damascus University · IR 2026</div>
                        </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # ── API Gateway ──────────────────────────────────────────────────
        st.markdown(_section_label("API Gateway", "server"), unsafe_allow_html=True)

        raw_url: str = (
            st.text_input(
                "gateway_url",
                value=st.session_state.get("api_url_raw", DEFAULT_API_URL),
                placeholder="http://127.0.0.1:8000",
                label_visibility="collapsed",
            )
            or ""
        )
        api_url = _normalize_url(raw_url)

        if "0.0.0.0" in raw_url:
            st.caption(f"0.0.0.0 is a bind address — using {api_url} instead")

        if st.session_state.get("api_url") != api_url:
            st.session_state.pop("health_ok", None)
        st.session_state["api_url"] = api_url
        st.session_state["api_url_raw"] = raw_url

        # Check connection immediately on load
        if "health_ok" not in st.session_state:
            try:
                health_data = _client(api_url).health()
                st.session_state.update({"health_ok": True, "health_data": health_data})
            except Exception as exc:
                st.session_state.update({"health_ok": False, "health_error": str(exc)})

        # Render full-width status pill
        if st.session_state.get("health_ok"):
            h = st.session_state.get("health_data", {})
            mongo_ok = bool(h.get("mongodb"))
            sub_color = "var(--success)" if mongo_ok else "var(--warning)"
            sub_label = "Ready" if mongo_ok else "Degraded"
            st.markdown(
                f'<div class="ir-status ir-status-ok" style="margin-bottom: 10px; display: flex; align-items: center; justify-content: center; height: 40px;">{_icon("check", 14)}'
                f"<span>Connected</span>"
                f'<span class="ir-status-sub">· Mongo · '
                f'<b style="color:{sub_color}">{sub_label}</b></span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ir-status ir-status-bad" style="margin-bottom: 10px;">{_icon("alert", 14)}<span>Offline</span></div>',
                unsafe_allow_html=True,
            )
            if err := st.session_state.get("health_error"):
                st.caption(err)

        # Full-width text button
        if st.button(
            "Refresh Connection", icon=":material/refresh:", use_container_width=True
        ):
            try:
                health_data = _client(api_url).health()
                st.session_state.update({"health_ok": True, "health_data": health_data})
            except Exception as exc:
                st.session_state.update({"health_ok": False, "health_error": str(exc)})
            st.rerun()

        st.divider()

        # ── Dataset ──────────────────────────────────────────────────────
        st.markdown(_section_label("Dataset", "layers"), unsafe_allow_html=True)

        dataset_names: list[str] = []
        dataset_labels: dict[str, str] = {}
        try:
            for ds in _client(api_url).list_datasets():
                count = f"{ds.document_count:,}" if ds.document_count else "?"
                status = "" if ds.models_ready else "  (offline)"
                dataset_names.append(ds.name)
                dataset_labels[ds.name] = f"{ds.name}  ·  {count} docs{status}"
        except IRAPIError:
            dataset_names = ["webis-touche2020"]
            dataset_labels = {"webis-touche2020": "webis-touche2020 (offline)"}

        selected_dataset = st.selectbox(
            "dataset",
            options=dataset_names,
            format_func=lambda x: str(dataset_labels.get(x, x)),
            label_visibility="collapsed",
        )

        dataset: str = (
            selected_dataset
            if selected_dataset is not None
            else (dataset_names[0] if dataset_names else "")
        )

        st.divider()

        # ── Search configuration ─────────────────────────────────────────
        st.markdown(
            _section_label("Search Configuration", "sliders"), unsafe_allow_html=True
        )

        execution_mode: str = st.radio(
            "Execution mode",
            options=["basic", "enhanced"],
            format_func=lambda x: (
                "Basic — core pipeline"
                if x == "basic"
                else "Enhanced — spell-check, synonyms, history"
            ),
        )

        retrieval_mode: str = st.selectbox(
            "Retrieval model",
            options=list(RETRIEVAL_MODES),
            format_func=lambda x: RETRIEVAL_MODES[x],
            index=list(RETRIEVAL_MODES).index("hybrid_parallel"),
        )

        # Hybrid sub-controls — only rendered for hybrid modes
        is_hybrid = retrieval_mode in ("hybrid_parallel", "hybrid_serial")
        sparse_method = "bm25"
        dense_method = "sbert"
        alpha = 0.5
        cascade_top_n = 200

        if is_hybrid:
            st.markdown(_eyebrow("Hybrid Components", "venn"), unsafe_allow_html=True)
            col_sp, col_de = st.columns(2)
            with col_sp:
                sparse_method = st.selectbox(
                    "Sparse method",
                    options=["bm25", "tfidf"],
                    format_func=str.upper,
                )
            with col_de:
                dense_method = st.selectbox(
                    "Dense method",
                    options=["sbert", "word2vec"],
                    format_func=lambda x: "SBERT" if x == "sbert" else "Word2Vec",
                )

            if retrieval_mode == "hybrid_parallel":
                alpha = st.slider(
                    "Alpha  (sparse ←→ dense)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="0.0 = pure dense  ·  1.0 = pure sparse",
                )
            else:  # hybrid_serial
                cascade_top_n = st.slider(
                    "Cascade top-N",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    step=10,
                    help="BM25 candidates forwarded to the dense reranker.",
                )

        # BM25 parameters — only shown when BM25 is active in the pipeline
        show_bm25_params = retrieval_mode == "bm25" or (
            is_hybrid and sparse_method == "bm25"
        )
        bm25_k1 = 1.5
        bm25_b = 0.75

        if show_bm25_params:
            st.markdown(_eyebrow("BM25 Parameters", "target"), unsafe_allow_html=True)
            bm25_k1 = st.slider(
                "k₁  — TF saturation",
                min_value=0.1,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Controls TF saturation speed. Typical range: 1.2 – 2.0.",
            )
            bm25_b = st.slider(
                "b  — length normalisation",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="0 = no normalisation  ·  1 = full normalisation. Typical: 0.75.",
            )

        st.divider()

        # ── Result controls ──────────────────────────────────────────────
        st.markdown(_section_label("Result Controls", "list"), unsafe_allow_html=True)

        top_k: int = st.slider("Top-K results", min_value=1, max_value=50, value=10)
        preview_chars: int = st.slider(
            "Inline preview (chars)",
            min_value=0,
            max_value=2000,
            value=500,
            step=50,
            help=(
                "Characters shown inline per result. "
                "Set to 0 to hide previews. "
                "Full original text is always available via the panel — "
                "moving this slider never triggers a new search."
            ),
        )

    return {
        "api_url": api_url,
        "dataset": dataset,
        "execution_mode": execution_mode,
        "retrieval_mode": retrieval_mode,
        "sparse_method": sparse_method,
        "dense_method": dense_method,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "alpha": alpha,
        "cascade_top_n": cascade_top_n,
        "top_k": top_k,
        "preview_chars": preview_chars,
    }


# ─────────────────────────────────────── Main ───────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="IR Search Engine 2026",
        page_icon=":material/manage_search:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    cfg = render_sidebar()

    # ── Page header ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="ir-page-header">'
        f'<div class="ir-brand-badge">{_icon("search", 22)}</div>'
        "<div>"
        '<div class="ir-page-title">Information Retrieval Search Engine</div>'
        '<div class="ir-page-subtitle">Damascus University · IR 2026 Lab</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    mode_cls, mode_icon, _family = _mode_theme(cfg["retrieval_mode"])
    exec_icon = "zap" if cfg["execution_mode"] == "basic" else "sparkle"
    st.markdown(
        '<div class="ir-chip-row">'
        f'<span class="ir-chip">{_icon("layers", 12)}Dataset <b>{html.escape(cfg["dataset"])}</b></span>'
        f'<span class="ir-chip {mode_cls}">{_icon(mode_icon, 12)}'
        f'<b>{html.escape(RETRIEVAL_MODES[cfg["retrieval_mode"]])}</b></span>'
        f'<span class="ir-chip">{_icon(exec_icon, 12)}Mode <b>{cfg["execution_mode"]}</b></span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Search form (st.form makes Enter key submit the query) ────────────
    with st.form("search_form", clear_on_submit=False):
        q_col, btn_col = st.columns([6, 1])
        with q_col:
            query: str = (
                st.text_input(
                    "query",
                    value=st.session_state.pop("prefill_query", ""),
                    placeholder="Enter your query and press Enter or click Search…",
                    label_visibility="collapsed",
                )
                or ""
            )
        with btn_col:
            submitted: bool = st.form_submit_button(
                "Search",
                icon=":material/search:",
                type="primary",
                use_container_width=True,
            )

    # ── Autocomplete suggestions (enhanced mode, shown below the form) ────
    if cfg["execution_mode"] == "enhanced" and query and len(query) >= 2:
        try:
            suggestions = _client(cfg["api_url"]).get_suggestions(query, limit=5)
            if suggestions:
                st.markdown(
                    _eyebrow("Suggestions from search history", "clock"),
                    unsafe_allow_html=True,
                )
                sug_cols = st.columns(min(len(suggestions), 5))
                for idx, sug in enumerate(suggestions):
                    with sug_cols[idx % 5]:
                        if st.button(
                            sug,
                            key=f"sug_{idx}",
                            icon=":material/history:",
                            use_container_width=True,
                        ):
                            st.session_state["prefill_query"] = sug
                            st.rerun()
        except IRAPIError:
            pass  # suggestions are non-critical; never let them crash the UI

    # ── Execute search ────────────────────────────────────────────────────
    if submitted:
        if not query.strip():
            st.warning("Please enter a search query.", icon=":material/edit_note:")
        else:
            with st.spinner("Retrieving documents…"):
                t_start = time.perf_counter()
                try:
                    response = _client(cfg["api_url"]).search(
                        dataset=cfg["dataset"],
                        query=query,
                        execution_mode=cfg["execution_mode"],
                        retrieval_mode=cfg["retrieval_mode"],
                        sparse_method=cfg["sparse_method"],
                        dense_method=cfg["dense_method"],
                        bm25_k1=cfg["bm25_k1"],
                        bm25_b=cfg["bm25_b"],
                        alpha=cfg["alpha"],
                        cascade_top_n=cfg["cascade_top_n"],
                        top_k=cfg["top_k"],
                        include_snippet_chars=_FETCH_SNIPPET_CHARS,
                    )
                    st.session_state["response"] = response
                    st.session_state["elapsed"] = time.perf_counter() - t_start
                except IRAPIError as exc:
                    st.error(f"**Search failed:** {exc}", icon=":material/error:")
                    st.session_state.pop("response", None)
                except Exception as exc:
                    st.error(
                        f"**Unexpected error:** {exc}", icon=":material/bug_report:"
                    )
                    st.session_state.pop("response", None)

    # ── Render stored results ─────────────────────────────────────────────
    response: SearchResponse | None = st.session_state.get("response")
    if response is None:
        return

    elapsed: float = st.session_state.get("elapsed", 0.0)
    # Use the *current* sidebar value so moving the preview slider
    # updates all cards live without triggering a new search.
    preview_chars: int = cfg["preview_chars"]

    st.divider()
    render_query_processing(response.query_processing)

    res_mode_cls, res_mode_icon, _ = _mode_theme(response.retrieval_mode)
    res_exec_icon = "zap" if response.execution_mode == "basic" else "sparkle"
    st.markdown(
        '<div class="ir-chip-row">'
        f'<span class="ir-chip">{_icon("bars", 12)}<b>{response.total_results}</b> result(s)</span>'
        f'<span class="ir-chip">{_icon("clock", 12)}<b>{elapsed:.3f}s</b></span>'
        f'<span class="ir-chip {res_mode_cls}">{_icon(res_mode_icon, 12)}'
        f"<b>{html.escape(RETRIEVAL_MODES.get(response.retrieval_mode, response.retrieval_mode))}</b></span>"
        f'<span class="ir-chip">{_icon(res_exec_icon, 12)}<b>{response.execution_mode}</b></span>'
        "</div>",
        unsafe_allow_html=True,
    )

    if not response.results:
        st.info(
            "No documents returned. Try rephrasing or switching the retrieval model.",
            icon=":material/search_off:",
        )
        return

    # Pin max to 1.0 for dense models (absolute %), but allow scaling for BM25 scores > 1.0
    actual_max = max((r.score for r in response.results), default=1.0)
    max_score = max(1.0, actual_max)
    for item in response.results:
        render_result_card(item, preview_chars, max_score)


if __name__ == "__main__":
    main()
