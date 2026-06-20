"""
Reusable Streamlit-rendering components: the query-insights banner and
individual search-result cards, plus small colour/label lookup helpers
used by both the components above and the chart builders.
"""

from __future__ import annotations

import html

import streamlit as st

from frontend.api_client import QueryProcessingInfo, SearchResultItem
from frontend.ui.constants import _CLUSTER_PALETTE, _METRIC_LABELS, _MODEL_COLORS
from frontend.ui.helpers import _clean_display_text, _expansion_chips, _token_chips
from frontend.ui.icons import _icon


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
    )

    if item.cluster_id is not None:
        c_color = _cluster_color(item.cluster_id)
        card_html += (
            f'<span class="ir-cluster-badge" '
            f'style="color:{c_color};border-color:{c_color}40;background:{c_color}12">'
            f"C{item.cluster_id}</span>"
        )

    card_html += (
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


def _cluster_color(cluster_id: int) -> str:
    return _CLUSTER_PALETTE[cluster_id % len(_CLUSTER_PALETTE)]


def _model_color(model_name: str) -> str:
    """Map a model name to a consistent chart colour."""
    key = model_name.lower().replace(" ", "_").replace("-", "_")
    for fragment, color in _MODEL_COLORS.items():
        if fragment in key:
            return color
    # Fallback: pick from cluster palette based on hash
    return _CLUSTER_PALETTE[abs(hash(model_name)) % len(_CLUSTER_PALETTE)]


def _metric_label(key: str) -> str:
    return _METRIC_LABELS.get(key, key.upper())
