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
import sys
import time
from pathlib import Path

import streamlit as st

# make sibling package importable when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.api_client import IRAPIError, SearchResponse  # noqa: E402
from frontend.bootstrap import _render_bootstrap  # noqa: E402
from frontend.sidebar import render_sidebar  # noqa: E402
from frontend.tabs.clusters import render_clusters_tab  # noqa: E402
from frontend.tabs.evaluation import render_evaluation_tab  # noqa: E402
from frontend.ui.components import (
    render_query_processing,
    render_result_card,
)  # noqa: E402
from frontend.ui.constants import (
    DEFAULT_API_URL,
    RETRIEVAL_MODES,
    _FETCH_SNIPPET_CHARS,
)  # noqa: E402
from frontend.ui.helpers import _client, _mode_theme  # noqa: E402
from frontend.ui.icons import _eyebrow, _icon  # noqa: E402
from frontend.ui.styles import _CSS  # noqa: E402


def main() -> None:
    st.set_page_config(
        page_title="IR Search Engine 2026",
        page_icon=":material/manage_search:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Bootstrap gate ───────────────────────────────────────────────────
    # Read the API URL from session_state (or default) BEFORE rendering the
    # sidebar, so bootstrap can use the same URL the user last set.
    _boot_url: str = st.session_state.get("api_url_raw", DEFAULT_API_URL)

    # "datasets" in session_state means bootstrap already completed this session.
    # "force_url_edit" means the user clicked "Change URL" on the error screen —
    # let them into the sidebar to update the URL, then bootstrap again.
    if "datasets" not in st.session_state and "force_url_edit" not in st.session_state:
        # No datasets loaded yet — show blocking loader and stop rendering.
        _render_bootstrap(_boot_url)
        return  # _render_bootstrap always calls st.rerun() or st.stop() first,
        # but `return` keeps type-checkers happy.

    st.markdown(_CSS, unsafe_allow_html=True)

    cfg = render_sidebar()

    # If the user changed the API URL via the sidebar after a "Change URL" recovery,
    # trigger a fresh bootstrap with the new URL.
    if "force_url_edit" in st.session_state:
        st.session_state.pop("force_url_edit")
        # Clear stale datasets so the next rerun goes through bootstrap
        st.session_state.pop("datasets", None)
        st.session_state.pop("health_ok", None)
        st.rerun()
        return

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

    # ── Top-level tab navigation ─────────────────────────────────────────
    tab_search, tab_eval, tab_clusters = st.tabs(
        [
            ":material/search:  Search",
            ":material/analytics:  Evaluation",
            ":material/bubble_chart:  Clusters",
        ]
    )

    # ════════════════════════════════════════ SEARCH TAB ═════════════════
    with tab_search:
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

        # Search form (st.form makes Enter key submit the query)
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
                    width="stretch",
                )

        # Autocomplete suggestions (enhanced mode, shown below the form)
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
                                width="stretch",
                            ):
                                st.session_state["prefill_query"] = sug
                                st.rerun()
            except IRAPIError:
                pass  # suggestions are non-critical; never let them crash the UI

        # Execute search
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

        # Render stored results
        response: SearchResponse | None = st.session_state.get("response")
        if response is not None:
            elapsed: float = st.session_state.get("elapsed", 0.0)
            # Use current sidebar value — moving the slider updates cards live
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
            else:
                actual_max = max((r.score for r in response.results), default=1.0)
                max_score = max(1.0, actual_max)
                for item in response.results:
                    render_result_card(item, preview_chars, max_score)

    # ════════════════════════════════════════ EVALUATION TAB ═════════════
    with tab_eval:
        render_evaluation_tab(cfg["dataset"], cfg["api_url"])

    # ════════════════════════════════════════ CLUSTERS TAB ═══════════════
    with tab_clusters:
        render_clusters_tab(cfg["dataset"], cfg["api_url"])


if __name__ == "__main__":
    main()
