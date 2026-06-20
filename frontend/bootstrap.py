"""
Full-screen blocking loader shown exactly once per browser session, while
the API gateway loads retrieval models into memory. Also handles the
connection-error recovery screen.
"""

from __future__ import annotations

import html

import streamlit as st

from frontend.api_client import IRAPIError
from frontend.ui.helpers import _client
from frontend.ui.styles import _CSS

_BOOTSTRAP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@700;800&family=JetBrains+Mono:wght@400&display=swap');

/* ── Hide every native Streamlit chrome element during bootstrap ── */
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSpinner"],
footer { display: none !important; }

/* Kill default padding so the fixed overlay really fills the viewport */
.block-container,
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100vw !important;
}

/* ── Full-screen overlay ── */
.ir-boot-wrap {
    position: fixed;
    inset: 0;
    background: #0a0c12;
    z-index: 2147483647;          /* max int32 — above everything */
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 22px;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}
.ir-boot-logo {
    width: 72px; height: 72px;
    background: linear-gradient(135deg, #6366f1 0%, #38d9e8 100%);
    border-radius: 22px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 56px -6px rgba(99,102,241,0.65);
}
.ir-boot-title {
    font-family: 'Sora', 'Plus Jakarta Sans', sans-serif;
    font-size: 24px; font-weight: 800;
    color: #edf0f7; text-align: center; margin-bottom: 4px;
}
.ir-boot-sub { font-size: 13px; color: #5b6378; text-align: center; }
.ir-boot-ring {
    width: 44px; height: 44px;
    border: 3px solid rgba(99,102,241,0.14);
    border-top-color: #818cf8;
    border-radius: 50%;
    animation: ir-boot-spin 0.8s linear infinite;
}
@keyframes ir-boot-spin { to { transform: rotate(360deg); } }
.ir-boot-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #5b6378; text-align: center;
}
.ir-boot-hint {
    font-size: 12px; color: #3a3f52;
    text-align: center; max-width: 360px; line-height: 1.7;
}
</style>
"""

_BOOTSTRAP_HTML = """
<div class="ir-boot-wrap">
    <div class="ir-boot-logo">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
             stroke="white" stroke-width="2.5"
             stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
    </div>
    <div>
        <div class="ir-boot-title">IR Search Engine 2026</div>
        <div class="ir-boot-sub">Damascus University &nbsp;·&nbsp; IR 2026 Lab</div>
    </div>
    <div class="ir-boot-ring"></div>
    <div class="ir-boot-status">Loading retrieval models into memory…</div>
    <div class="ir-boot-hint">
        TF-IDF &nbsp;·&nbsp; SBERT &nbsp;·&nbsp; BM25 &nbsp;·&nbsp; Word2Vec
        are being initialised.<br>
        This happens <b style="color:#818cf8">once per session</b>.
        Please wait.
    </div>
</div>
"""


def _render_bootstrap(api_url: str) -> None:
    """
    Full-screen blocking loader — shown exactly once per browser session.

    Hides all Streamlit chrome (sidebar, header, toolbar) via CSS, renders a
    branded overlay, then calls ``/api/v1/datasets`` which triggers server-side
    model loading.  On success the result is stored in session_state and
    ``st.rerun()`` transitions into the normal app.  On failure, a recovery
    screen with a Retry button is shown.

    This function always ends with either ``st.rerun()`` or ``st.stop()`` —
    nothing after the call in ``main()`` will execute.
    """

    # ── Error-recovery state ─────────────────────────────────────────────
    if "datasets_error" in st.session_state:
        st.markdown(_CSS, unsafe_allow_html=True)
        st.markdown("<br>" * 8, unsafe_allow_html=True)
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown(
                '<div style="text-align:center;margin-bottom:20px">'
                '<div style="font-size:40px;margin-bottom:14px">⚠️</div>'
                '<div style="font-family:Sora,sans-serif;font-size:20px;font-weight:800;'
                'color:#edf0f7;margin-bottom:6px">Cannot reach API Gateway</div>'
                f'<div style="font-family:JetBrains Mono,monospace;font-size:12px;'
                f'color:#5b6378;margin-bottom:18px">{html.escape(api_url)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.error(st.session_state["datasets_error"])
            st.caption(
                "Make sure the gateway is running:  "
                "`uvicorn Services.gateway.main:app --host 0.0.0.0 --port 8000 --reload`"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("↺  Retry", type="primary", width="stretch"):
                    st.session_state.pop("datasets_error", None)
                    st.rerun()
            with col_b:
                if st.button("⚙  Change URL", width="stretch"):
                    # Let the user edit the URL by skipping bootstrap this once.
                    st.session_state.pop("datasets_error", None)
                    st.session_state["force_url_edit"] = True
                    st.rerun()
        st.stop()
        return

    # ── Loading state ────────────────────────────────────────────────────
    # Inject the overlay before entering the blocking call so Streamlit's
    # first render flush includes it (the spinner widget flushes output).
    st.markdown(_BOOTSTRAP_CSS + _BOOTSTRAP_HTML, unsafe_allow_html=True)

    with st.spinner(""):  # empty label — the overlay provides all visible text
        try:
            datasets = _client(api_url).list_datasets()
            st.session_state["datasets"] = datasets

            # Pre-warm health check while the connection is already open
            try:
                health = _client(api_url).health()
                st.session_state.update({"health_ok": True, "health_data": health})
            except Exception:
                st.session_state["health_ok"] = False

        except IRAPIError as exc:
            st.session_state["datasets_error"] = str(exc)

    # Success → normal app.  Error → next rerun shows error-recovery state.
    st.rerun()
