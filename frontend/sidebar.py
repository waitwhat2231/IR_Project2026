"""Sidebar: API gateway connection, dataset picker, and search configuration controls."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.ui.constants import DEFAULT_API_URL, RETRIEVAL_MODES
from frontend.ui.helpers import _client, _normalize_url
from frontend.ui.icons import _eyebrow, _icon, _section_label


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

        # If the user changed the URL, clear datasets so bootstrap re-runs.
        # This is the ONLY safe way to reload models — through the full-screen
        # blocking loader that prevents any other requests during loading.
        if st.session_state.get("api_url") != api_url:
            st.session_state.pop("health_ok", None)
            st.session_state.pop("datasets", None)  # triggers re-bootstrap
        st.session_state["api_url"] = api_url
        st.session_state["api_url_raw"] = raw_url

        # Health check is lightweight (no model loading) — safe on every rerun.
        # Bootstrap already pre-warmed it, so this only fires after the first load.
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
        if st.button("Refresh Connection", icon=":material/refresh:", width="stretch"):
            try:
                health_data = _client(api_url).health()
                st.session_state.update({"health_ok": True, "health_data": health_data})
            except Exception as exc:
                st.session_state.update({"health_ok": False, "health_error": str(exc)})
            st.rerun()

        st.divider()

        # ── Dataset ──────────────────────────────────────────────────────
        st.markdown(_section_label("Dataset", "layers"), unsafe_allow_html=True)

        # Read from session_state — list_datasets() was called ONCE at boot.
        # Never call it again here; doing so re-triggers model loading into RAM.
        _ds_list = st.session_state.get("datasets", [])
        dataset_names: list[str] = []
        dataset_labels: dict[str, str] = {}
        for ds in _ds_list:
            count = f"{ds.document_count:,}" if ds.document_count else "?"
            ready = "\u2705" if ds.models_ready else "\u26a0\ufe0f"
            dataset_names.append(ds.name)
            dataset_labels[ds.name] = f"{ready}  {ds.name}  \u00b7  {count} docs"
        if not dataset_names:
            dataset_names = ["webis-touche2020"]
            dataset_labels = {"webis-touche2020": "webis-touche2020"}

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
