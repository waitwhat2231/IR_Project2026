"""Clusters tab: corpus scatter plot + per-cluster summary cards."""

from __future__ import annotations

import html

import streamlit as st

from frontend.api_client import (
    ClusterInfo,
    ClusterScatterResponse,
    ClustersResponse,
    IRAPIError,
)
from frontend.ui.charts import _PLOTLY, _scatter_chart
from frontend.ui.components import _cluster_color
from frontend.ui.helpers import _client
from frontend.ui.icons import _eyebrow, _icon


def _render_cluster_card(cluster: ClusterInfo) -> None:
    """HTML card for one cluster."""
    color = _cluster_color(cluster.id)
    terms_html = "".join(
        f'<span class="ir-cluster-term">{html.escape(t)}</span>'
        for t in cluster.top_terms[:8]
    )
    st.markdown(
        f'<div class="ir-cluster-card" style="border-top-color:{color}">'
        f'<div class="ir-cluster-header">'
        f'<span class="ir-cluster-id" style="color:{color};background:rgba(0,0,0,0.0);'
        f'border-color:{color}40">Cluster {cluster.id}</span>'
        f'<span class="ir-cluster-pct">{cluster.pct:.1f}% · {cluster.size:,} docs</span>'
        f"</div>"
        f'<div class="ir-cluster-terms">{terms_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_clusters_tab(dataset: str, api_url: str) -> None:
    """Full Clusters tab: scatter plot + cluster cards grid."""
    st.markdown(
        _eyebrow("Document Clustering  —  SBERT + MiniBatchKMeans + UMAP", "cpu"),
        unsafe_allow_html=True,
    )

    col_load, col_scatter = st.columns([1, 3])
    with col_load:
        load_clusters = st.button(
            "Load Clusters",
            icon=":material/bubble_chart:",
            type="primary",
            width="stretch",
            key="load_clusters_btn",
        )
        load_scatter = st.button(
            "Load Scatter",
            icon=":material/scatter_plot:",
            width="stretch",
            key="load_scatter_btn",
        )

    if load_clusters or load_scatter:
        client = _client(api_url)

        if load_clusters:
            with st.spinner("Loading cluster summaries…"):
                try:
                    clusters_data = client.get_clusters(dataset)
                    st.session_state[f"clusters_{dataset}"] = clusters_data
                except IRAPIError as exc:
                    st.error(f"Could not load clusters: {exc}", icon=":material/error:")

        if load_scatter:
            with st.spinner(
                "Loading scatter data (~10K points)… this may take a moment."
            ):
                try:
                    scatter_data = client.get_cluster_scatter(dataset)
                    st.session_state[f"scatter_{dataset}"] = scatter_data
                except IRAPIError as exc:
                    st.error(
                        f"Could not load scatter data: {exc}", icon=":material/error:"
                    )

    clusters_data: ClustersResponse | None = st.session_state.get(f"clusters_{dataset}")
    scatter_data: ClusterScatterResponse | None = st.session_state.get(
        f"scatter_{dataset}"
    )

    if clusters_data is None and scatter_data is None:
        st.info(
            "Run `python offline/step10_cluster.py` first, then click **Load Clusters** or **Load Scatter**.",
            icon=":material/bubble_chart:",
        )
        return

    # ── Overview chips ───────────────────────────────────────────────────
    src = clusters_data or scatter_data
    if src:
        n_clusters = src.n_clusters
        n_docs = (
            src.n_docs
            if isinstance(src, ClustersResponse)
            else getattr(src, "n_points", 0)
        )
        st.markdown(
            '<div class="ir-chip-row">'
            f'<span class="ir-chip">{_icon("cpu", 12)}{n_clusters} clusters</span>'
            f'<span class="ir-chip">{_icon("doc", 12)}{n_docs:,} documents</span>'
            f'<span class="ir-chip ir-chip-cyan">{_icon("sparkle", 12)}SBERT embeddings · MiniBatchKMeans · UMAP</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Scatter plot ─────────────────────────────────────────────────────
    if scatter_data is not None:
        if not _PLOTLY:
            st.warning("Plotly not installed — cannot render scatter chart.")
        else:
            fig = _scatter_chart(scatter_data)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

    # ── Cluster cards grid ───────────────────────────────────────────────
    ref = clusters_data or scatter_data
    cluster_list = (
        clusters_data.clusters
        if clusters_data
        else (scatter_data.clusters if scatter_data else [])
    )

    if cluster_list:
        st.markdown(
            _eyebrow(
                f"All {len(cluster_list)} clusters — top discriminative terms", "list"
            ),
            unsafe_allow_html=True,
        )
        # 3-column grid
        cols = st.columns(3)
        for i, cluster in enumerate(sorted(cluster_list, key=lambda c: c.id)):
            with cols[i % 3]:
                _render_cluster_card(cluster)
