"""Evaluation tab: per-phase metric charts, baseline-vs-enhanced comparison, and delta table."""

from __future__ import annotations

import html

import streamlit as st

from frontend.api_client import (
    EvaluationCompareResponse,
    EvaluationResponse,
    IRAPIError,
)
from frontend.ui.charts import (
    _PLOTLY,
    _compare_bar_chart,
    _metric_bar_chart,
    _per_query_bar_chart,
)
from frontend.ui.components import _metric_label, _model_color
from frontend.ui.helpers import _client
from frontend.ui.icons import _eyebrow


def _render_eval_phase(data: EvaluationResponse) -> None:
    """Charts + stats for a single evaluation phase."""
    if not data.models:
        st.info(
            "No model results found in this evaluation file.", icon=":material/info:"
        )
        return

    # ── Summary stat cards (best model per primary metric) ──────────────
    primary_metrics = ["map", "ndcg", "P_10", "recall"]
    available = [
        k for k in primary_metrics if any(k in m.aggregate for m in data.models)
    ]

    if available:
        cols = st.columns(len(available))
        for col, key in zip(cols, available):
            best = max(data.models, key=lambda m: m.aggregate.get(key, 0.0))
            with col:
                st.markdown(
                    f'<div class="ir-stat-card">'
                    f'<div class="ir-stat-label">{_metric_label(key)}</div>'
                    f'<div class="ir-stat-value">{best.aggregate.get(key, 0):.3f}</div>'
                    f'<div class="ir-stat-sub">best: <b>{best.name}</b></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── 2×2 metric charts ───────────────────────────────────────────────
    if not _PLOTLY:
        st.warning(
            "Plotly not installed. Run `pip install plotly` to see charts.",
            icon=":material/bar_chart:",
        )
        return

    all_metric_keys = sorted(
        {k for m in data.models for k in m.aggregate},
        key=lambda k: primary_metrics.index(k) if k in primary_metrics else 99,
    )

    # Render in a 2-column grid
    cols = st.columns(2)
    for i, key in enumerate(all_metric_keys):
        with cols[i % 2]:
            fig = _metric_bar_chart(data.models, key, _metric_label(key))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Per-query breakdown ──────────────────────────────────────────────
    models_with_pq = [m for m in data.models if m.per_query]
    if models_with_pq:
        with st.expander("Per-query breakdown", icon=":material/table_rows:"):
            selected_model = st.selectbox(
                "Model",
                options=[m.name for m in models_with_pq],
                key=f"pq_model_{data.phase}",
            )
            sel = next((m for m in models_with_pq if m.name == selected_model), None)
            if sel and sel.per_query:
                sel_metric = st.selectbox(
                    "Metric",
                    options=all_metric_keys,
                    format_func=_metric_label,
                    key=f"pq_metric_{data.phase}",
                )
                query_ids = sorted(sel.per_query.keys())
                q_vals = [sel.per_query[q].get(sel_metric, 0.0) for q in query_ids]
                pq_fig = _per_query_bar_chart(
                    query_ids, q_vals, selected_model, _metric_label(sel_metric)
                )
                st.plotly_chart(
                    pq_fig, width="stretch", config={"displayModeBar": False}
                )


def _render_eval_compare(data: EvaluationCompareResponse) -> None:
    """Side-by-side baseline vs enhanced charts + delta table."""
    if data.baseline is None and data.enhanced is None:
        st.info(
            "No evaluation data available for either phase.", icon=":material/info:"
        )
        return

    if data.baseline is None or data.enhanced is None:
        available = data.baseline or data.enhanced
        st.warning(
            f"Only **{'baseline' if data.baseline else 'enhanced'}** phase has results. "
            "Showing single-phase view.",
            icon=":material/warning:",
        )
        _render_eval_phase(available)  # type: ignore[arg-type]
        return

    if not _PLOTLY:
        st.warning("Plotly not installed. Run `pip install plotly` to see charts.")
        return

    # ── Comparison charts ────────────────────────────────────────────────
    primary_metrics = ["map", "ndcg", "P_10", "recall"]
    all_keys = sorted(
        {k for m in data.baseline.models for k in m.aggregate},
        key=lambda k: primary_metrics.index(k) if k in primary_metrics else 99,
    )

    cols = st.columns(2)
    for i, key in enumerate(all_keys):
        with cols[i % 2]:
            fig = _compare_bar_chart(
                data.baseline, data.enhanced, key, _metric_label(key)
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── Delta table ──────────────────────────────────────────────────────
    with st.expander(
        "Δ Delta table — baseline vs enhanced", icon=":material/compare_arrows:"
    ):
        base_map = {m.name: m.aggregate for m in data.baseline.models}
        enh_map = {m.name: m.aggregate for m in data.enhanced.models}
        rows_html = ""
        for model_name in base_map:
            for key in all_keys:
                b = base_map[model_name].get(key, 0.0)
                e = enh_map.get(model_name, {}).get(key, 0.0)
                delta = e - b
                if abs(delta) < 1e-6:
                    delta_html = '<span class="ir-delta-neu">—</span>'
                elif delta > 0:
                    delta_html = f'<span class="ir-delta-pos">+{delta:.4f}</span>'
                else:
                    delta_html = f'<span class="ir-delta-neg">{delta:.4f}</span>'
                rows_html += (
                    f"<tr><td>{html.escape(model_name)}</td>"
                    f"<td>{_metric_label(key)}</td>"
                    f"<td>{b:.4f}</td><td>{e:.4f}</td>"
                    f"<td>{delta_html}</td></tr>"
                )
        st.markdown(
            '<table style="width:100%;font-size:13px;border-collapse:collapse">'
            "<thead><tr>"
            "<th style='text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,0.1)'>Model</th>"
            "<th style='text-align:left;padding:8px;border-bottom:1px solid rgba(255,255,255,0.1)'>Metric</th>"
            "<th style='text-align:right;padding:8px;border-bottom:1px solid rgba(255,255,255,0.1)'>Baseline</th>"
            "<th style='text-align:right;padding:8px;border-bottom:1px solid rgba(255,255,255,0.1)'>Enhanced</th>"
            "<th style='text-align:right;padding:8px;border-bottom:1px solid rgba(255,255,255,0.1)'>Δ</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )


def render_evaluation_tab(dataset: str, api_url: str) -> None:
    """Full Evaluation tab: phase selector → load → charts."""
    st.markdown(
        _eyebrow("Model Performance Evaluation", "bars"),
        unsafe_allow_html=True,
    )

    col_phase, col_load = st.columns([4, 1])
    with col_phase:
        phase: str = st.radio(
            "phase",
            options=["baseline", "enhanced", "compare"],
            format_func=lambda x: {
                "baseline": "Baseline  (core pipeline only)",
                "enhanced": "Enhanced  (+ spell, synonyms, history)",
                "compare": "Compare  (baseline vs enhanced)",
            }[x],
            horizontal=True,
            label_visibility="collapsed",
        )
    with col_load:
        load_eval = st.button(
            "Load",
            icon=":material/analytics:",
            type="primary",
            width="stretch",
            key="load_eval_btn",
        )

    session_key = f"eval_{dataset}_{phase}"

    if load_eval:
        with st.spinner("Fetching evaluation results…"):
            try:
                client = _client(api_url)
                if phase == "compare":
                    result = client.get_evaluation_compare(
                        dataset, include_per_query=True
                    )
                else:
                    result = client.get_evaluation(
                        dataset, phase=phase, include_per_query=True
                    )
                st.session_state[session_key] = result
            except IRAPIError as exc:
                st.error(f"Could not load evaluation: {exc}", icon=":material/error:")
                return

    data = st.session_state.get(session_key)
    if data is None:
        st.info(
            "Click **Load** to fetch evaluation results for this dataset.",
            icon=":material/analytics:",
        )
        return

    if phase == "compare":
        _render_eval_compare(data)
    else:
        _render_eval_phase(data)
