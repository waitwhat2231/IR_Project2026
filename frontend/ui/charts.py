"""
Plotly chart builders. This is the only module that imports
``plotly.graph_objects`` directly — every other module that needs a chart
calls a function here instead of touching ``go`` itself.

Plotly is an optional dependency: callers must check ``_PLOTLY`` before
calling any builder below (the app already shows a
"Plotly not installed" warning instead of calling these when it's False).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

try:
    import plotly.graph_objects as go  # noqa: F811

    _PLOTLY = True
except ImportError:
    _PLOTLY = False

from frontend.api_client import (
    ClusterScatterResponse,
    EvaluationResponse,
    ModelEvaluation,
)
from frontend.ui.components import _cluster_color, _model_color


def _plotly_layout(height: int = 320) -> dict:
    """Shared Plotly layout settings matching the app's dark theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19,24,38,0.6)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#99a2b8", size=12),
        height=height,
        margin=dict(l=12, r=12, t=36, b=12),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showline=False,
            tickfont=dict(size=11, color="#5b6378"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            showline=False,
            tickfont=dict(size=11, color="#5b6378"),
        ),
        legend=dict(
            bgcolor="rgba(19,24,38,0.9)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#131826",
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(family="JetBrains Mono, monospace", size=12),
        ),
    )


def _metric_bar_chart(
    models: list[ModelEvaluation],
    metric_key: str,
    metric_label: str,
) -> "go.Figure":
    """Single grouped bar chart for one metric across all models."""
    names = [m.name for m in models]
    values = [round(m.aggregate.get(metric_key, 0.0), 4) for m in models]
    colors = [_model_color(m.name) for m in models]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker=dict(
                color=colors,
                line=dict(width=0),
                opacity=0.88,
            ),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, family="JetBrains Mono, monospace", color="#edf0f7"),
            hovertemplate="<b>%{x}</b><br>"
            + metric_label
            + ": %{y:.4f}<extra></extra>",
        )
    )
    layout = _plotly_layout(height=280)
    layout["title"] = dict(
        text=metric_label, font=dict(size=13, color="#edf0f7"), x=0.02
    )
    layout["yaxis"]["range"] = [0, max(values) * 1.28 + 0.01] if values else [0, 1]
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def _compare_bar_chart(
    baseline: EvaluationResponse,
    enhanced: EvaluationResponse,
    metric_key: str,
    metric_label: str,
) -> "go.Figure":
    """Grouped bar chart: baseline vs enhanced side by side per model."""
    # Use only models present in both phases
    base_map = {m.name: m.aggregate.get(metric_key, 0.0) for m in baseline.models}
    enh_map = {m.name: m.aggregate.get(metric_key, 0.0) for m in enhanced.models}
    all_names = list(base_map.keys())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=all_names,
            y=[base_map.get(n, 0.0) for n in all_names],
            marker=dict(color="#5b6378", opacity=0.85, line=dict(width=0)),
            text=[f"{base_map.get(n,0):.3f}" for n in all_names],
            textposition="outside",
            textfont=dict(size=10, color="#99a2b8"),
            hovertemplate="Baseline<br>%{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Enhanced",
            x=all_names,
            y=[enh_map.get(n, 0.0) for n in all_names],
            marker=dict(color="#818cf8", opacity=0.90, line=dict(width=0)),
            text=[f"{enh_map.get(n,0):.3f}" for n in all_names],
            textposition="outside",
            textfont=dict(size=10, color="#edf0f7"),
            hovertemplate="Enhanced<br>%{x}: %{y:.4f}<extra></extra>",
        )
    )
    layout = _plotly_layout(height=300)
    layout["title"] = dict(
        text=metric_label, font=dict(size=13, color="#edf0f7"), x=0.02
    )
    layout["barmode"] = "group"
    layout["bargap"] = 0.22
    layout["bargroupgap"] = 0.06
    layout["legend"]["orientation"] = "h"
    layout["legend"]["y"] = -0.18
    fig.update_layout(**layout)
    return fig


def _scatter_chart(scatter: ClusterScatterResponse) -> "go.Figure":
    """2D UMAP scatter coloured by cluster_id."""
    cluster_infos = {c.id: c for c in scatter.clusters}

    # Group points by cluster_id
    by_cluster: dict[int, list] = {}
    for pt in scatter.points:
        by_cluster.setdefault(pt.cluster_id, []).append(pt)

    fig = go.Figure()
    for cid, pts in sorted(by_cluster.items()):
        info = cluster_infos.get(cid)
        top = ", ".join(info.top_terms[:4]) if info else f"Cluster {cid}"
        color = _cluster_color(cid)
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in pts],
                y=[p.y for p in pts],
                mode="markers",
                name=f"C{cid}: {top}",
                marker=dict(color=color, size=3.5, opacity=0.72, line=dict(width=0)),
                customdata=[p.doc_id for p in pts],
                hovertemplate=(
                    f"<b>Cluster {cid}</b><br>"
                    f"{top}<br>"
                    "doc: %{customdata}<extra></extra>"
                ),
            )
        )

    layout = _plotly_layout(height=580)
    layout["xaxis"].update(showgrid=False, showticklabels=False, zeroline=False)
    layout["yaxis"].update(showgrid=False, showticklabels=False, zeroline=False)
    layout["title"] = dict(
        text=f"Corpus Map — {scatter.n_points:,} documents, {scatter.n_clusters} clusters",
        font=dict(size=13, color="#edf0f7"),
        x=0.02,
    )
    layout["plot_bgcolor"] = "rgba(10,12,18,0.95)"
    layout["legend"]["font"]["size"] = 10
    fig.update_layout(**layout)
    return fig


def _per_query_bar_chart(
    query_ids: list[str],
    values: list[float],
    model_name: str,
    metric_label: str,
) -> "go.Figure":
    """Per-query metric breakdown for a single model (Evaluation tab expander)."""
    fig = go.Figure(
        go.Bar(
            x=query_ids,
            y=values,
            marker=dict(
                color=_model_color(model_name), opacity=0.85, line=dict(width=0)
            ),
            hovertemplate="Query %{x}<br>" + metric_label + ": %{y:.4f}<extra></extra>",
        )
    )
    layout = _plotly_layout(height=260)
    layout["title"] = dict(
        text=f"{model_name} — {metric_label} per query",
        font=dict(size=12, color="#edf0f7"),
        x=0.01,
    )
    layout["showlegend"] = False
    layout["xaxis"]["tickangle"] = -35
    fig.update_layout(**layout)
    return fig
