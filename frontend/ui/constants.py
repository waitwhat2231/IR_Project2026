"""
Shared constants: API defaults, retrieval-mode labels, colour palettes,
and metric-key → display-label mappings.
"""

from __future__ import annotations

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

# Cluster colour palette — 20 distinct colours matching the dark theme.
# Index into this list with cluster_id % len(_CLUSTER_PALETTE).
_CLUSTER_PALETTE: list[str] = [
    "#818cf8",
    "#38d9e8",
    "#e879f9",
    "#34d399",
    "#fbbf24",
    "#fb7185",
    "#60a5fa",
    "#a78bfa",
    "#f472b6",
    "#2dd4bf",
    "#fb923c",
    "#4ade80",
    "#22d3ee",
    "#c084fc",
    "#f87171",
    "#facc15",
    "#86efac",
    "#67e8f9",
    "#d8b4fe",
    "#fda4af",
]

# Human-readable labels for pytrec_eval metric keys
_METRIC_LABELS: dict[str, str] = {
    "map": "MAP",
    "ndcg": "nDCG",
    "ndcg_cut_10": "nDCG@10",
    "ndcg_cut_20": "nDCG@20",
    "recall": "Recall",
    "recall_100": "Recall@100",
    "P_10": "P@10",
    "P_5": "P@5",
    "P_20": "P@20",
}

# Fixed colour per model family — keeps charts consistent across phases
_MODEL_COLORS: dict[str, str] = {
    "bm25": "#818cf8",
    "tfidf": "#38d9e8",
    "sbert": "#e879f9",
    "word2vec": "#34d399",
    "hybrid_parallel": "#fbbf24",
    "hybrid_serial": "#fb7185",
}
