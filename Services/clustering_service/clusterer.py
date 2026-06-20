# Services/clustering_service/clusterer.py
"""
ClusterManager — runtime interface to pre-computed document clusters.

This class does NO computation at query time. It loads the three files
written by step10_cluster.py exactly once (on first request), holds them
in memory, and answers lookups in O(1).

Files consumed (all written by step10_cluster.py):
  data/models/clusters_{dataset}/clusters.json        — cluster summaries
  data/models/clusters_{dataset}/doc_cluster_map.pkl  — {doc_id: cluster_id}
  data/models/clusters_{dataset}/scatter_2d.json      — 2D sample points

Public API (used by the gateway):
  manager.load(dataset_name, cluster_dir)
  manager.is_ready() -> bool
  manager.get_summary() -> dict
  manager.get_all_clusters() -> list[dict]
  manager.get_cluster_info(cluster_id) -> dict | None
  manager.get_cluster_for_doc(doc_id) -> int | None
  manager.annotate_doc_ids(doc_ids) -> {doc_id: cluster_id | None}
  manager.get_scatter_data() -> list[dict]
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional


class ClusterManager:

    def __init__(self):
        self._summary: Optional[dict] = None  # clusters.json content
        self._doc_map: Optional[Dict[str, int]] = None  # {doc_id → cluster_id}
        self._scatter: Optional[List[dict]] = None  # scatter_2d.json content
        self._dataset: Optional[str] = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, dataset_name: str, cluster_dir: Path) -> None:
        """
        Load all three cluster files into memory.
        Raises FileNotFoundError if step10_cluster.py hasn't been run yet.
        """
        cluster_dir = Path(cluster_dir)

        clusters_path = cluster_dir / "clusters.json"
        doc_map_path = cluster_dir / "doc_cluster_map.pkl"
        scatter_path = cluster_dir / "scatter_2d.json"

        if not clusters_path.exists():
            raise FileNotFoundError(
                f"clusters.json not found at {clusters_path}.\n"
                f"Run:  python offline/step10_cluster.py"
            )

        print(f"[ClusterManager] Loading clusters for '{dataset_name}' ...")

        with open(clusters_path, encoding="utf-8") as f:
            self._summary = json.load(f)

        with open(doc_map_path, "rb") as f:
            self._doc_map = pickle.load(f)

        with open(scatter_path, encoding="utf-8") as f:
            self._scatter = json.load(f)

        self._dataset = dataset_name

        # Narrow Optional -> concrete types for the type checker; these are
        # guaranteed non-None immediately after the assignments above.
        assert self._summary is not None
        assert self._scatter is not None

        print(
            f"[ClusterManager] Ready — "
            f"{self._summary['n_clusters']} clusters, "
            f"{self._summary['n_docs']:,} docs, "
            f"{len(self._scatter):,} scatter points"
        )

    def is_ready(self) -> bool:
        return self._summary is not None

    # ── Cluster metadata ──────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """High-level metadata: dataset, n_clusters, n_docs, generated_at."""
        self._require_loaded()
        assert self._summary is not None
        return {
            "dataset": self._summary["dataset"],
            "n_clusters": self._summary["n_clusters"],
            "n_docs": self._summary["n_docs"],
            "embedding_source": self._summary["embedding_source"],
            "generated_at": self._summary["generated_at"],
        }

    def get_all_clusters(self) -> List[dict]:
        """
        All cluster summaries WITHOUT centroids.
        Centroids are 384-float arrays per cluster — useful internally
        but not needed by the UI and would bloat the API response.
        """
        self._require_loaded()
        assert self._summary is not None
        return [
            {
                "id": c["id"],
                "size": c["size"],
                "pct": c["pct"],
                "top_terms": c["top_terms"],
                "representative_doc_ids": c["representative_doc_ids"],
            }
            for c in self._summary["clusters"]
        ]

    def get_cluster_info(self, cluster_id: int) -> Optional[dict]:
        """Single cluster summary by id. Returns None if id is out of range."""
        self._require_loaded()
        assert self._summary is not None
        for c in self._summary["clusters"]:
            if c["id"] == cluster_id:
                return {
                    "id": c["id"],
                    "size": c["size"],
                    "pct": c["pct"],
                    "top_terms": c["top_terms"],
                    "representative_doc_ids": c["representative_doc_ids"],
                }
        return None

    # ── Per-document lookup ───────────────────────────────────────────────────

    def get_cluster_for_doc(self, doc_id: str) -> Optional[int]:
        """O(1) lookup: which cluster does this document belong to?"""
        self._require_loaded()
        assert self._doc_map is not None
        return self._doc_map.get(doc_id)

    def annotate_doc_ids(self, doc_ids: List[str]) -> Dict[str, Optional[int]]:
        """
        Given a list of doc_ids (e.g. search result doc_ids), return their
        cluster assignments. Used by the gateway to annotate search results
        with cluster membership so the UI can colour-code them.
        """
        self._require_loaded()
        assert self._doc_map is not None
        return {did: self._doc_map.get(did) for did in doc_ids}

    # ── Scatter data ──────────────────────────────────────────────────────────

    def get_scatter_data(self) -> List[dict]:
        """
        Returns the 2D scatter sample:
          [{"doc_id": "...", "x": 1.23, "y": -4.56, "cluster_id": 3}, ...]

        This is a ~10K-point stratified sample of the corpus.
        The UI renders it as a coloured scatter plot.
        """
        self._require_loaded()
        assert self._scatter is not None
        return self._scatter

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self.is_ready():
            raise RuntimeError(
                "ClusterManager not loaded. Call load() first, "
                "or run step10_cluster.py to generate the cluster files."
            )
