# Services/retrieval_service/bm25_retriever.py
"""
BM25 retriever built on top of InvertedIndexManager.

No model is trained here — all data lives in the InvertedIndexManager
that step3 already built and saved. BM25's only "parameters" are k1
and b, which are hyperparameters, not learned weights.

The save file stores: dataset_name, k1, b.
At load time the retriever re-attaches to the InvertedIndexManager
by name, which loads from data/indexes/{name}_inverted.pkl.

Save layout:
  {save_dir}/bm25_config.pkl   ← {dataset_name, k1, b}

BM25 scoring formula per query term t in document d:
  score(t,d) = IDF(t) * tf_norm(t,d)

  where:
  IDF(t)       = log((N - df + 0.5) / (df + 0.5) + 1)   [from InvertedIndexManager.idf()]
  tf_norm(t,d) = tf * (k1+1) / (tf + k1 * (1 - b + b * |d| / avgdl))
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.config import MODEL_DIR
from Services.indexing_service.inverted_index import InvertedIndexManager


class BM25Retriever:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        k1 (1.5): TF saturation.
          Controls how quickly additional occurrences of a term
          stop contributing to the score.
          k1=1.5 → a term at tf=10 scores ~2.5× more than tf=1,
          not 10×. This is the saturation ceiling effect.

        b (0.75): Length normalization.
          Penalizes long documents to prevent them from winning
          simply by being large.
          b=0.75 is the standard default for most collections.
          For long debate documents it prevents verbose posts from
          dominating over concise, focused ones.
        """
        self.k1: float = k1
        self.b: float = b
        self.dataset_name: Optional[str] = None
        self.idx: Optional[InvertedIndexManager] = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_index(self, dataset_name: str):
        """
        Attaches to the InvertedIndexManager saved by step3.
        This is the only setup BM25 needs — no separate training.
        """
        self.dataset_name = dataset_name
        print(f"  Loading InvertedIndexManager for '{dataset_name}'...")
        self.idx = InvertedIndexManager(dataset_name)

        if self.idx.N == 0:
            raise RuntimeError(
                f"InvertedIndexManager loaded but N=0. "
                f"Did step3 complete successfully for '{dataset_name}'?"
            )

        print(f"  BM25 ready:")
        print(f"    Documents : {self.idx.N:,}")
        print(f"    Terms     : {len(self.idx.index):,}")
        print(f"    Avg dl    : {self.idx.avg_dl:.1f} tokens")
        print(f"    k1={self.k1}  b={self.b}")

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(
        self,
        query_tokens: List[str],
        k1: float,
        b: float,
    ) -> Dict[str, float]:
        """
        Computes BM25 scores for all documents containing at least one
        query term. Documents in no posting list stay at 0 and are
        never inserted into the dict — efficient for large corpora.
        """
        assert self.idx is not None, "InvertedIndexManager not loaded!"
        scores: Dict[str, float] = {}
        avg_dl = self.idx.avg_dl

        for term in query_tokens:
            postings = self.idx.get_postings(term)  # {doc_id: tf}
            if not postings:
                continue

            idf = self.idx.idf(term)

            for doc_id, tf in postings.items():
                doc_len = self.idx.doc_lengths.get(doc_id, avg_dl)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_dl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (
                    numerator / denominator
                )

        return scores

    def _top_k(self, scores: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
        if not scores:
            return []
        doc_ids_arr = list(scores.keys())
        scores_arr = np.array(list(scores.values()), dtype=np.float32)

        if top_k >= len(scores_arr):
            top_i = np.argsort(scores_arr)[::-1]
        else:
            top_i_part = np.argpartition(scores_arr, -top_k)[-top_k:]
            top_i = top_i_part[np.argsort(scores_arr[top_i_part])[::-1]]

        return [(doc_ids_arr[i], float(scores_arr[i])) for i in top_i]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_tokens: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        query_tokens: preprocessed tokens from TextPreprocessor.
        Same stemmer as step2 — stems must match what is in the index.
        """
        if self.idx is None:
            raise RuntimeError("BM25Retriever not loaded. Call load_index() first.")
        return self._top_k(self._score(query_tokens, self.k1, self.b), top_k)

    def retrieve_with_params(
        self,
        query_tokens: List[str],
        k1: float,
        b: float,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve with custom k1/b without changing self.k1/self.b.
        Called by the UI when users move the parameter sliders.
        """
        if self.idx is None:
            raise RuntimeError("BM25Retriever not loaded.")
        return self._top_k(self._score(query_tokens, k1, b), top_k)

    def retrieve_batch(
        self,
        queries: Dict[str, List[str]],
        top_k: int = 1000,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch retrieval for evaluation. queries: {qid: [tokens]}"""
        return {qid: self.retrieve(tokens, top_k) for qid, tokens in queries.items()}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: Path):
        """
        Saves only k1, b, and dataset_name.
        The InvertedIndexManager is NOT duplicated here —
        it stays in data/indexes/{name}_inverted.pkl from step3.
        At load time we re-attach by name.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_path = save_dir / "bm25_config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(
                {
                    "dataset_name": self.dataset_name,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
            )

        print(f"  BM25 saved → {config_path}")
        print(f"    dataset: {self.dataset_name}  k1={self.k1}  b={self.b}")

    @classmethod
    def load(cls, save_dir: Path) -> "BM25Retriever":
        """
        Loads config, then re-attaches to the InvertedIndexManager.
        """
        save_dir = Path(save_dir)
        config_path = save_dir / "bm25_config.pkl"

        if not config_path.exists():
            raise FileNotFoundError(f"BM25 config not found: {config_path}")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        obj = cls(k1=config["k1"], b=config["b"])
        obj.load_index(config["dataset_name"])
        return obj
