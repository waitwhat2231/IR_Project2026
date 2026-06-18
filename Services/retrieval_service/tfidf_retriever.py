# Services/retrieval_service/tfidf_retriever.py
"""
TF-IDF Retriever — beir/webis-touche2020/v2

Completely independent of InvertedIndexManager.
InvertedIndexManager is BM25's concern, not ours.

This retriever:
  - Fits sklearn TfidfVectorizer on the processed corpus
  - Saves a sparse .npz matrix + vectorizer metadata
  - Retrieves by cosine similarity at query time

Save format (two files, same prefix):
  {prefix}_matrix.npz    ← scipy sparse (N_docs × vocab)
  {prefix}_meta.pkl      ← vectorizer object + ordered doc_ids list
"""

import gc
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _whitespace_tokenizer(text: str):
    """Top-level tokenizer so pickle can serialize the TfidfVectorizer."""
    return text.split()


class TFIDFRetriever:
    """
    Sparse VSM retrieval using TF-IDF + cosine similarity.

    Why these hyperparameter choices for webis-touche2020:
      sublinear_tf=True  — debate documents are long; a word appearing
                           50 times vs 5 times should not get 10x the weight
      max_df=0.85        — words like "argument", "believe", "people" appear
                           in almost every debate post; not discriminative
      min_df=3           — with 382K docs, a term in <3 docs is a typo
                           or scraping artifact, not a real vocabulary item
      max_features=150K  — caps vocabulary to control sparse matrix size
                           and prevent memory explosion on the long tail
      norm="l2"          — required for cosine similarity to work correctly
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None  # scipy sparse (N_docs × vocab_size)
        self.doc_ids: List[str] = []

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, corpus: Dict[str, str]):
        """
        corpus: {doc_id: "stemmed joined string"}

        The strings must be preprocessed identically to how step2
        processed the documents — same stemmer, same stopwords, same join.
        sklearn's tokenizer is overridden with str.split() because the
        text is already clean; the default regex would re-tokenize it.
        """
        print(f"  Fitting TF-IDF on {len(corpus):,} documents...")

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=_whitespace_tokenizer,  # already preprocessed
            preprocessor=None,  # skip sklearn's cleaning
            token_pattern=None,  # type: ignore # disabled when tokenizer= is set
            sublinear_tf=True,
            max_df=0.85,
            min_df=3,
            max_features=150_000,
            norm="l2",
        )

        self.doc_ids = list(corpus.keys())
        corpus_texts = list(corpus.values())

        self.doc_matrix = self.vectorizer.fit_transform(corpus_texts)

        print(f"  Matrix shape : {self.doc_matrix.shape}")
        print(f"  Vocabulary   : {len(self.vectorizer.vocabulary_):,} terms")
        if self.doc_matrix is not None:
            nnz_val = getattr(self.doc_matrix, "nnz", 0)
            print(f"  Non-zeros    : {nnz_val:,}")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_processed_str: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Single-query retrieval.

        query_processed_str MUST go through the exact same preprocessor
        as the documents in step2 — lowercase, stopwords removed,
        Porter stemmed, joined by spaces.

        OOV terms (not seen during fit) are silently ignored by sklearn.

        Returns: [(doc_id, cosine_score), ...] descending by score.
        """
        if self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError(
                "TFIDFRetriever is not fitted. "
                "Call fit() or load() before retrieve()."
            )

        q_vec = self.vectorizer.transform([query_processed_str])
        scores = cosine_similarity(q_vec, self.doc_matrix).flatten()

        # argpartition is O(N) vs full argsort O(N log N)
        # For 382K docs this saves ~40ms per query
        if top_k >= len(scores):
            top_i = np.argsort(scores)[::-1]
        else:
            top_i_part = np.argpartition(scores, -top_k)[-top_k:]
            top_i = top_i_part[np.argsort(scores[top_i_part])[::-1]]

        return [(self.doc_ids[i], float(scores[i])) for i in top_i]

    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 1000,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch retrieval for all evaluation queries in one matrix op.
        Much faster than calling retrieve() in a loop because sklearn
        transforms all queries into a single (n_queries × vocab) matrix
        and computes all cosine similarities in one BLAS call.

        queries: {query_id: processed_query_string}
        Returns: {query_id: [(doc_id, score), ...]}
        """
        if self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError("TFIDFRetriever is not fitted.")

        qids = list(queries.keys())
        qtexts = list(queries.values())

        q_matrix = self.vectorizer.transform(qtexts)  # (n_q, vocab)
        all_scores = cosine_similarity(q_matrix, self.doc_matrix)  # (n_q, n_docs)

        results = {}
        for i, qid in enumerate(qids):
            row = all_scores[i]
            if top_k >= len(row):
                top_i = np.argsort(row)[::-1]
            else:
                top_i_part = np.argpartition(row, -top_k)[-top_k:]
                top_i = top_i_part[np.argsort(row[top_i_part])[::-1]]
            results[qid] = [(self.doc_ids[j], float(row[j])) for j in top_i]

        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, prefix: Path):
        """
        Saves two files next to each other:
          {prefix}_matrix.npz   ← sparse matrix binary
          {prefix}_meta.pkl     ← vectorizer + doc_ids

        Keeping them separate means you can memory-map the matrix
        without unpickling the Python vectorizer object.
        """
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        matrix_path = Path(str(prefix) + "_matrix.npz")
        meta_path = Path(str(prefix) + "_meta.pkl")

        save_npz(str(matrix_path), self.doc_matrix)

        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "doc_ids": self.doc_ids,
                },
                f,
            )

        print(f"  TF-IDF saved:")
        print(f"    {matrix_path}  " f"({matrix_path.stat().st_size / 1e6:.1f} MB)")
        print(f"    {meta_path}    " f"({meta_path.stat().st_size / 1e6:.1f} MB)")

    @classmethod
    def load(cls, prefix: Path) -> "TFIDFRetriever":
        """
        Loads from the two files written by save().
        prefix = same path passed to save(), no extension.

        Example:
            model = TFIDFRetriever.load(MODEL_DIR / "tfidf_webis-touche2020")
        """
        prefix = Path(prefix)
        matrix_path = Path(str(prefix) + "_matrix.npz")
        meta_path = Path(str(prefix) + "_meta.pkl")

        if not matrix_path.exists():
            raise FileNotFoundError(f"Matrix not found: {matrix_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta not found: {meta_path}")

        obj = cls()
        obj.doc_matrix = load_npz(str(matrix_path))

        with open(meta_path, "rb") as f:
            state = pickle.load(f)

        obj.vectorizer = state["vectorizer"]
        obj.doc_ids = state["doc_ids"]

        print(f"  TF-IDF loaded:")
        print(f"    Docs  : {obj.doc_matrix.shape[0]:,}")
        print(f"    Terms : {obj.doc_matrix.shape[1]:,}")

        return obj
