# Services/retrieval_service/sbert_retriever.py
"""
SBERT dense retriever using sentence-transformers.

Input:  ORIGINAL text (not preprocessed/stemmed)
Model:  all-MiniLM-L6-v2  →  384-dim normalized embeddings
Index:  FAISS IndexFlatIP  →  exact cosine similarity search

Save layout:
  {save_dir}/doc_embeddings.npy   ← (N, 384) float32
  {save_dir}/faiss.index          ← FAISS binary index
  {save_dir}/meta.pkl             ← doc_ids + model_name
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SBERTRetriever:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name:     str                         = model_name
        self.model:          Optional[SentenceTransformer] = None
        self.doc_embeddings: Optional[np.ndarray]        = None  # (N, 384)
        self.faiss_index                                 = None
        self.doc_ids:        List[str]                   = []

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_documents(
        self,
        docs:       dict,           # {doc_id: "original text"}
        batch_size: int = 128,
    ):
        """
        docs must contain ORIGINAL text — not stemmed, not lowercased.
        SBERT uses WordPiece tokenization internally.
        Stemmed text ("coronaviru vaccin") produces degraded embeddings.

        normalize_embeddings=True:
          All vectors become unit length.
          Inner product then equals cosine similarity.
          FAISS IndexFlatIP computes cosine natively.
        """
        if self.model is None:
            print(f"  Loading SBERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name,device="cuda")

        self.doc_ids = list(docs.keys())
        texts        = list(docs.values())

        print(f"  Encoding {len(texts):,} documents...")
        print(f"  Batch size: {batch_size}  "
              f"Est. batches: {len(texts) // batch_size:,}")

        self.doc_embeddings = self.model.encode(
            texts,
            batch_size           = batch_size,
            show_progress_bar    = True,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )
        print(f"  Shape: {self.doc_embeddings.shape}  "
              f"({self.doc_embeddings.nbytes / 1e6:.1f} MB)")

        self._build_faiss()

    def _build_faiss(self):
        dim              = self.doc_embeddings.shape[1]   # 384
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.doc_embeddings.astype(np.float32))
        print(f"  FAISS index: {self.faiss_index.ntotal:,} vectors, dim={dim}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_raw: str,     # original query — NOT preprocessed
        top_k:     int = 10,
    ) -> List[Tuple[str, float]]:
        if self.model is None or self.faiss_index is None:
            raise RuntimeError("SBERTRetriever not ready. Call encode_documents() or load().")

        q_emb = self.model.encode(
            [query_raw],
            normalize_embeddings = True,
            convert_to_numpy     = True,
        ).astype(np.float32)                      # (1, 384)

        scores, indices = self.faiss_index.search(q_emb, top_k)

        return [
            (self.doc_ids[idx], float(scores[0][r]))
            for r, idx in enumerate(indices[0])
            if idx >= 0
        ]

    def get_embedding(self, text: str) -> np.ndarray:
        """Single embedding — used by hybrid serial reranker."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model.encode(
            [text],
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )[0]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "doc_embeddings.npy", self.doc_embeddings)
        faiss.write_index(self.faiss_index, str(save_dir / "faiss.index"))

        with open(save_dir / "meta.pkl", "wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "model_name": self.model_name}, f)

        emb_mb   = (save_dir / "doc_embeddings.npy").stat().st_size / 1e6
        faiss_mb = (save_dir / "faiss.index").stat().st_size / 1e6
        print(f"  SBERT saved → {save_dir}/")
        print(f"    doc_embeddings.npy : {emb_mb:.1f} MB")
        print(f"    faiss.index        : {faiss_mb:.1f} MB")

    @classmethod
    def load(cls, save_dir: Path) -> "SBERTRetriever":
        save_dir = Path(save_dir)

        with open(save_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        obj                = cls(model_name=meta["model_name"])
        obj.doc_ids        = meta["doc_ids"]
        obj.doc_embeddings = np.load(save_dir / "doc_embeddings.npy")
        obj.faiss_index    = faiss.read_index(str(save_dir / "faiss.index"))

        print(f"  Loading SBERT model for query encoding: {obj.model_name}")
        obj.model = SentenceTransformer(obj.model_name)

        print(f"  SBERT loaded ← {save_dir}/")
        print(f"    Docs : {len(obj.doc_ids):,}  Dim: {obj.doc_embeddings.shape[1]}")
        return obj