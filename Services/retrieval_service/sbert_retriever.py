# Services/retrieval_service/sbert_retriever.py
"""
SBERT dense retriever using sentence-transformers.

Input:  ORIGINAL text (not preprocessed/stemmed)
Model:  all-MiniLM-L6-v2  ->  384-dim normalized embeddings
Index:  configurable      ->  "flat" (exact) | "ivf" (ANN) | "hnsw" (ANN)

CHANGES vs. the exact-only version:
  - __init__ takes index_type + ANN tuning params
  - _build_faiss() branches on index_type and trains IVF if needed
  - retrieve() now also returns whether the index is approximate, and
    lets you override the recall/speed knob (nprobe / efSearch) per call
  - save()/load() persist index_type + ANN params in meta.pkl so a
    reloaded retriever doesn't silently fall back to defaults
  - NEW: score_subset() scores the query against a *given* list of doc_ids
    directly from the precomputed embedding matrix -- no FAISS search at
    all. This is what the hybrid serial/cascade reranker should use.

    Why this exists: retrieve_serial() used to call
    retrieve(query_raw, top_k=len(self.doc_ids)) and then filter the
    result down to its ~200 candidates. That "ask FAISS for the whole
    corpus" pattern is always wasteful, but with index_type="hnsw" (the
    default) it's actively WRONG: HNSW's search is bounded by efSearch
    (default 64) and simply cannot return N valid results when N is the
    full corpus size -- most of the requested slots come back as -1 and
    get silently dropped. The leftover lookup table then only covers a
    small, essentially random slice of the corpus, so most of the
    cascade's real candidates silently get a reranked score of 0.0
    instead of their actual similarity. score_subset() sidesteps the
    index entirely for this case: it's a direct (len(doc_ids), 384) x
    (384,) dot product against rows we already have in memory, exact,
    fast, and correct regardless of which FAISS index_type is configured.

Save layout (unchanged):
  {save_dir}/doc_embeddings.npy   <- (N, 384) float32
  {save_dir}/faiss.index          <- FAISS binary index (any type)
  {save_dir}/meta.pkl             <- doc_ids + model_name + index config
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SBERTRetriever:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "hnsw",  # "flat" | "ivf" | "hnsw"
        nlist: int = 100,  # IVF: number of clusters
        nprobe: int = 10,  # IVF: clusters searched per query (speed/recall dial)
        hnsw_m: int = 32,  # HNSW: neighbors per node (build-time graph density)
        hnsw_ef_construction: int = 200,  # HNSW: build-time search width
        hnsw_ef_search: int = 64,  # HNSW: query-time search width (speed/recall dial)
    ):
        self.model_name: str = model_name
        self.model: Optional[SentenceTransformer] = None
        self.doc_embeddings: Optional[np.ndarray] = None  # (N, 384)
        self.faiss_index: Any = None
        self.doc_ids: List[str] = []

        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # Lazily-built doc_id -> row-in-doc_embeddings lookup, used only by
        # score_subset(). Built once on first use and cached; invalidated
        # automatically whenever doc_ids is reassigned by encode_documents()
        # or load() (see _ensure_id_index()).
        self._id_to_row: Optional[Dict[str, int]] = None
        self._id_to_row_built_for: Optional[int] = None  # id(self.doc_ids) it was built from

    # -- Encoding -----------------------------------------------------------

    def encode_documents(
        self,
        docs: dict,  # {doc_id: "original text"}
        batch_size: int = 128,
    ):
        """
        docs must contain ORIGINAL text -- not stemmed, not lowercased.
        SBERT uses WordPiece tokenization internally.
        Stemmed text ("coronaviru vaccin") produces degraded embeddings.

        normalize_embeddings=True:
          All vectors become unit length.
          Inner product then equals cosine similarity.
          Every FAISS index type below computes this metric natively
          when built with faiss.METRIC_INNER_PRODUCT.
        """
        if self.model is None:
            print(f"   Loading SBERT model: {self.model_name}")
            import torch

            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   [*] SentenceTransformer will run on: {target_device.upper()}")
            self.model = SentenceTransformer(self.model_name, device=target_device)

        self.doc_ids = list(docs.keys())
        texts = list(docs.values())

        print(f"  Encoding {len(texts):,} documents...")
        print(
            f"  Batch size: {batch_size}  "
            f"Est. batches: {len(texts) // batch_size:,}"
        )

        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        print(
            f"  Shape: {self.doc_embeddings.shape}  "
            f"({self.doc_embeddings.nbytes / 1e6:.1f} MB)"
        )

        self._build_faiss()

    def _build_faiss(self):
        assert (
            self.doc_embeddings is not None
        ), "encode_documents() must run before _build_faiss()"
        dim = self.doc_embeddings.shape[1]  # 384
        vecs = self.doc_embeddings.astype(np.float32)

        if self.index_type == "flat":
            # Exact search -- unchanged baseline behavior.
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(vecs)

        elif self.index_type == "ivf":
            # Approximate: cluster vectors, search only nprobe clusters.
            # IVF needs a *trained* coarse quantizer before vectors can be added --
            # this is the one structurally new step vs. flat search.
            n_train = vecs.shape[0]
            # FAISS needs roughly >= 30-50x nlist training points for stable clusters.
            effective_nlist = min(self.nlist, max(1, n_train // 39))
            if effective_nlist < self.nlist:
                print(
                    f"  [!] Lowering nlist {self.nlist} -> {effective_nlist} "
                    f"(not enough docs to train that many clusters)"
                )

            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT
            )
            print(
                f"  Training IVF quantizer on {n_train:,} vectors "
                f"(nlist={effective_nlist})..."
            )
            self.faiss_index.train(vecs)  # k-means clustering happens here
            self.faiss_index.add(vecs)
            self.faiss_index.nprobe = self.nprobe  # speed/recall dial, can change later

        elif self.index_type == "hnsw":
            # Approximate: greedy search over a multi-layer proximity graph.
            # No training step needed -- the graph is built incrementally on add().
            self.faiss_index = faiss.IndexHNSWFlat(
                dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            self.faiss_index.hnsw.efConstruction = self.hnsw_ef_construction
            self.faiss_index.add(vecs)
            self.faiss_index.hnsw.efSearch = self.hnsw_ef_search  # speed/recall dial

        else:
            raise ValueError(f"Unknown index_type: {self.index_type!r}")

        print(
            f"  FAISS index ({self.index_type}): "
            f"{self.faiss_index.ntotal:,} vectors, dim={dim}"
        )

    # -- Retrieval ------------------------------------------------------------

    def retrieve(
        self,
        query_raw: str,  # original query -- NOT preprocessed
        top_k: int = 10,
        nprobe: Optional[int] = None,  # override IVF's speed/recall dial for this query
        ef_search: Optional[
            int
        ] = None,  # override HNSW's speed/recall dial for this query
    ) -> List[Tuple[str, float]]:
        if self.model is None or self.faiss_index is None:
            raise RuntimeError(
                "SBERTRetriever not ready. Call encode_documents() or load()."
            )

        # Let a caller (e.g. the UI's "more accurate / faster" toggle) tune the
        # recall/speed trade-off per query without rebuilding the index.
        if self.index_type == "ivf" and nprobe is not None:
            self.faiss_index.nprobe = nprobe
        if self.index_type == "hnsw" and ef_search is not None:
            self.faiss_index.hnsw.efSearch = ef_search

        q_emb = self.model.encode(
            [query_raw],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(
            np.float32
        )  # (1, 384)

        scores, indices = self.faiss_index.search(q_emb, top_k)

        return [
            (self.doc_ids[idx], float(scores[0][r]))
            for r, idx in enumerate(indices[0])
            if idx >= 0
        ]

    def _ensure_id_index(self) -> Dict[str, int]:
        """
        Build (and cache) a doc_id -> row-in-doc_embeddings lookup.

        Rebuilt automatically if self.doc_ids has been reassigned since the
        last build (encode_documents()/load() both replace the list object
        outright, so an identity check is enough to detect that).
        """
        if self._id_to_row is None or self._id_to_row_built_for != id(self.doc_ids):
            self._id_to_row = {doc_id: row for row, doc_id in enumerate(self.doc_ids)}
            self._id_to_row_built_for = id(self.doc_ids)
        return self._id_to_row

    def score_subset(
        self,
        query_raw: str,  # original query -- NOT preprocessed
        doc_ids: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Score `query_raw` against ONLY `doc_ids`, directly from the
        precomputed embedding matrix -- no FAISS search involved at all.

        Use this instead of retrieve(..., top_k=len(self.doc_ids)) whenever
        you already have a short candidate list (e.g. the hybrid cascade's
        first-stage sparse results) and just need to re-rank it semantically.
        It is exact regardless of index_type ("flat"/"ivf"/"hnsw"), because
        it never touches the (possibly approximate) FAISS index -- it just
        re-uses the same unit-normalized vectors the index was built from.

        doc_ids not found in this retriever's embedding store are skipped
        (not padded with 0.0): a 0.0 default would be indistinguishable
        from "found, but completely dissimilar," which would silently bias
        downstream ranking the same way the bug this method replaces did.
        """
        if self.model is None or self.doc_embeddings is None:
            raise RuntimeError(
                "SBERTRetriever not ready. Call encode_documents() or load()."
            )
        if not doc_ids:
            return []

        id_to_row = self._ensure_id_index()

        rows: List[int] = []
        kept_ids: List[str] = []
        for doc_id in doc_ids:
            row = id_to_row.get(doc_id)
            if row is not None:
                rows.append(row)
                kept_ids.append(doc_id)

        if not rows:
            return []

        q_emb = self.model.encode(
            [query_raw],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)[0]  # (384,)

        candidate_vecs = self.doc_embeddings[rows]  # (len(rows), 384)
        scores = candidate_vecs @ q_emb  # inner product == cosine (both unit-norm)

        return list(zip(kept_ids, (float(s) for s in scores)))

    def get_embedding(self, text: str) -> np.ndarray:
        """Single embedding -- used by hybrid serial reranker."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

    # -- Persistence ------------------------------------------------------------

    def save(self, save_dir: Path):
        assert (
            self.doc_embeddings is not None
        ), "Nothing to save — call encode_documents() first"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "doc_embeddings.npy", self.doc_embeddings)
        faiss.write_index(self.faiss_index, str(save_dir / "faiss.index"))

        with open(save_dir / "meta.pkl", "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "model_name": self.model_name,
                    # Persist index config so load() reconstructs the same search
                    # behavior instead of silently defaulting back to "flat".
                    "index_type": self.index_type,
                    "nlist": self.nlist,
                    "nprobe": self.nprobe,
                    "hnsw_m": self.hnsw_m,
                    "hnsw_ef_construction": self.hnsw_ef_construction,
                    "hnsw_ef_search": self.hnsw_ef_search,
                },
                f,
            )

        emb_mb = (save_dir / "doc_embeddings.npy").stat().st_size / 1e6
        faiss_mb = (save_dir / "faiss.index").stat().st_size / 1e6
        print(f"  SBERT saved -> {save_dir}/")
        print(f"    index_type         : {self.index_type}")
        print(f"    doc_embeddings.npy : {emb_mb:.1f} MB")
        print(f"    faiss.index        : {faiss_mb:.1f} MB")

    @classmethod
    def load(cls, save_dir: Path) -> "SBERTRetriever":
        save_dir = Path(save_dir)

        with open(save_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        obj = cls(
            model_name=meta["model_name"],
            index_type=meta.get("index_type", "flat"),
            nlist=meta.get("nlist", 100),
            nprobe=meta.get("nprobe", 10),
            hnsw_m=meta.get("hnsw_m", 32),
            hnsw_ef_construction=meta.get("hnsw_ef_construction", 200),
            hnsw_ef_search=meta.get("hnsw_ef_search", 64),
        )
        obj.doc_ids = meta["doc_ids"]
        obj.doc_embeddings = np.load(save_dir / "doc_embeddings.npy")
        obj.faiss_index = faiss.read_index(str(save_dir / "faiss.index"))

        assert obj.doc_embeddings is not None, "doc_embeddings.npy failed to load"
        embedding_dim = obj.doc_embeddings.shape[1]  # capture now while narrowed

        # IVF's nprobe and HNSW's efSearch live on the loaded index object itself,
        # but FAISS's deserialization doesn't always restore them -- reapply explicitly.
        if obj.index_type == "ivf":
            obj.faiss_index.nprobe = obj.nprobe
        elif obj.index_type == "hnsw":
            obj.faiss_index.hnsw.efSearch = obj.hnsw_ef_search

        print(f"  Loading SBERT model for query encoding: {obj.model_name}")
        obj.model = SentenceTransformer(obj.model_name)

        print(f"  SBERT loaded <- {save_dir}/")
        print(
            f"    Docs : {len(obj.doc_ids):,}  Dim: {embedding_dim}  "
            f"Index: {obj.index_type}"
        )
        return obj