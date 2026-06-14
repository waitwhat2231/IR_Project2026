# Services/retrieval_service/word2vec_retriever.py
"""
Word2Vec dense retriever using gensim.

Input:  PREPROCESSED tokens (same pipeline as step2)
Model:  Trained on YOUR corpus — learns domain vocabulary
Vector: Document = mean of its word vectors, L2 normalized

Key difference from SBERT:
  SBERT  = pre-trained, sentence-level, 384-dim, needs original text
  Word2Vec = trained here, word-level, 200-dim, needs stemmed tokens
             learns that "remdesivir" and "dexamethasone" cluster
             together because they appear in similar debate contexts

Save layout:
  {save_dir}/word2vec.model       ← gensim model (word vectors + vocab)
  {save_dir}/doc_embeddings.npy   ← (N, 200) float32 document vectors
  {save_dir}/doc_ids.pkl          ← ordered doc_id list
"""

import gc
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class _EpochLogger(CallbackAny2Vec):                  # ← add this class here
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        self.epoch += 1
        print(f"    Epoch {self.epoch}/5 starting...", flush=True)
    def on_epoch_end(self, model):
        print(f"    Epoch {self.epoch}/5 done. Vocab={len(model.wv):,}", flush=True)

class Word2VecRetriever:

    def __init__(
        self,
        vector_size: int = 200,
        window:      int = 5,
        min_count:   int = 3,     # matches TF-IDF min_df=3
        workers:     int = 4,
        epochs:      int = 5,
    ):
        self.vector_size     = vector_size
        self.window          = window
        self.min_count       = min_count
        self.workers         = workers
        self.epochs          = epochs
        self.model:          Optional[Word2Vec]   = None
        self.doc_embeddings: Optional[np.ndarray] = None   # (N, 200)
        self.doc_ids:        List[str]            = []

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, corpus_path: Path):
        """
        corpus_path: data/processed/{name}/processed_docs.pkl
                     the multi-block stream written by step2

        Pass 1: train Word2Vec via _PickleStreamCorpus
                gensim calls __iter__ once per epoch — each call
                re-streams from disk so all 382K docs are seen
                per epoch without loading everything into RAM

        Pass 2: compute and store mean document vectors
        """
        corpus_path = Path(corpus_path)
        print(f"  Training Word2Vec (dim={self.vector_size}, window={self.window}, epochs={self.epochs})...")

        stream = _PickleStreamCorpus(corpus_path)

        print("  Building vocabulary...", flush=True)
        self.model = Word2Vec(
            sentences    = stream,
            vector_size  = self.vector_size,
            window       = self.window,
            min_count    = self.min_count,
            workers      = 4,
            epochs       = self.epochs,
            compute_loss = False,
            callbacks    = [_EpochLogger()],
        )
        print(f"  Vocabulary: {len(self.model.wv):,} words")
        print(f"  Computing document vectors (mean pooling)...")
        self._compute_doc_vectors(corpus_path)

    def _mean_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Mean of word vectors for known tokens.
        OOV tokens are skipped silently.
        Result is L2 normalized so dot product = cosine similarity.
        """
        vecs = [
            self.model.wv[t]
            for t in tokens
            if t in self.model.wv
        ]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)

        vec  = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _compute_doc_vectors(self, corpus_path: Path):
        from shared.pickle_stream import stream_chunks

        all_vectors = []
        all_doc_ids = []
        chunk_num   = 0

        for chunk in stream_chunks(corpus_path):
            chunk_num += 1
            print(f"    Vectorizing chunk {chunk_num} "
                  f"({len(chunk):,} docs)...", end="\r")

            for doc_id, doc_data in chunk.items():
                tokens = doc_data.get("processed_tokens", [])
                all_vectors.append(self._mean_vector(tokens))
                all_doc_ids.append(doc_id)

            del chunk
            gc.collect()

        self.doc_embeddings = np.vstack(all_vectors).astype(np.float32)
        self.doc_ids        = all_doc_ids

        print(f"\n  Doc vectors: {self.doc_embeddings.shape}  "
              f"({self.doc_embeddings.nbytes / 1e6:.1f} MB)")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_tokens: List[str],    # preprocessed tokens — same as step2
        top_k:        int = 10,
    ) -> List[Tuple[str, float]]:
        if self.model is None or self.doc_embeddings is None:
            raise RuntimeError("Word2VecRetriever not ready. Call fit() or load().")

        q_vec  = self._mean_vector(query_tokens)    # (200,)
        scores = self.doc_embeddings @ q_vec         # (N_docs,) cosine scores

        if top_k >= len(scores):
            top_i = np.argsort(scores)[::-1]
        else:
            top_i_part = np.argpartition(scores, -top_k)[-top_k:]
            top_i      = top_i_part[np.argsort(scores[top_i_part])[::-1]]

        return [(self.doc_ids[i], float(scores[i])) for i in top_i]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save(str(save_dir / "word2vec.model"))
        np.save(save_dir / "doc_embeddings.npy", self.doc_embeddings)

        with open(save_dir / "doc_ids.pkl", "wb") as f:
            pickle.dump(self.doc_ids, f)

        model_mb = (save_dir / "word2vec.model").stat().st_size / 1e6
        emb_mb   = (save_dir / "doc_embeddings.npy").stat().st_size / 1e6
        print(f"  Word2Vec saved → {save_dir}/")
        print(f"    word2vec.model     : {model_mb:.1f} MB")
        print(f"    doc_embeddings.npy : {emb_mb:.1f} MB")

    @classmethod
    def load(cls, save_dir: Path) -> "Word2VecRetriever":
        save_dir = Path(save_dir)

        obj                = cls()
        obj.model          = Word2Vec.load(str(save_dir / "word2vec.model"))
        obj.doc_embeddings = np.load(save_dir / "doc_embeddings.npy")

        with open(save_dir / "doc_ids.pkl", "rb") as f:
            obj.doc_ids = pickle.load(f)

        obj.vector_size = obj.doc_embeddings.shape[1]

        print(f"  Word2Vec loaded ← {save_dir}/")
        print(f"    Docs  : {len(obj.doc_ids):,}")
        print(f"    Vocab : {len(obj.model.wv):,} words")
        print(f"    Dim   : {obj.vector_size}")
        return obj


# ── Streaming corpus helper ────────────────────────────────────────────────────

class _PickleStreamCorpus:
    """
    Iterable over token lists from the multi-block pickle stream.

    Gensim calls __iter__ once per epoch.
    Each call re-opens the file from the start.
    Memory at any point: one 20K-doc chunk.
    """

    def __init__(self, pkl_path: Path):
        self.pkl_path = pkl_path

    def __iter__(self):
        from shared.pickle_stream import stream_chunks
        for chunk in stream_chunks(self.pkl_path):
            for doc_data in chunk.values():
                tokens = doc_data.get("processed_tokens", [])
                if tokens:
                    yield tokens