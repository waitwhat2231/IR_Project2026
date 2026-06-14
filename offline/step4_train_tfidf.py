# offline/step4_train_tfidf.py
"""
Trains the TF-IDF model for beir/webis-touche2020/v2.

Reads:  data/processed/{name}/processed_docs.pkl  (multi-block stream)
Writes: data/models/tfidf_{name}_matrix.npz
        data/models/tfidf_{name}_meta.pkl

The critical step here is draining ALL pickle blocks from the stream
file written by step2. A single pickle.load() would give only the
first 20K documents. We use stream_chunks() which runs the
while True / EOFError loop internally to drain every block.

Memory note:
  TfidfVectorizer.fit() must see every document at once to compute
  corpus-wide IDF. We collect all processed_str values into a list
  before calling fit(). For 382K docs this costs ~1.5 GB.
  Peak during fit+transform: ~4-5 GB.
  If this crashes: set USE_LOW_MEMORY = True.
"""

import gc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from shared.config        import PROCESSED_DIR, MODEL_DIR, DATASETS
from shared.pickle_stream import stream_chunks
from Services.retrieval_service.tfidf_retriever import TFIDFRetriever


# ── Toggle ─────────────────────────────────────────────────────────────────────
USE_LOW_MEMORY = False   # set True if standard fit() exhausts your RAM


# ── Corpus builder (shared by both strategies) ─────────────────────────────────

def build_corpus(dataset_name: str) -> dict:
    """
    Drains the multi-block pickle stream into {doc_id: processed_str}.

    Why we need processed_str and not processed_tokens:
      sklearn's TfidfVectorizer.fit_transform() expects strings.
      It internally tokenizes using our lambda x: x.split().
      Passing a list of tokens would require a custom tokenizer
      that returns the list as-is, which works but is less clean.

    All chunks are drained via stream_chunks() — this is the fix
    for the silent 20K-only bug from the old single pickle.load() call.
    """
    pkl_path = PROCESSED_DIR / dataset_name / "processed_docs.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"  [ERROR] {pkl_path} not found.\n"
            f"  Run offline/step2_preprocess.py first."
        )

    print(f"  Draining stream: {pkl_path}")

    corpus      = {}   # {doc_id: processed_str}
    chunk_num   = 0
    total_docs  = 0

    for chunk in stream_chunks(pkl_path):
        chunk_num  += 1
        total_docs += len(chunk)

        for doc_id, doc_data in chunk.items():
            processed_str = doc_data.get("processed_str", "")
            if processed_str:                # skip empty documents
                corpus[doc_id] = processed_str

        print(
            f"    chunk {chunk_num}: +{len(chunk):,} docs "
            f"(total: {total_docs:,})",
            end="\r",
        )

        del chunk
        gc.collect()

    print(f"\n  Stream drained: {chunk_num} chunks, {total_docs:,} docs total.")
    empty = total_docs - len(corpus)
    if empty:
        print(f"  Skipped {empty:,} empty documents.")

    return corpus


# ── Strategy A: Standard TfidfVectorizer ───────────────────────────────────────

def train_standard(dataset_name: str):
    """
    Correct TF-IDF with proper corpus-wide IDF weighting.
    Requires ~4-5 GB RAM peak.

    This is the right model to use. Use strategy B only as a
    last resort if you genuinely cannot allocate 5 GB.
    """
    print(f"\n  [Standard TF-IDF]")

    out_prefix  = MODEL_DIR / f"tfidf_{dataset_name}"
    matrix_path = Path(str(out_prefix) + "_matrix.npz")
    meta_path   = Path(str(out_prefix) + "_meta.pkl")

    # Skip if already trained
    if matrix_path.exists() and meta_path.exists():
        print(f"  Already trained. Delete these files to retrain:")
        print(f"    {matrix_path}")
        print(f"    {meta_path}")
        return

    # Step 1: Build corpus dict from the multi-block pickle stream
    corpus = build_corpus(dataset_name)

    # Step 2: Fit and transform
    # This is the one point where the full corpus is in RAM simultaneously.
    # sklearn needs all texts to count df per term and compute IDF weights.
    model = TFIDFRetriever()
    model.fit(corpus)

    # Step 3: Free corpus before saving (matrix is the big object now)
    del corpus
    gc.collect()

    # Step 4: Save
    model.save(out_prefix)

    del model
    gc.collect()

    print(f"  Standard TF-IDF training complete for '{dataset_name}'.")


# ── Strategy B: HashingVectorizer (never loads full corpus) ────────────────────

def train_low_memory(dataset_name: str):
    """
    Approximate TF-IDF using HashingVectorizer + TfidfTransformer.

    Streams the corpus chunk by chunk — peak RAM is one chunk + matrix.
    Trade-off: no vocabulary dict (hashing trick), cannot inspect
    feature → word mapping. Retrieval quality is nearly identical.

    Saved files use the same naming convention as the standard model
    so the retriever service loads them transparently.
    The meta.pkl carries "mode": "hashing" so downstream code can
    detect which variant was saved.
    """
    import pickle
    from scipy.sparse import vstack, save_npz
    from sklearn.feature_extraction.text import (
        HashingVectorizer,
        TfidfTransformer,
    )

    print(f"\n  [Low-Memory HashingVectorizer]")

    pkl_path   = PROCESSED_DIR / dataset_name / "processed_docs.pkl"
    out_prefix = MODEL_DIR / f"tfidf_{dataset_name}"

    if not pkl_path.exists():
        raise FileNotFoundError(f"  {pkl_path} not found.")

    # Pass 1: build per-chunk TF sparse matrices
    hasher = HashingVectorizer(
        tokenizer      = lambda x: x.split(),
        preprocessor   = None,
        token_pattern  = None,
        n_features     = 2**20,      # 1M hash buckets
        norm           = None,       # TfidfTransformer applies norm
        alternate_sign = False,
    )

    chunk_matrices = []
    doc_ids_all    = []
    chunk_num      = 0

    for chunk in stream_chunks(pkl_path):
        chunk_num  += 1
        chunk_ids   = list(chunk.keys())
        chunk_texts = [chunk[did].get("processed_str", "") for did in chunk_ids]

        tf_chunk = hasher.transform(chunk_texts)
        chunk_matrices.append(tf_chunk)
        doc_ids_all.extend(chunk_ids)

        print(f"    Hashing chunk {chunk_num} ({len(chunk_ids):,} docs)...",
              end="\r")

        del chunk, chunk_texts
        gc.collect()

    print(f"\n  Stacking {len(chunk_matrices)} sparse matrices...")
    full_tf = vstack(chunk_matrices, format="csr")
    del chunk_matrices
    gc.collect()

    # Pass 2: fit TfidfTransformer on accumulated TF matrix
    print(f"  Fitting TfidfTransformer (shape={full_tf.shape})...")
    transformer  = TfidfTransformer(sublinear_tf=True, norm="l2")
    tfidf_matrix = transformer.fit_transform(full_tf)
    del full_tf
    gc.collect()

    # Save in the same layout TFIDFRetriever.load() expects
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(str(out_prefix) + "_matrix.npz")
    meta_path   = Path(str(out_prefix) + "_meta.pkl")

    save_npz(str(matrix_path), tfidf_matrix)

    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "vectorizer":   hasher,        # used for query.transform()
                "transformer":  transformer,   # used for query.transform()
                "doc_ids":      doc_ids_all,
                "mode":         "hashing",     # signals non-standard load path
            },
            f,
        )

    print(f"  Low-memory TF-IDF saved:")
    print(f"    {matrix_path}  ({matrix_path.stat().st_size / 1e6:.1f} MB)")
    print(f"    {meta_path}    ({meta_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Low-memory TF-IDF training complete for '{dataset_name}'.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # DATASETS now has one entry: {"webis-touche2020": "beir/webis-touche2020/v2"}
    for name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset : {name}")
        print(f"Strategy: {'Low-Memory HashingVectorizer' if USE_LOW_MEMORY else 'Standard TF-IDF'}")
        print(f"{'='*60}")

        if USE_LOW_MEMORY:
            train_low_memory(name)
        else:
            train_standard(name)

    print("\n\nStep 4 COMPLETE — TF-IDF model trained and saved.")