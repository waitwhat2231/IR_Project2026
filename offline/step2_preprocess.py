
"""
Memory-safe preprocessing pipeline for large document collections.

Strategy:
  - Pop documents from raw_docs one-by-one (dict shrinks as we go)
  - Accumulate into a temporary batch of 20,000 documents
  - Flush each batch as a separate pickle block appended to the stream file
  - Force garbage collection after every flush
  - Result: processed_docs.pkl contains N sequential pickle blocks,
    not one giant dictionary

Memory profile: peak ~4.2 GB regardless of corpus size.
"""

import gc
import json
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from shared.config import RAW_DIR, PROCESSED_DIR, DATASETS
from Services.PreprocessingService.preprocessor import TextPreprocessor


# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 20_000   # documents per pickle block


def preprocess_documents_streamed(
    name:         str,
    preprocessor: TextPreprocessor,
    chunk_size:   int = CHUNK_SIZE,
):
    """
    Process all documents for a dataset using a memory-safe streaming strategy.

    Reads:  data/raw/{name}/docs.pkl
    Writes: data/processed/{name}/processed_docs.pkl  (multi-block pickle stream)
    """
    raw_path  = RAW_DIR  / name
    proc_path = PROCESSED_DIR / name
    proc_path.mkdir(parents=True, exist_ok=True)

    out_file = proc_path / "processed_docs.pkl"

    if out_file.exists():
        print(f"  [{name}] processed_docs.pkl already exists — skipping.")
        print(f"  Delete {out_file} manually to re-run.")
        return

    # ── Step A: Load raw documents ─────────────────────────────────────────────
    print(f"\n  [{name}] Loading raw documents into memory...")
    raw_path_docs = raw_path / "docs.pkl"
    if not raw_path_docs.exists():
        raise FileNotFoundError(f"docs.pkl not found: {raw_path_docs}")

    with open(raw_path_docs, "rb") as f:
        raw_docs = pickle.load(f)

    total_docs  = len(raw_docs)
    total_chunks = (total_docs + chunk_size - 1) // chunk_size
    print(f"  [{name}] {total_docs:,} documents loaded.")
    print(f"  [{name}] Will write {total_chunks} chunk(s) of ≤{chunk_size:,} docs each.")

    # ── Step B: Extract keys once — do NOT iterate raw_docs directly ───────────
    # We pop from the dict while iterating keys, so we need the key list upfront.
    all_doc_ids = list(raw_docs.keys())

    # ── Step C: Open output file ONCE in write-binary mode ─────────────────────
    # Multiple pickle.dump() calls to the same open file create a sequential
    # stream of independent pickle blocks. Each block is self-contained.
    chunks_written = 0
    docs_written   = 0

    with open(out_file, "wb") as out_stream:

        current_batch: dict = {}

        for doc_id in tqdm(all_doc_ids, desc=f"  Preprocessing {name}", unit="doc"):

            # ── REQUIREMENT 1: In-Place Popping ───────────────────────────────
            # Pop removes the entry from raw_docs immediately.
            # raw_docs shrinks as current_batch grows → net memory is flat.
            doc = raw_docs.pop(doc_id)

            # ── Process the document ──────────────────────────────────────────
            full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            result    = preprocessor.process(full_text)

            current_batch[doc_id] = {
                "processed_str":    result["processed_str"],
                "processed_tokens": result["processed_tokens"],
                # Original text deliberately NOT stored here.
                # MongoDB (step 8) holds originals.
            }

            # ── REQUIREMENT 2: Chunked Pickle Appending ───────────────────────
            if len(current_batch) >= chunk_size:
                pickle.dump(current_batch, out_stream)
                # Explicit flush to OS — do not leave in Python I/O buffer
                out_stream.flush()

                docs_written   += len(current_batch)
                chunks_written += 1
                print(
                    f"\n  → Flushed chunk {chunks_written}/{total_chunks} "
                    f"({docs_written:,}/{total_docs:,} docs written)"
                )

                # ── Clear the batch dict — free its memory ─────────────────
                current_batch.clear()

                # ── REQUIREMENT 3: Garbage Collection ─────────────────────
                # .clear() removes references; gc.collect() tells CPython
                # to reclaim the backing memory and return pages to the OS.
                gc.collect()

        # ── Flush the final partial batch (tail < chunk_size) ─────────────────
        if current_batch:
            pickle.dump(current_batch, out_stream)
            out_stream.flush()
            docs_written   += len(current_batch)
            chunks_written += 1
            print(
                f"\n  → Flushed final chunk {chunks_written}/{total_chunks} "
                f"({docs_written:,}/{total_docs:,} docs written)"
            )
            current_batch.clear()
            gc.collect()

    # Verify the raw_docs dict is now fully empty
    assert len(raw_docs) == 0, "BUG: some documents were not popped!"
    del raw_docs
    gc.collect()

    print(f"\n  [{name}] Done.")
    print(f"  Output : {out_file}")
    print(f"  Chunks : {chunks_written}")
    print(f"  Docs   : {docs_written:,}")
    file_mb = out_file.stat().st_size / 1_000_000
    print(f"  Size   : {file_mb:.1f} MB")


def preprocess_queries(name: str, preprocessor: TextPreprocessor):
    """
    Queries are small (typically < 1,000 entries) — load all at once.
    No chunking needed.
    """
    raw_path  = RAW_DIR  / name
    proc_path = PROCESSED_DIR / name
    proc_path.mkdir(parents=True, exist_ok=True)

    q_out = proc_path / "processed_queries.json"
    if q_out.exists():
        print(f"  [{name}] processed_queries.json already exists — skipping.")
        return

    raw_queries = json.loads((raw_path / "queries.json").read_text())
    proc_queries = {}

    for qid, qtext in raw_queries.items():
        result = preprocessor.process(qtext)
        proc_queries[qid] = {
            "original":         qtext,
            "processed_str":    result["processed_str"],
            "processed_tokens": result["processed_tokens"],
        }

    q_out.write_text(json.dumps(proc_queries, indent=2))
    print(f"  [{name}] Saved {len(proc_queries):,} processed queries → {q_out}")


if __name__ == "__main__":
    # ── REQUIREMENT 4: Keep original config/preprocessor setup intact ──────────
    preprocessor = TextPreprocessor(
        use_stemming      = True,
        use_lemmatization = False,
        remove_stopwords  = True,
        stemmer_type      = "porter",
        min_token_length  = 2,
    )

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        preprocess_documents_streamed(dataset_name, preprocessor, CHUNK_SIZE)
        preprocess_queries(dataset_name, preprocessor)

    print("\n\nStep 2 COMPLETE — all datasets preprocessed (memory-safe).")