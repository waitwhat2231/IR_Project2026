# offline/step3_build_inverted_index.py
"""
Builds the Inverted Index for each dataset.

Reads:  data/processed/{name}/processed_docs.pkl  (multi-block pickle stream)
Writes: data/indexes/{name}_inverted.pkl          (InvertedIndex object)

Streams the processed_docs file chunk by chunk so we never hold
more than one 20K-doc batch in memory alongside the growing index.
"""

import pickle
import gc
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from shared.config        import PROCESSED_DIR, INDEX_DIR, DATASETS
from shared.pickle_stream import stream_chunks          # the while/EOFError reader
from Services.indexing_service.inverted_index import InvertedIndexManager


def build_and_save_inverted_index(dataset_name: str):
    # ── Paths ──────────────────────────────────────────────────────────────────
    docs_file = PROCESSED_DIR / dataset_name / "processed_docs.pkl"
    out_file  = INDEX_DIR / f"{dataset_name}_inverted.pkl"   # flat path, no subdir

    print(f"\n{'='*60}")
    print(f"Building Inverted Index for: {dataset_name}")
    print(f"{'='*60}")

    # ── Guard: step 2 must have run first ──────────────────────────────────────
    if not docs_file.exists():
        print(f"  [ERROR] {docs_file} not found.")
        print(f"  Run offline/step2_preprocess.py first.")
        return

    # ── Guard: skip if already built ──────────────────────────────────────────
    if out_file.exists():
        print(f"  Index already exists at {out_file} — skipping.")
        print(f"  Delete the file manually to rebuild.")
        return

    # ── Accumulator structures ─────────────────────────────────────────────────
    # These grow as we stream; only the index itself stays in memory.
    inverted:    dict = defaultdict(dict)   # {term: {doc_id: tf}}
    doc_lengths: dict = {}                  # {doc_id: token_count}
    all_doc_ids: list = []

    total_docs_processed = 0
    chunk_number         = 0

    # ── Stream the multi-block pickle file chunk by chunk ──────────────────────
    # This is the fix for Problem 1.
    # stream_chunks() uses the while True / EOFError pattern internally
    # so every one of the ~19 blocks gets read, not just the first one.
    for chunk in stream_chunks(docs_file):
        chunk_number += 1
        chunk_size    = len(chunk)
        print(f"  Processing chunk {chunk_number} ({chunk_size:,} docs)...",
              end="\r")

        for doc_id, doc_data in chunk.items():
            tokens = doc_data.get("processed_tokens", [])

            # ── Track document length (needed for BM25 avg_dl) ────────────────
            doc_lengths[doc_id] = len(tokens)
            all_doc_ids.append(doc_id)

            # ── Count term frequency inside this document ──────────────────────
            tf: dict = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            # ── Post to inverted lists ─────────────────────────────────────────
            # inverted[term][doc_id] = how many times term appears in doc_id
            for term, count in tf.items():
                inverted[term][doc_id] = count

        total_docs_processed += chunk_size

        # Discard the chunk and reclaim its memory before loading the next one
        del chunk
        gc.collect()

    print(f"\n  All {chunk_number} chunks processed.")
    print(f"  Total documents indexed: {total_docs_processed:,}")

    # ── Compute corpus-level statistics ───────────────────────────────────────
    # These are the values BM25 needs that your original code was missing.
    N      = len(all_doc_ids)
    avg_dl = sum(doc_lengths.values()) / N if N > 0 else 0.0
    df     = {term: len(postings) for term, postings in inverted.items()}

    print(f"  Unique terms:  {len(inverted):,}")
    print(f"  Avg doc len:   {avg_dl:.1f} tokens")

    # ── Assemble the InvertedIndex object ──────────────────────────────────────
    # This is the fix for Problems 3 and 4.
    # We fill the object's attributes directly instead of calling .build()
    # because .build() expects the full dict in RAM which we avoided.
    idx             = InvertedIndexManager(dataset_name)
    idx.index       = dict(inverted)    # defaultdict → plain dict for pickling
    idx.df          = df
    idx.doc_lengths = doc_lengths
    idx.doc_ids     = all_doc_ids
    idx.N           = N
    idx.avg_dl      = avg_dl

    del inverted, doc_lengths, all_doc_ids, df
    gc.collect()

    # ── Save using the class's own save method ─────────────────────────────────
   # ── Save the index object manually ───────────────────────────────────────
    print(f"  Saving Inverted Index to {out_file}...")
    
    # We use pickle to dump the entire object, which now holds all 
    # the index, df, doc_lengths, and corpus stats we populated.
    with open(out_file, "wb") as f:
        pickle.dump(idx, f)
        
    print(f"  Success! Index saved.")  # handles pickle + size reporting internally

    # ── Quick sanity check ─────────────────────────────────────────────────────
    print(f"\n  Sanity check:")
    sample_terms = list(idx.index.keys())[:3]
    for term in sample_terms:
        postings = idx.get_postings(term)
        idf_val  = idx.idf(term)
        print(f"    term='{term}'  df={len(postings):,}  idf={idf_val:.4f}  "
              f"sample_docs={list(postings.keys())[:2]}")

    print(f"\n  Step 3 COMPLETE for '{dataset_name}'.")


if __name__ == "__main__":
    for name in DATASETS:
        build_and_save_inverted_index(name)

    print("\n\nAll datasets indexed.")
    print("Step 3 COMPLETE — Inverted Index built successfully.")