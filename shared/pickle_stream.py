# shared/pickle_stream.py
"""
Utilities for reading multi-block pickle stream files.

Background:
  pickle.dump() called N times to the same open file creates N
  independent, self-contained blocks in that file, laid out as:

  [ block_0 ][ block_1 ][ block_2 ] ... [ block_N-1 ][ EOF ]
       ↑           ↑           ↑
   20K docs     20K docs    tail docs

  A single pickle.load() reads exactly ONE block and advances
  the file cursor to the start of the next block. To read all
  blocks, call pickle.load() repeatedly until EOFError is raised.

  The while True / EOFError pattern is the standard, correct way
  to drain a multi-block pickle file. There is no "block count"
  header to read — you simply keep loading until the file ends.
"""

import gc
import pickle
from pathlib import Path
from typing import Generator, Tuple, Any, Dict


def stream_chunks(pkl_path: Path) -> Generator[dict, None, None]:
    """
    Yields one chunk (dict of up to 20K docs) at a time.
    Memory held = one chunk at a time.

    Usage:
        for chunk in stream_chunks(path):
            for doc_id, doc in chunk.items():
                ...
    """
    with open(pkl_path, "rb") as f:
        while True:
            try:
                chunk = pickle.load(f)   # reads exactly one pickle block
                yield chunk
                # Caller processes chunk here; when the next iteration
                # begins, the previous chunk goes out of scope → GC eligible.
            except EOFError:
                # No more blocks — we have read the entire file.
                break


def stream_documents(pkl_path: Path) -> Generator[Tuple[str, Dict], None, None]:
    """
    Flattens the chunk stream into individual (doc_id, doc_data) pairs.
    Lowest possible memory footprint — one document in scope at a time
    (from Python's perspective; the chunk is still loaded per 20K batch).

    Usage:
        for doc_id, doc in stream_documents(path):
            tokens = doc["processed_tokens"]
            ...
    """
    for chunk in stream_chunks(pkl_path):
        yield from chunk.items()
        # chunk goes out of scope after this iteration step;
        # gc will reclaim it before loading the next chunk.


def load_all_into_dict(pkl_path: Path) -> dict:
    """
    Reconstructs the full dictionary by draining all chunks.
    Only use this when you genuinely need the entire corpus in RAM
    (e.g. fitting sklearn TfidfVectorizer, which requires all texts).
    For 389K docs this will use ~3 GB — acceptable if you have the RAM.
    If you don't, use stream_chunks() with incremental algorithms instead.
    """
    merged = {}
    for chunk in stream_chunks(pkl_path):
        merged.update(chunk)
        gc.collect()   # encourage reclaim between chunks during merge
    return merged


def count_documents(pkl_path: Path) -> int:
    """Count total documents without loading content into RAM."""
    total = 0
    for chunk in stream_chunks(pkl_path):
        total += len(chunk)
    return total