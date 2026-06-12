# offline/step1_download.py
"""
Downloads both datasets via ir_datasets.
Saves: docs.pkl, queries.json, qrels.json for each dataset.

docs.pkl   = {doc_id: {"title": "...", "text": "..."}}
queries.json = {query_id: "query text"}
qrels.json   = {query_id: {doc_id: relevance_score}}

NOTE: docs.pkl here is TEMPORARY. After offline step 8,
MongoDB will hold the originals. These files are used
only during offline processing (steps 2-7).
"""

import ir_datasets
import pickle
import json
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from shared.config import RAW_DIR, DATASETS


def download_and_save(dataset_key: str, name: str):
    save_path = RAW_DIR / name
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  |  Key: {dataset_key}")
    print(f"{'='*60}")

    dataset = ir_datasets.load(dataset_key)

    # ── Documents ─────────────────────────────────────────────────────────────
    docs_file = save_path / "docs.pkl"
    if docs_file.exists():
        print(f"  docs.pkl already exists — skipping download.")
        with open(docs_file, "rb") as f:
            docs = pickle.load(f)
    else:
        docs = {}
        for doc in tqdm(dataset.docs_iter(), desc="  Downloading docs"):
            title = getattr(doc, "title", "") or ""
            text  = getattr(doc, "text",  "") or ""
            docs[doc.doc_id] = {"title": title, "text": text}

        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)
        print(f"  Saved {len(docs):,} documents → {docs_file}")

    # ── Queries ───────────────────────────────────────────────────────────────
    queries_file = save_path / "queries.json"
    if not queries_file.exists():
        queries = {q.query_id: q.text for q in dataset.queries_iter()}
        queries_file.write_text(json.dumps(queries, indent=2))
        print(f"  Saved {len(queries):,} queries → {queries_file}")
    else:
        print(f"  queries.json already exists — skipping.")

    # ── Qrels (ground truth for evaluation) ───────────────────────────────────
    qrels_file = save_path / "qrels.json"
    if not qrels_file.exists():
        qrels = {}
        for qrel in dataset.qrels_iter():
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
        qrels_file.write_text(json.dumps(qrels, indent=2))
        print(f"  Saved qrels for {len(qrels):,} queries → {qrels_file}")
    else:
        print(f"  qrels.json already exists — skipping.")

    print(f"\n  Summary for '{name}':")
    print(f"    Documents: {len(docs):,}")
    print(f"    Sample ID: {list(docs.keys())[0]}")
    print(f"    Sample title: {list(docs.values())[0]['title'][:70]}")


if __name__ == "__main__":
    for name, key in DATASETS.items():
        download_and_save(key, name)
    print("\n\nStep 1 COMPLETE — all datasets downloaded.")