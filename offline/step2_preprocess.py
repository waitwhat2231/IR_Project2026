# offline/step2_preprocess.py
"""
Preprocesses all documents and queries for both datasets.

For each document, saves:
  - processed_str:    "coronaviru vaccin effect"  (for TF-IDF)
  - processed_tokens: ["coronaviru","vaccin","effect"]  (for BM25/W2V)

Does NOT store original text — that stays in docs.pkl
until step8 loads it to MongoDB.
"""

import pickle
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from shared.config import RAW_DIR, PROCESSED_DIR, DATASETS
from Services.PreprocessingService.preprocessor import TextPreprocessor


def preprocess_dataset(name: str, preprocessor: TextPreprocessor):
    raw_path   = RAW_DIR / name
    proc_path  = PROCESSED_DIR / name
    proc_path.mkdir(parents=True, exist_ok=True)

    # ── Documents ─────────────────────────────────────────────────────────────
    out_file = proc_path / "processed_docs.pkl"
    if out_file.exists():
        print(f"  [{name}] processed_docs.pkl already exists — skipping.")
    else:
        print(f"\n  [{name}] Loading raw documents...")
        with open(raw_path / "docs.pkl", "rb") as f:
            raw_docs = pickle.load(f)

        processed = {}
        for doc_id, doc in tqdm(raw_docs.items(), desc=f"  Preprocessing {name}"):
            full_text = (doc.get("title","") + " " + doc.get("text","")).strip()
            result    = preprocessor.process(full_text)
            processed[doc_id] = {
                "processed_str":    result["processed_str"],
                "processed_tokens": result["processed_tokens"],
                # Original text NOT stored here — it goes to MongoDB in step 8
            }

        with open(out_file, "wb") as f:
            pickle.dump(processed, f)
        print(f"  Saved {len(processed):,} preprocessed docs → {out_file}")

    # ── Queries ───────────────────────────────────────────────────────────────
    q_out = proc_path / "processed_queries.json"
    if q_out.exists():
        print(f"  [{name}] processed_queries.json already exists — skipping.")
    else:
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
        print(f"  Saved {len(proc_queries):,} processed queries → {q_out}")


if __name__ == "__main__":
    # Use stemming (faster than lemmatization, good for IR)
    preprocessor = TextPreprocessor(
        use_stemming      = True,
        use_lemmatization = False,
        remove_stopwords  = True,
        stemmer_type      = "porter",
        min_token_length  = 2,
    )
    for name in DATASETS:
        preprocess_dataset(name, preprocessor)
    print("\nStep 2 COMPLETE — all text preprocessed.")