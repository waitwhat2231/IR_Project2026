# offline/step5_train_sbert.py
"""
Encodes all documents with SBERT and builds a FAISS index.

Reads:  data/raw/{name}/docs.pkl          (single-block, step1 output)
Writes: data/models/sbert_{name}/
            doc_embeddings.npy
            faiss.index
            meta.pkl

NOTE: raw docs.pkl is a SINGLE-BLOCK pickle written by step1.
      We use a plain pickle.load() here — correct for step1 output.
      Do NOT use stream_chunks() here.

Time estimate:
  CPU: 45–90 minutes for 382K docs
  GPU: 5–10 minutes
"""

import gc
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from shared.config import RAW_DIR, MODEL_DIR, DATASETS
from Services.retrieval_service.sbert_retriever import SBERTRetriever


def train_sbert(dataset_name: str, batch_size: int = 128):
    save_dir = MODEL_DIR / f"sbert_{dataset_name}"

    print(f"\n{'='*60}")
    print(f"SBERT training for: {dataset_name}")
    print(f"{'='*60}")

    if (save_dir / "meta.pkl").exists():
        print(f"  Already exists at {save_dir}/ — skipping.")
        print(f"  Delete the folder to retrain.")
        return

    # ── Load original documents from step1 output ─────────────────────────────
    raw_path = RAW_DIR / dataset_name / "docs.pkl"
    if not raw_path.exists():
        raise FileNotFoundError(f"  {raw_path} not found. Run step1 first.")

    print(f"  Loading raw documents (single pickle.load — step1 output)...")
    with open(raw_path, "rb") as f:
        raw_docs = pickle.load(f)   # single block — correct
    print(f"  Loaded {len(raw_docs):,} documents.")

    # Build text dict — original text capped at 512 chars
    # 512 chars ≈ BERT's effective context window after WordPiece tokenization
    original_texts = {
        doc_id: (
            (doc.get("title", "") or "") + " " +
            (doc.get("text",  "") or "")
        ).strip()[:512]
        for doc_id, doc in raw_docs.items()
    }

    del raw_docs
    gc.collect()

    # ── Encode and save ───────────────────────────────────────────────────────
    sbert = SBERTRetriever(model_name="all-MiniLM-L6-v2")
    sbert.encode_documents(original_texts, batch_size=batch_size)

    del original_texts
    gc.collect()

    sbert.save(save_dir)

    del sbert
    gc.collect()

    print(f"\n  Step 5 COMPLETE — SBERT trained for '{dataset_name}'.")


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in DATASETS:
        train_sbert(name, batch_size=128)

    print("\n\nStep 5 COMPLETE — SBERT models trained.")