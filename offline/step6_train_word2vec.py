# offline/step6_train_word2vec.py
"""
Trains Word2Vec on the preprocessed token corpus.

Reads:  data/processed/{name}/processed_docs.pkl   (multi-block stream)
Writes: data/models/word2vec_{name}/
            word2vec.model
            doc_embeddings.npy
            doc_ids.pkl

Uses _PickleStreamCorpus so gensim iterates the token lists
without loading all 382K documents into RAM at once.

Time estimate: 15–30 minutes for 382K docs (CPU, 5 epochs).
"""

import gc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from shared.config import PROCESSED_DIR, MODEL_DIR, DATASETS
from Services.retrieval_service.word2vec_retriever import Word2VecRetriever


def train_word2vec(dataset_name: str):
    save_dir = MODEL_DIR / f"word2vec_{dataset_name}"

    print(f"\n{'='*60}")
    print(f"Word2Vec training for: {dataset_name}")
    print(f"{'='*60}")

    if (save_dir / "doc_ids.pkl").exists():
        print(f"  Already exists at {save_dir}/ — skipping.")
        print(f"  Delete the folder to retrain.")
        return

    corpus_path = PROCESSED_DIR / dataset_name / "processed_docs.pkl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"  {corpus_path} not found. Run step2 first.")

    w2v = Word2VecRetriever(
        vector_size = 200,
        window      = 5,
        min_count   = 3,   # matches TF-IDF min_df=3
        workers     = 4,
        epochs      = 5,
    )
    w2v.fit(corpus_path)   # streams internally — memory safe
    w2v.save(save_dir)

    del w2v
    gc.collect()

    print(f"\n  Step 6 COMPLETE — Word2Vec trained for '{dataset_name}'.")


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in DATASETS:
        train_word2vec(name)

    print("\n\nStep 6 COMPLETE — Word2Vec models trained.")