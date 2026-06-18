# test_word2vec.py  — put this in your project root
import sys
from pathlib import Path

sys.path.append("c:/Users/LOQ/Desktop/IR_Project/IR_System")

from Services.retrieval_service.word2vec_retriever import Word2VecRetriever

# ── 1. Load the saved model ───────────────────────────────────────────────────
save_dir = Path(
    "c:/Users/LOQ/Desktop/IR_Project/IR_System/data/models/word2vec_webis-touche2020"
)
w2v = Word2VecRetriever.load(save_dir)

# ── 2. Sanity check the vocab ─────────────────────────────────────────────────
print("\n--- Vocab check ---")
test_words = ["abortion", "climate", "debate", "evidence", "government"]
for word in test_words:
    assert w2v.model is not None
    if word in w2v.model.wv:
        similar = w2v.model.wv.most_similar(word, topn=3)
        print(f"  '{word}' → {similar}")
    else:
        print(f"  '{word}' NOT in vocab")

# ── 3. Test retrieval with a raw query ────────────────────────────────────────
print("\n--- Retrieval check ---")
# Word2Vec needs preprocessed tokens (same pipeline as step2)
# so we manually tokenize a simple query for the test
query_tokens = ["should", "abortion", "legal"]  # pretend these are preprocessed
results = w2v.retrieve(query_tokens, top_k=5)
print(f"  Query tokens: {query_tokens}")
for rank, (doc_id, score) in enumerate(results, 1):
    print(f"  {rank}. {doc_id}  score={score:.4f}")

# ── 4. Check embedding matrix shape ──────────────────────────────────────────
assert w2v.doc_embeddings is not None
print("\n--- Embedding matrix ---")
print(f"  Shape : {w2v.doc_embeddings.shape}")  # should be (382K, 200)
print(f"  Dtype : {w2v.doc_embeddings.dtype}")  # should be float32
print(f"  Docs  : {len(w2v.doc_ids):,}")
