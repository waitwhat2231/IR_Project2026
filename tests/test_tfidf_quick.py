import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from Services.retrieval_service.tfidf_retriever import TFIDFRetriever
from shared.config import MODEL_DIR

# ── Load the trained model ──────────────────────────────────────
prefix = MODEL_DIR / "tfidf_webis-touche2020"
print("Loading TF-IDF model...")
model = TFIDFRetriever.load(prefix)
assert model.doc_matrix is not None
print(
    f"Loaded: {model.doc_matrix.shape[0]:,} docs, {model.doc_matrix.shape[1]:,} terms"
)

# ── Test queries (typical Touche2020 debate-style questions) ────
test_queries = [
    "should abortion be legal",
    "is social media harmful to society",
    "should the death penalty be abolished",
    "is climate change caused by humans",
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = model.retrieve(query, top_k=5)
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"  {rank}. {doc_id}  (score={score:.4f})")
