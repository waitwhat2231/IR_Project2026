# test_bm25.py — put in your project root
import sys
from pathlib import Path
sys.path.append('c:/Users/LOQ/Desktop/IR_Project/IR_System')

from Services.retrieval_service.bm25_retriever import BM25Retriever

# ── 1. Load the saved model ───────────────────────────────────────────────────
save_dir = Path('c:/Users/LOQ/Desktop/IR_Project/IR_System/data/models/bm25_webis-touche2020')
bm25 = BM25Retriever.load(save_dir)

# ── 2. Basic retrieval ────────────────────────────────────────────────────────
print("\n--- Basic retrieval ---")
queries = {
    "abortion" : ["abort", "right", "legal", "ban"],
    "climate"  : ["climat", "chang", "global", "warm"],
    "vaccines" : ["vaccin", "mandatori", "children", "school"],
}
for name, tokens in queries.items():
    results = bm25.retrieve(tokens, top_k=3)
    print(f"\n  Query '{name}': {tokens}")
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"    #{rank}  {doc_id}  score={score:.4f}")

# ── 3. Parameter sensitivity ──────────────────────────────────────────────────
print("\n--- Parameter sensitivity (k1/b effect) ---")
tokens = ["abort", "right", "legal", "ban"]
param_grid = [(1.0, 0.5), (1.5, 0.75), (2.0, 1.0)]
for k1, b in param_grid:
    results = bm25.retrieve_with_params(tokens, k1=k1, b=b, top_k=1)
    top_id, top_score = results[0] if results else ("none", 0)
    print(f"  k1={k1}  b={b}  →  top1={top_id}  score={top_score:.4f}")

# ── 4. Batch retrieval ────────────────────────────────────────────────────────
print("\n--- Batch retrieval ---")
batch = {
    "q1": ["abort", "right", "legal"],
    "q2": ["climat", "chang", "warm"],
    "q3": ["vaccin", "mandatori"],
}
batch_results = bm25.retrieve_batch(batch, top_k=2)
for qid, results in batch_results.items():
    print(f"  {qid}: {[(doc_id, round(score,3)) for doc_id, score in results]}")

# ── 5. Edge cases ─────────────────────────────────────────────────────────────
print("\n--- Edge cases ---")
# OOV query — should return empty list, not crash
oov_results = bm25.retrieve(["xyzxyzxyz", "nonexistenttoken"], top_k=5)
print(f"  OOV query → {oov_results} (should be [])")

# Single token
single = bm25.retrieve(["abort"], top_k=3)
print(f"  Single token 'abort' → {len(single)} results (should be 3)")