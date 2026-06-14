import sys, json, math
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from shared.config import MODEL_DIR
from Services.retrieval_service.tfidf_retriever import TFIDFRetriever

# ── Load qrels ─────────────────────────────────────────────────────────────────
QRELS_PATH = Path(r"C:\Users\LOQ\Desktop\IR_Project\IR_System\data\raw\webis-touche2020\qrels.json")
with open(QRELS_PATH) as f:
    qrels = json.load(f)
print(f"Loaded qrels: {len(qrels)} queries")

# ── Load already-preprocessed queries (same pipeline as docs) ──────────────────
QUERIES_PATH = Path(r"C:\Users\LOQ\Desktop\IR_Project\IR_System\data\processed\webis-touche2020\processed_queries.json")
with open(QUERIES_PATH) as f:
    raw = json.load(f)

# {query_id: processed_str} — only keep queries that have qrels
processed_queries = {
    qid: data["processed_str"]
    for qid, data in raw.items()
    if qid in qrels
}
print(f"Queries with qrels: {len(processed_queries)}")

# ── Load TF-IDF model ──────────────────────────────────────────────────────────
print("\nLoading TF-IDF model...")
model = TFIDFRetriever.load(MODEL_DIR / "tfidf_webis-touche2020")

# ── Batch retrieve ─────────────────────────────────────────────────────────────
TOP_K = 1000
print(f"Retrieving top-{TOP_K} docs for {len(processed_queries)} queries...")
run = model.retrieve_batch(processed_queries, top_k=TOP_K)

# ── Metrics ────────────────────────────────────────────────────────────────────
def ndcg_at_k(ranked_docs, relevant, k):
    dcg = 0.0
    for i, (doc_id, _) in enumerate(ranked_docs[:k]):
        rel = relevant.get(doc_id, 0)
        dcg += rel / math.log2(i + 2)
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(ranked_docs, relevant, k, rel_threshold=1):
    hits = sum(1 for doc_id, _ in ranked_docs[:k] if relevant.get(doc_id, 0) >= rel_threshold)
    return hits / k

def recall_at_k(ranked_docs, relevant, k, rel_threshold=1):
    total = sum(1 for r in relevant.values() if r >= rel_threshold)
    if total == 0: return 0.0
    hits = sum(1 for doc_id, _ in ranked_docs[:k] if relevant.get(doc_id, 0) >= rel_threshold)
    return hits / total

def average_precision(ranked_docs, relevant, rel_threshold=1):
    total = sum(1 for r in relevant.values() if r >= rel_threshold)
    if total == 0: return 0.0
    hits, ap = 0, 0.0
    for i, (doc_id, _) in enumerate(ranked_docs):
        if relevant.get(doc_id, 0) >= rel_threshold:
            hits += 1
            ap += hits / (i + 1)
    return ap / total

# ── Per-query metrics ──────────────────────────────────────────────────────────
results_per_query = {}
for qid, ranked in run.items():
    relevant = qrels[qid]
    results_per_query[qid] = {
        "ndcg@10":     ndcg_at_k(ranked, relevant, 10),
        "ndcg@100":    ndcg_at_k(ranked, relevant, 100),
        "p@10":        precision_at_k(ranked, relevant, 10),
        "recall@100":  recall_at_k(ranked, relevant, 100),
        "recall@1000": recall_at_k(ranked, relevant, 1000),
        "ap":          average_precision(ranked, relevant),
    }

# ── Aggregate ──────────────────────────────────────────────────────────────────
n = len(results_per_query)
metrics = list(next(iter(results_per_query.values())).keys())

print("\n" + "=" * 55)
print(f"  TF-IDF Evaluation — Touche2020 ({n} queries)")
print("=" * 55)
for m in metrics:
    avg = sum(r[m] for r in results_per_query.values()) / n
    print(f"  {m:<15}: {avg:.4f}")
print("=" * 55)

# ── Per-query breakdown ────────────────────────────────────────────────────────
print("\nPer-query NDCG@10:")
for qid in sorted(results_per_query, key=lambda x: int(x)):
    score    = results_per_query[qid]["ndcg@10"]
    original = raw[qid]["original"][:50]
    print(f"  Q{qid:>2}: {score:.4f}  — {original}")