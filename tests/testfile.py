import sys, json
from pathlib import Path
sys.path.append(str(Path("C:/Users/LOQ/Desktop/IR_Project/IR_System")))

from shared.config import MODEL_DIR
from Services.retrieval_service.tfidf_retriever import TFIDFRetriever
from Services.PreprocessingService.preprocessor import TextPreprocessor
from shared.pickle_stream import stream_chunks

preprocessor = TextPreprocessor(use_stemming=True, use_lemmatization=False,
                                 remove_stopwords=True, stemmer_type="porter", min_token_length=2)
model = TFIDFRetriever.load(MODEL_DIR / "tfidf_webis-touche2020")

with open(r"C:\Users\LOQ\Desktop\IR_Project\IR_System\data\raw\webis-touche2020\qrels.json") as f:
    qrels = json.load(f)

# ── Find what the judged docs for query 1 actually contain ────────────────────
print("Looking up judged docs for query 1 in the corpus...")

target_ids = set(qrels["1"].keys())
found = {}

pkl_path = Path(r"C:\Users\LOQ\Desktop\IR_Project\IR_System\data\processed\webis-touche2020\processed_docs.pkl")
for chunk in stream_chunks(pkl_path):
    for doc_id, doc_data in chunk.items():
        if doc_id in target_ids:
            found[doc_id] = doc_data.get("processed_str", "")
    if len(found) >= 5:
        break

print(f"\nSample judged docs for query 1 (processed_str):")
for doc_id, text in list(found.items())[:5]:
    rel = qrels["1"][doc_id]
    print(f"\n  doc_id : {doc_id}  (relevance={rel})")
    print(f"  text   : {text[:200]}")

# ── Also check what rank the judged docs appear at ────────────────────────────
print("\n\nChecking rank of judged docs under different query phrasings...")

test_queries = [
    "should abortion be legal",
    "abortion legal",
    "Is abortion beneficial for society",
    "abortion",
]

all_judged_q1 = set(qrels["1"].keys())

for q in test_queries:
    processed = preprocessor.process(q)["processed_str"]
    results   = model.retrieve(processed, top_k=len(model.doc_ids))
    
    # Find rank of first judged doc
    first_judged_rank = None
    for rank, (doc_id, score) in enumerate(results, 1):
        if doc_id in all_judged_q1:
            first_judged_rank = (rank, doc_id, score)
            break
    
    print(f"\n  Query: '{q}'  →  processed: '{processed}'")
    if first_judged_rank:
        print(f"  First judged doc at rank {first_judged_rank[0]} (score={first_judged_rank[2]:.4f})")
    else:
        print(f"  No judged doc found in entire corpus!")