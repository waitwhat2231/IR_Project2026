"""
Interactive Search Script via Terminal.
Connects with HybridRetriever and integrates Requirement 5 (Query Refinement).
"""

import sys
from pathlib import Path

# إضافة جذر المشروع للمسارات
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from Services.retrieval_service.hybrid_retriever import HybridRetriever
from Services.PreprocessingService.preprocessor import TextPreprocessor
from Services.retrieval_service.query_refiner import QueryRefiner

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


def get_mongo_connection():
    """Connect to the secure Mongo database if it is a banking library."""
    if not PYMONGO_AVAILABLE:
        return None
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1500)
        db = client["ir_project"]  
        collection = db["webis-touche2020"]
        client.server_info()
        return collection
    except Exception:
        return None


def main():
    print("=" * 60)
    print("🎯 Welcome to IR 2026 Interactive Terminal Search Engine")
    print("=" * 60)

    # 1. إعداد المسارات وتحميل الموديلات
    dataset_name = "webis-touche2020"
    base_models_dir = PROJECT_ROOT / "data" / "models"
    
    bm25_dir = base_models_dir / f"bm25_{dataset_name}"
    tfidf_prefix = base_models_dir / f"tfidf_{dataset_name}"
    sbert_dir = base_models_dir / f"sbert_{dataset_name}"
    w2v_dir = base_models_dir / f"word2vec_{dataset_name}"

    # تحميل المسترجع الهجين
    hybrid_system = HybridRetriever()
    print("\n[1/4] Loading All Models and Retrievers into memory...")
    hybrid_system.load_all_retrievers(
        bm25_dir=bm25_dir, tfidf_prefix=tfidf_prefix, sbert_dir=sbert_dir, w2v_dir=w2v_dir
    )

    #Download the original preprocessor

    print("[2/4] Initializing Majd's Text Preprocessor...")
    preprocessor = TextPreprocessor(
        use_stemming=True, use_lemmatization=False, remove_stopwords=True,
        stemmer_type="porter", min_token_length=2
    )

# Download Query Improvement Service (Requirement 5)
    print("[3/4] Initializing Query Refinement Service (SpellCheck & Synonyms)...")
    search_history_file = PROJECT_ROOT / "data" / "search_history.json"
    refiner = QueryRefiner(history_path=search_history_file)

    print("[4/4] Checking MongoDB Storage Status...")
    mongo_coll = get_mongo_connection()
    if mongo_coll:
        print("Connected to MongoDB successfully.")
    else:
        print("Running in Local Index Mode (IDs & Scores only).")

    print("\n" + "═" * 60)
    print("System Ready! Enter 'exit' or 'quit' to stop.")
    print("═" * 60)

# Interactive Research Circle
    while True:
        query_raw = input("\n🔍 Enter your Search Query: ").strip()
        if not query_raw or query_raw.lower() in ['exit', 'quit']:
            print("Exiting Search Engine. See you later!")
            break

# ====== [A] Spelling Correction ======
        corrected_query = refiner.correct_spelling(query_raw)
        if corrected_query.lower() != query_raw.lower():
            print(f"🔧 Did you mean: '{corrected_query}'?")
            choice = input("Use corrected query? (y/n): ").strip().lower()
            if choice == 'y':
                query_raw = corrected_query

# Save the query in the log after approval to run the suggestions system later
        refiner.save_to_history(query_raw)

# ====== [B] First, expand the query with clean synonyms (Synonym Expansion) ======
        clean_expanded_words = refiner.expand_with_synonyms(query_raw)
        print(f"Clean Synonyms Found (Original words): {clean_expanded_words}")

# ====== [C] Processing words and synonyms together using a preprocessor ======
        query_tokens = []
        for word in clean_expanded_words:
            proc_res = preprocessor.process(word)
            if proc_res["processed_tokens"]:
                query_tokens.extend(proc_res["processed_tokens"])
                
        # إزالة التكرار من التوكنز النهائية الصافية
        query_tokens = list(set(query_tokens))
        print(f"Final Stemmed Tokens for Search: {query_tokens}")

        if not query_tokens:
            print("The query became empty! Try another query.")
            continue

        # اختيار نمط البحث
        print("\nChoose Retrieval Mode:")
        print("  1. Parallel Hybrid (BM25 + SBERT) [Alpha=0.5]")
        print("  2. Serial Cascade Hybrid (BM25 -> SBERT Rerank)")
        mode = input("Select choice (1 or 2): ").strip()

        # تنفيذ البحث بناءً على الاختيار
        if mode == '2':
            print(f"\n Executing Serial Cascade Retrieval (Top 200 -> Rerank)...")
            results = hybrid_system.retrieve_serial(
                query_raw=query_raw, query_tokens=query_tokens,
                sparse_method="bm25", dense_method="sbert", cascade_top_n=200, top_k=3
            )
        else:
            print(f"\nExecuting Parallel Retrieval...")
            results = hybrid_system.retrieve_hybrid(
                query_raw=query_raw, query_tokens=query_tokens,
                sparse_method="bm25", dense_method="sbert", alpha=0.5, top_k=3
            )

        # عرض النتائج
        print("\n" + "═" * 50)
        print(f"Top 3 Search Results for: '{query_raw}'")
        print("═" * 50)

        if not results:
            print("No matching documents found.")
            continue

        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"\nRank {rank} | Doc ID: {doc_id} | Score: {score:.4f}")
            print("-" * 45)
            
            if mongo_coll:
                try:
                    doc_data = mongo_coll.find_one({"_id": doc_id}) or mongo_coll.find_one({"id": doc_id})
                    if doc_data:
                        print(f"Title  : {doc_data.get('title', 'No Title')}")
                        print(f"Content: {doc_data.get('text', 'No Content')[:300]}...")
                except Exception:
                    pass
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()