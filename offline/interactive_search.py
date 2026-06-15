"""
Interactive Search Script via Terminal.
Connects with HybridRetriever and gracefully handles MongoDB / Pymongo absence.
"""

import sys
from pathlib import Path

# إضافة جذر المشروع للمسارات
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from Services.retrieval_service.hybrid_retriever import HybridRetriever
from Services.PreprocessingService.preprocessor import TextPreprocessor

# محاولة استيراد المونغو بأمان
try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


def get_mongo_connection():
    """الاتصال بقاعدة بيانات المونغو بأمان إذا كانت المكتبة متوفرة"""
    if not PYMONGO_AVAILABLE:
        return None
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1500)
        db = client["ir_project"]  # تأكد من اسم قاعدة البيانات عندك
        collection = db["webis-touche2020"]
        # فحص الاتصال
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
    print("\n[1/3] Loading All Models and Retrievers into memory...")
    hybrid_system.load_all_retrievers(
        bm25_dir=bm25_dir, tfidf_prefix=tfidf_prefix, sbert_dir=sbert_dir, w2v_dir=w2v_dir
    )

    # تحميل المطحنة (Preprocessor) لمعالجة الاستعلام ديناميكياً
    print("[2/3] Initializing Query Preprocessor...")
    preprocessor = TextPreprocessor(
        use_stemming=True, use_lemmatization=False, remove_stopwords=True,
        stemmer_type="porter", min_token_length=2
    )

    # الاتصال بالمونغو جلب النصوص
    print("[3/3] Checking MongoDB Storage Status...")
    mongo_coll = get_mongo_connection()
    if mongo_coll:
        print("✅ Connected to MongoDB successfully.")
    else:
        print("ℹ️ Running in Local Index Mode (IDs & Scores only - MongoDB connection skipped).")

    print("\n" + "═" * 60)
    print("🚀 System Ready! Enter 'exit' or 'quit' to stop.")
    print("═" * 60)

    # حلقة البحث التفاعلية
    while True:
        query_raw = input("\n🔍 Enter your Search Query: ").strip()
        if not query_raw or query_raw.lower() in ['exit', 'quit']:
            print("👋 Exiting Search Engine. See you later!")
            break

        # تطبيق "معالجة الاستعلام" (Query Processing Requirement)
        proc_res = preprocessor.process(query_raw)
        query_tokens = proc_res["processed_tokens"]
        print(f"📝 Processed Tokens (Stemmed): {query_tokens}")

        if not query_tokens:
            print("⚠️ The query became empty after removing stopwords! Try another query.")
            continue

        print("\nChoose Retrieval Mode:")
        print("  1. Parallel Hybrid (BM25 + SBERT) [Alpha=0.5]")
        print("  2. Serial Cascade Hybrid (BM25 -> SBERT Rerank)")
        mode = input("Select choice (1 or 2): ").strip()

        # تنفيذ البحث بناء على الاختيار
        if mode == '2':
            print(f"\n🏃 Executing Serial Cascade Retrieval...")
            results = hybrid_system.retrieve_serial(
                query_raw=query_raw, query_tokens=query_tokens,
                sparse_method="bm25", dense_method="sbert", cascade_top_n=200, top_k=3
            )
        else:
            print(f"\n🏃 Executing Parallel Retrieval...")
            results = hybrid_system.retrieve_hybrid(
                query_raw=query_raw, query_tokens=query_tokens,
                sparse_method="bm25", dense_method="sbert", alpha=0.5, top_k=3
            )

        # عرض النتائج
        print("\n" + "═" * 50)
        print(f"🏆 Top 3 Search Results for: '{query_raw}'")
        print("═" * 50)

        if not results:
            print("❌ No matching documents found.")
            continue

        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"\n🥇 Rank {rank} | Doc ID: {doc_id} | Score: {score:.4f}")
            print("-" * 45)
            
            # محاولة جلب النص لو المونغو شغال
            if mongo_coll:
                try:
                    doc_data = mongo_coll.find_one({"_id": doc_id}) or mongo_coll.find_one({"id": doc_id})
                    if doc_data:
                        print(f"📌 Title  : {doc_data.get('title', 'No Title')}")
                        print(f"📖 Content: {doc_data.get('text', 'No Content')[:300]}...")
                except Exception:
                    pass
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()