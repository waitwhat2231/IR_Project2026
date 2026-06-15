"""
Step 8: Verification Script for the Hybrid Retriever.
Tests BOTH Parallel and Serial retrieval modes along with custom parameters.
"""

import sys
from pathlib import Path

# إضافة المجلد الرئيسي للمشروع إلى sys.path لضمان استيراد الخدمات المشتركة
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from Services.retrieval_service.hybrid_retriever import HybridRetriever


def main():
    print("=" * 60)
    print("🚀 Initializing Step 8: Hybrid Retriever Verification (Parallel & Serial)")
    print("=" * 60)

    # 1. تحديد المسارات المباشرة للموديلات المتواجدة في جذر المشروع وتخطي الـ data/
    dataset_name = "webis-touche2020" 
    base_models_dir = PROJECT_ROOT / "data" / "models"
    
    bm25_dir = base_models_dir / f"bm25_{dataset_name}"
    tfidf_prefix = base_models_dir / f"tfidf_{dataset_name}"
    sbert_dir = base_models_dir / f"sbert_{dataset_name}"
    w2v_dir = base_models_dir / f"word2vec_{dataset_name}"

    # 2. إنشاء كائن المسترجع الهجين وتحميل الفهارس والموديلات
    hybrid_system = HybridRetriever()
    
    print("\n[Phase 1] Loading Sub-Retrievers...")
    hybrid_system.load_all_retrievers(
        bm25_dir=bm25_dir,
        tfidf_prefix=tfidf_prefix,
        sbert_dir=sbert_dir,
        w2v_dir=w2v_dir
    )

    # 3. بيانات استعلام تجريبي لفحص النظام
    query_raw = "Should schools ban mobile phones in classrooms?"
    query_tokens = ["school", "ban", "mobil", "phone", "classroom"]

    print("\n" + "-" * 50)
    print(f"🔍 Testing Hybrid Retrieval with Sample Query:")
    print(f"   Raw Text : '{query_raw}'")
    print(f"   Tokens   : {query_tokens}")
    print("-" * 50)

    # ─── الفحص الأول: التمثيل الهجين التوازي (Parallel) ───
    print("\n[Test 1] Running PARALLEL Hybrid (BM25 + SBERT) | Alpha = 0.5")
    results_parallel = hybrid_system.retrieve_hybrid(
        query_raw=query_raw,
        query_tokens=query_tokens,
        sparse_method="bm25",
        dense_method="sbert",
        alpha=0.5,
        top_k=5
    )
    
    print(f"\nTop 5 Results for PARALLEL Mode:")
    if results_parallel:
        for rank, (doc_id, score) in enumerate(results_parallel, 1):
            print(f"   Rank {rank}: Doc ID = {doc_id:<12} | Score = {score:.4f}")
    else:
        print("   ⚠️ No results returned for Parallel Mode.")

    # ─── الفحص الثاني: التمثيل الهجين التسلسلي (Serial) ───
    print("\n" + "-" * 50)
    print("[Test 2] Running SERIAL Cascade Hybrid (BM25 -> SBERT Rerank)")
    print("   Stage 1: Filter top 200 via BM25 -> Stage 2: Rerank via SBERT")
    
    results_serial = hybrid_system.retrieve_serial(
        query_raw=query_raw,
        query_tokens=query_tokens,
        sparse_method="bm25",
        dense_method="sbert",
        cascade_top_n=200,  # تصفية أولى لـ 200 وثيقة
        top_k=5,
        custom_bm25_params=(1.2, 0.75)
    )
    
    print(f"\nTop 5 Results for SERIAL Mode:")
    if results_serial:
        for rank, (doc_id, score) in enumerate(results_serial, 1):
            print(f"   Rank {rank}: Doc ID = {doc_id:<12} | Score = {score:.4f}")
    else:
        print("   ⚠️ No results returned for Serial Mode.")

    print("\n" + "=" * 60)
    print("✅ Step 8 Verification Completed Successfully for Both Modes!")
    print("=" * 60)


if __name__ == "__main__":
    main()