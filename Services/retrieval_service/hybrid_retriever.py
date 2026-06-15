from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from Services.indexing_service.inverted_index import InvertedIndexManager
from Services.retrieval_service.sbert_retriever import SBERTRetriever
from Services.retrieval_service.tfidf_retriever import TFIDFRetriever
from Services.retrieval_service.word2vec_retriever import Word2VecRetriever


class HybridRetriever:

    def __init__(self):
        self.bm25: Optional[InvertedIndexManager] = None
        self.tfidf: Optional[TFIDFRetriever] = None
        self.sbert: Optional[SBERTRetriever] = None
        self.w2v: Optional[Word2VecRetriever] = None

    def load_all_retrievers(self, bm25_dir: Path, tfidf_prefix: Path, sbert_dir: Path, w2v_dir: Path):
        """Loads all sub-retrievers from data/models."""
        project_root = Path(__file__).parent.parent.parent
        data_models_dir = project_root / "data" / "models"
        dataset_name = bm25_dir.name.replace("bm25_", "")

        # 1. Load BM25
        try:
            print(f"   Loading Majd's InvertedIndexManager for [{dataset_name}]...")
            self.bm25 = InvertedIndexManager(dataset_name)
        except Exception as e:
            print(f"[Warning] Could not load BM25: {e}")

        # 2. Load TF-IDF
        try:
            actual_tfidf_prefix = data_models_dir / f"tfidf_{dataset_name}"
            self.tfidf = TFIDFRetriever.load(actual_tfidf_prefix)
        except Exception as e:
            print(f"[Warning] Could not load TF-IDF: {e}")

        # 3. Load SBERT
        try:
            actual_sbert_dir = data_models_dir / f"sbert_{dataset_name}"
            self.sbert = SBERTRetriever.load(actual_sbert_dir)
        except Exception as e:
            print(f"[Warning] Could not load SBERT: {e}")

        # 4. Load Word2Vec
        try:
            actual_w2v_dir = data_models_dir / f"word2vec_{dataset_name}"
            self.w2v = Word2VecRetriever.load(actual_w2v_dir)
        except Exception as e:
            print(f"[Warning] Could not load Word2Vec: {e}")

    def _normalize_scores(self, results: List[Tuple[str, float]]) -> Dict[str, float]:
        """Applies Min-Max Normalization to map scores into [0, 1]."""
        if not results:
            return {}
        scores = [score for _, score in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {doc_id: 1.0 for doc_id, _ in results}
        return {doc_id: (score - min_s) / (max_s - min_s) for doc_id, score in results}

    def _compute_custom_bm25(self, query_tokens: List[str], k1: float, b: float, top_k: int) -> List[Tuple[str, float]]:
        """Calculates BM25 score dynamically using Majd's inverted index and UI sliders."""
        if not self.bm25:
            return []
        scores: Dict[str, float] = {}
        avg_dl = self.bm25.avg_dl
        
        for token in query_tokens:
            postings = self.bm25.get_postings(token)
            if not postings:
                continue
            idf = self.bm25.idf(token)
            for doc_id, tf in postings.items():
                doc_len = self.bm25.doc_lengths.get(doc_id, avg_dl)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_dl))
                scores[doc_id] = scores.get(doc_id, 0.0) + (idf * (numerator / denominator))
                
        sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_res[:top_k]

    # ─── النوع الأول: التمثيل التوازي (Parallel Representation) ───
    def retrieve_hybrid(
        self,
        query_raw: str,
        query_tokens: List[str],
        sparse_method: str = "bm25",  
        dense_method: str = "sbert",  
        alpha: float = 0.5,            
        top_k: int = 10,
        custom_bm25_params: Optional[Tuple[float, float]] = None  
    ) -> List[Tuple[str, float]]:
        """Executes parallel retrieval combining sparse and dense systems simultaneously."""
        # 1. Fetch Sparse
        sparse_results = []
        if sparse_method == "bm25" and self.bm25:
            k1, b = custom_bm25_params if custom_bm25_params else (1.2, 0.75)
            sparse_results = self._compute_custom_bm25(query_tokens, k1=k1, b=b, top_k=top_k * 3)
        elif sparse_method == "tfidf" and self.tfidf:
            sparse_results = self.tfidf.retrieve(" ".join(query_tokens), top_k=top_k * 3)

        # 2. Fetch Dense
        dense_results = []
        if dense_method == "sbert" and self.sbert:
            dense_results = self.sbert.retrieve(query_raw, top_k=top_k * 3)
        elif dense_method == "w2v" and self.w2v:
            dense_results = self.w2v.retrieve(query_tokens, top_k=top_k * 3)

        # 3. Fusion
        norm_sparse = self._normalize_scores(sparse_results)
        norm_dense = self._normalize_scores(dense_results)
        all_candidates = set(norm_sparse.keys()).union(set(norm_dense.keys()))
        
        hybrid_scores: Dict[str, float] = {}
        for doc_id in all_candidates:
            hybrid_scores[doc_id] = (alpha * norm_sparse.get(doc_id, 0.0)) + ((1.0 - alpha) * norm_dense.get(doc_id, 0.0))

        return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ─── النوع الثاني: التمثيل التسلسلي المكتمل (Serial / Cascade Representation) ───
    def retrieve_serial(
        self,
        query_raw: str,
        query_tokens: List[str],
        sparse_method: str = "bm25",
        dense_method: str = "sbert",
        cascade_top_n: int = 200,  # عدد المستندات الممررة للمرحلة الثانية للـ Reranking
        top_k: int = 10,
        custom_bm25_params: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[str, float]]:
        """Executes serial multi-stage retrieval (Sparse Filter -> Dense Reranker)."""
        # المرحلة 1: تصفية أولية سريعة باستخدام الموديل النصي (جلب عدد cascade_top_n من الوثائق)
        if sparse_method == "bm25" and self.bm25:
            k1, b = custom_bm25_params if custom_bm25_params else (1.2, 0.75)
            candidate_docs = self._compute_custom_bm25(query_tokens, k1=k1, b=b, top_k=cascade_top_n)
        elif sparse_method == "tfidf" and self.tfidf:
            candidate_docs = self.tfidf.retrieve(" ".join(query_tokens), top_k=cascade_top_n)
        else:
            return []

        if not candidate_docs:
            return []

        candidate_ids = [doc_id for doc_id, _ in candidate_docs]

        # المرحلة 2: إعادة ترتيب (Reranking) للوثائق المرشحة فقط باستخدام الـ Embedding المختار
        reranked_scores: Dict[str, float] = {}
        
        if dense_method == "sbert" and self.sbert:
            # نقوم بحساب التشابه مع الاستعلام للوثائق المرشحة فقط عبر الفهرس
            # لتبسيط العملية وتجنب الـ Full-scan، نأخذ سكور الـ SBERT المباشر لهذه الـ IDs
            all_sbert_res = self.sbert.retrieve(query_raw, top_k=len(self.sbert.doc_ids))
            sbert_lookup = {d_id: score for d_id, score in all_sbert_res}
            for d_id in candidate_ids:
                reranked_scores[d_id] = sbert_lookup.get(d_id, 0.0)
                
        elif dense_method == "w2v" and self.w2v:
            all_w2v_res = self.w2v.retrieve(query_tokens, top_k=len(self.w2v.doc_ids))
            w2v_lookup = {d_id: score for d_id, score in all_w2v_res}
            for d_id in candidate_ids:
                reranked_scores[d_id] = w2v_lookup.get(d_id, 0.0)

        # ترتيب النتائج النهائية بناءً على سكور إعادة الترتيب الدلالي
        sorted_serial = sorted(reranked_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_serial[:top_k]