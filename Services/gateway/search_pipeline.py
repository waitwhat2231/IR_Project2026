"""Search orchestration for the API Gateway — mirrors offline/interactive_search.py."""

import torch

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Services.PreprocessingService.preprocessor import TextPreprocessor
from Services.retrieval_service.hybrid_retriever import HybridRetriever
from Services.retrieval_service.query_refiner import QueryRefiner
from shared.config import BM25_B, BM25_K1, DATASETS, MODEL_DIR, TOP_K
from shared.database import DocumentDatabase

PROJECT_ROOT = Path(__file__).parent.parent.parent


class SearchPipeline:
    """Loads retrievers per dataset and runs basic or enhanced search."""

    def __init__(self):
        self._retrievers: Dict[str, HybridRetriever] = {}
        self.preprocessor = TextPreprocessor(
            use_stemming=True,
            use_lemmatization=False,
            remove_stopwords=True,
            stemmer_type="porter",
            min_token_length=2,
        )
        history_path = PROJECT_ROOT / "data" / "search_history.json"
        self.refiner = QueryRefiner(history_path=history_path)
        self._mongo: Optional[DocumentDatabase] = None
        self._mongo_connected = False

    def connect_mongo(self) -> Optional[DocumentDatabase]:
        if self._mongo_connected and self._mongo is not None:
            return self._mongo
        try:
            self._mongo = DocumentDatabase()
            self._mongo.connect()
            self._mongo_connected = True
            return self._mongo
        except Exception:
            self._mongo = None
            self._mongo_connected = False
            return None

    def get_retriever(self, dataset: str) -> HybridRetriever:
        if dataset not in self._retrievers:
            if dataset not in DATASETS:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}"
                )
            base = MODEL_DIR
            hybrid = HybridRetriever()
            hybrid.load_all_retrievers(
                bm25_dir=base / f"bm25_{dataset}",
                tfidf_prefix=base / f"tfidf_{dataset}",
                sbert_dir=base / f"sbert_{dataset}",
                w2v_dir=base / f"word2vec_{dataset}",
            )
            self._retrievers[dataset] = hybrid

            # Feed the refiner's spell-checker the *real* corpus vocabulary so
            # proper nouns / named entities / domain terms that genuinely appear
            # in this dataset's documents (e.g. "Hitler") aren't mistaken for
            # typos and "corrected" into an unrelated dictionary word (e.g.
            # "hitter"). This is free: it just reuses the inverted index's
            # term keys, which are already resident in memory from the load
            # above -- no extra disk I/O, no extra model.
            if hybrid.bm25 is not None:
                self.refiner.load_corpus_vocabulary(hybrid.bm25.index.keys())

        return self._retrievers[dataset]

    def models_ready(self, dataset: str) -> bool:
        try:
            r = self.get_retriever(dataset)
            return any([r.bm25, r.tfidf, r.sbert, r.w2v])
        except ValueError:
            return False

    def document_count(self, dataset: str) -> Optional[int]:
        mongo = self.connect_mongo()
        if mongo is None:
            return None
        try:
            return mongo.count(dataset)
        except Exception:
            return None

    def _preprocess_query(
        self,
        query: str,
        execution_mode: str,
    ) -> Tuple[str, bool, List[str], List[str]]:
        """Return (processed_query, spell_corrected, expanded_terms, query_tokens)."""
        processed_query = query.strip()
        spell_corrected = False
        expanded_terms: List[str] = []

        if execution_mode == "enhanced":
            corrected = self.refiner.correct_spelling(processed_query)
            if corrected.lower() != processed_query.lower():
                processed_query = corrected
                spell_corrected = True
            self.refiner.save_to_history(processed_query)
            expanded_terms = self.refiner.expand_with_synonyms(processed_query)
            query_tokens: List[str] = []
            for word in expanded_terms:
                proc = self.preprocessor.process(word)
                if proc["processed_tokens"]:
                    query_tokens.extend(proc["processed_tokens"])
            query_tokens = list(set(query_tokens))
        else:
            proc = self.preprocessor.process(processed_query)
            query_tokens = proc["processed_tokens"]

        return processed_query, spell_corrected, expanded_terms, query_tokens

    def _retrieve(
        self,
        hybrid: HybridRetriever,
        *,
        retrieval_mode: str,
        query_raw: str,
        query_tokens: List[str],
        sparse_method: str,
        dense_method: str,
        bm25_k1: float,
        bm25_b: float,
        alpha: float,
        cascade_top_n: int,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        bm25_params = (bm25_k1, bm25_b)

        if retrieval_mode == "bm25":
            return hybrid._compute_custom_bm25(
                query_tokens,
                k1=bm25_k1,
                b=bm25_b,
                top_k=top_k,
            )
        if retrieval_mode == "tfidf":
            if not hybrid.tfidf:
                return []
            return hybrid.tfidf.retrieve(" ".join(query_tokens), top_k=top_k)
        if retrieval_mode == "sbert":
            if not hybrid.sbert:
                return []
            return hybrid.sbert.retrieve(query_raw, top_k=top_k)
        if retrieval_mode == "word2vec":
            if not hybrid.w2v:
                return []
            return hybrid.w2v.retrieve(query_tokens, top_k=top_k)
        if retrieval_mode == "hybrid_serial":
            return hybrid.retrieve_serial(
                query_raw=query_raw,
                query_tokens=query_tokens,
                sparse_method=sparse_method,
                dense_method=dense_method,
                cascade_top_n=cascade_top_n,
                top_k=top_k,
                custom_bm25_params=bm25_params,
            )
        # hybrid_parallel (default)
        return hybrid.retrieve_hybrid(
            query_raw=query_raw,
            query_tokens=query_tokens,
            sparse_method=sparse_method,
            dense_method=dense_method,
            alpha=alpha,
            top_k=top_k,
            custom_bm25_params=bm25_params,
        )

    def search(
        self,
        *,
        dataset: str,
        query: str,
        execution_mode: str = "basic",
        retrieval_mode: str = "hybrid_parallel",
        sparse_method: str = "bm25",
        dense_method: str = "sbert",
        bm25_k1: float = BM25_K1,
        bm25_b: float = BM25_B,
        alpha: float = 0.5,
        cascade_top_n: int = 200,
        top_k: int = TOP_K,
        include_snippet_chars: int = 300,
    ) -> dict:
        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}"
            )

        processed_query, spell_corrected, expanded_terms, query_tokens = (
            self._preprocess_query(
                query,
                execution_mode,
            )
        )
        if not query_tokens:
            raise ValueError("Query produced no tokens after preprocessing.")

        hybrid = self.get_retriever(dataset)
        raw_results = self._retrieve(
            hybrid,
            retrieval_mode=retrieval_mode,
            query_raw=processed_query,
            query_tokens=query_tokens,
            sparse_method=sparse_method,
            dense_method=dense_method,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            alpha=alpha,
            cascade_top_n=cascade_top_n,
            top_k=top_k,
        )

        doc_ids = [doc_id for doc_id, _ in raw_results]
        docs_by_id: Dict[str, dict] = {}
        if doc_ids:
            mongo = self.connect_mongo()
            if mongo is not None:
                try:
                    docs_by_id = mongo.get_by_ids(doc_ids)
                except Exception:
                    docs_by_id = {}

        results = []
        for rank, (doc_id, score) in enumerate(raw_results, start=1):
            doc = docs_by_id.get(doc_id, {})
            text = doc.get("text")
            if text and include_snippet_chars > 0:
                text = text[:include_snippet_chars]
            elif include_snippet_chars == 0:
                text = None
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(float(score), 6),
                    "title": doc.get("title") or None,
                    "text": text,
                }
            )

        return {
            "dataset": dataset,
            "execution_mode": execution_mode,
            "retrieval_mode": retrieval_mode,
            "query_processing": {
                "original_query": query,
                "processed_query": processed_query,
                "spell_corrected": spell_corrected,
                "expanded_terms": expanded_terms,
                "query_tokens": query_tokens,
            },
            "results": results,
            "total_results": len(results),
        }

    def suggestions(self, prefix: str, limit: int = 5) -> List[str]:
        return self.refiner.get_suggestions(prefix)[:limit]