"""Pydantic models for the API Gateway (Requirement 9)."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


ExecutionMode = Literal["basic", "enhanced"]
RetrievalMode = Literal[
    "bm25", "tfidf", "sbert", "word2vec", "hybrid_parallel", "hybrid_serial"
]
SparseMethod = Literal["bm25", "tfidf"]
DenseMethod = Literal["sbert", "word2vec"]


class DatasetInfo(BaseModel):
    name: str
    ir_dataset_key: str
    document_count: Optional[int] = None
    models_ready: bool = False


class DatasetsResponse(BaseModel):
    datasets: List[DatasetInfo]


class RetrievalOptions(BaseModel):
    execution_modes: List[str] = ["basic", "enhanced"]
    retrieval_modes: List[str] = [
        "bm25", "tfidf", "sbert", "word2vec", "hybrid_parallel", "hybrid_serial",
    ]
    sparse_methods: List[str] = ["bm25", "tfidf"]
    dense_methods: List[str] = ["sbert", "word2vec"]
    defaults: dict


class SearchRequest(BaseModel):
    """Search body — maps to Req 9 UI controls (dataset, mode, model params)."""

    dataset: str = Field(..., description="Dataset name (must exist in config DATASETS)")
    query: str = Field(..., min_length=1, description="User search query")

    execution_mode: ExecutionMode = Field(
        "basic",
        description="basic = preprocess only; enhanced = spell check + synonyms + history",
    )

    retrieval_mode: RetrievalMode = Field(
        "hybrid_parallel",
        description="Which representation / hybrid strategy to use",
    )
    sparse_method: SparseMethod = Field("bm25", description="Lexical leg for hybrid modes")
    dense_method: DenseMethod = Field("sbert", description="Semantic leg for hybrid modes")

    bm25_k1: float = Field(1.2, ge=0.1, le=3.0, description="BM25 k1 (probabilistic model)")
    bm25_b: float = Field(0.75, ge=0.0, le=1.0, description="BM25 b (length normalization)")

    alpha: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Parallel hybrid: weight on sparse scores (1-alpha on dense)",
    )
    cascade_top_n: int = Field(
        200, ge=10, le=5000,
        description="Serial hybrid: BM25 candidates passed to dense reranker",
    )
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")

    include_snippet_chars: int = Field(
        300, ge=0, le=2000,
        description="Truncate document text in response (0 = omit text body)",
    )


class QueryProcessingInfo(BaseModel):
    original_query: str
    processed_query: str
    spell_corrected: bool = False
    expanded_terms: List[str] = Field(default_factory=list)
    query_tokens: List[str] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    rank: int
    doc_id: str
    score: float
    title: Optional[str] = None
    text: Optional[str] = None


class SearchResponse(BaseModel):
    dataset: str
    execution_mode: ExecutionMode
    retrieval_mode: RetrievalMode
    query_processing: QueryProcessingInfo
    results: List[SearchResultItem]
    total_results: int


class SuggestionsResponse(BaseModel):
    query_prefix: str
    suggestions: List[str]
