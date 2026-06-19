"""Pydantic models for the API Gateway (Requirement 9)."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ExecutionMode = Literal["basic", "enhanced"]
RetrievalMode = Literal[
    "bm25", "tfidf", "sbert", "word2vec", "hybrid_parallel", "hybrid_serial"
]
SparseMethod = Literal["bm25", "tfidf"]
DenseMethod  = Literal["sbert", "word2vec"]


# ── Existing schemas (unchanged) ──────────────────────────────────────────────

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
    dense_methods:  List[str] = ["sbert", "word2vec"]
    defaults: dict


class SearchRequest(BaseModel):
    dataset: str  = Field(..., description="Dataset name (must exist in config DATASETS)")
    query:   str  = Field(..., min_length=1, description="User search query")

    execution_mode: ExecutionMode = Field("basic")
    retrieval_mode: RetrievalMode = Field("hybrid_parallel")
    sparse_method:  SparseMethod  = Field("bm25")
    dense_method:   DenseMethod   = Field("sbert")

    bm25_k1: float = Field(1.2,  ge=0.1,  le=3.0)
    bm25_b:  float = Field(0.75, ge=0.0,  le=1.0)
    alpha:   float = Field(0.5,  ge=0.0,  le=1.0)

    cascade_top_n:         int = Field(200, ge=10, le=5000)
    top_k:                 int = Field(10,  ge=1,  le=100)
    include_snippet_chars: int = Field(300, ge=0,  le=2000)


class QueryProcessingInfo(BaseModel):
    original_query:  str
    processed_query: str
    spell_corrected: bool       = False
    expanded_terms:  List[str]  = Field(default_factory=list)
    query_tokens:    List[str]  = Field(default_factory=list)


class SearchResultItem(BaseModel):
    rank:       int
    doc_id:     str
    score:      float
    title:      Optional[str] = None
    text:       Optional[str] = None
    cluster_id: Optional[int] = None   # populated when ClusterManager is loaded


class SearchResponse(BaseModel):
    dataset:          str
    execution_mode:   ExecutionMode
    retrieval_mode:   RetrievalMode
    query_processing: QueryProcessingInfo
    results:          List[SearchResultItem]
    total_results:    int


class SuggestionsResponse(BaseModel):
    query_prefix: str
    suggestions:  List[str]


# ── Evaluation schemas ────────────────────────────────────────────────────────

class ModelEvaluation(BaseModel):
    name:        str
    aggregate:   Dict[str, float]
    per_query:   Optional[Dict[str, Dict[str, float]]] = None
    elapsed_sec: Optional[float]                       = None


class EvaluationResponse(BaseModel):
    dataset:      str
    phase:        str
    num_queries:  int
    top_k:        int
    generated_at: str
    models:       List[ModelEvaluation]


class EvaluationCompareResponse(BaseModel):
    dataset:  str
    baseline: Optional[EvaluationResponse] = None
    enhanced: Optional[EvaluationResponse] = None


# ── Cluster schemas ────────────────────────────────────────────────────────────

class ClusterInfo(BaseModel):
    """
    Summary for one cluster.

    top_terms: top-10 discriminative stemmed terms for this cluster.
               Discriminative = frequent in this cluster but NOT in most others.
               These are what the UI shows as the cluster label / tooltip.

    representative_doc_ids: 3 document IDs closest to the cluster centroid.
                            The UI can fetch their titles to give a human-readable
                            description of what the cluster is about.

    pct: percentage of the corpus that belongs to this cluster.
    """
    id:                      int
    size:                    int
    pct:                     float
    top_terms:               List[str]
    representative_doc_ids:  List[str]


class ClustersResponse(BaseModel):
    """Response for GET /api/v1/clusters — all cluster summaries."""
    dataset:          str
    n_clusters:       int
    n_docs:           int
    embedding_source: str
    generated_at:     str
    clusters:         List[ClusterInfo]


class ScatterPoint(BaseModel):
    """One document in 2D space."""
    doc_id:     str
    x:          float
    y:          float
    cluster_id: int


class ClusterScatterResponse(BaseModel):
    """
    Response for GET /api/v1/clusters/scatter

    points: ~10K stratified sample, each with a UMAP 2D coordinate and
            cluster_id for colouring. The UI renders this as a scatter plot.

    clusters: all cluster summaries so the UI can build the colour legend
              (cluster id → colour, label from top_terms) alongside the chart.
    """
    dataset:    str
    n_clusters: int
    n_points:   int
    clusters:   List[ClusterInfo]
    points:     List[ScatterPoint]