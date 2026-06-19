/**
 * Types mirror Services/gateway/schemas.py exactly. Keep them in sync with
 * the backend — these are not guessed shapes, they're transcribed from the
 * Pydantic models that FastAPI enforces as response_model.
 */

export type ExecutionMode = 'basic' | 'enhanced';

export type RetrievalMode =
  | 'bm25'
  | 'tfidf'
  | 'sbert'
  | 'word2vec'
  | 'hybrid_parallel'
  | 'hybrid_serial';

export type SparseMethod = 'bm25' | 'tfidf';
export type DenseMethod = 'sbert' | 'word2vec';

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------
export interface HealthResponse {
  status: string;
  mongodb: boolean;
  datasets_configured: string[];
  clusters_loaded: Record<string, boolean>;
}

// ---------------------------------------------------------------------------
// GET /api/v1/datasets
// ---------------------------------------------------------------------------
export interface DatasetInfo {
  name: string;
  ir_dataset_key: string;
  document_count: number | null;
  models_ready: boolean;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
}

// ---------------------------------------------------------------------------
// GET /api/v1/options
// ---------------------------------------------------------------------------
export interface RetrievalOptionsDefaults {
  execution_mode: ExecutionMode;
  retrieval_mode: RetrievalMode;
  sparse_method: SparseMethod;
  dense_method: DenseMethod;
  bm25_k1: number;
  bm25_b: number;
  alpha: number;
  cascade_top_n: number;
  top_k: number;
}

export interface RetrievalOptionsResponse {
  execution_modes: ExecutionMode[];
  retrieval_modes: RetrievalMode[];
  sparse_methods: SparseMethod[];
  dense_methods: DenseMethod[];
  defaults: RetrievalOptionsDefaults;
}

// ---------------------------------------------------------------------------
// POST /api/v1/search
// ---------------------------------------------------------------------------
export interface SearchRequestBody {
  dataset: string;
  query: string;
  execution_mode: ExecutionMode;
  retrieval_mode: RetrievalMode;
  sparse_method: SparseMethod;
  dense_method: DenseMethod;
  bm25_k1: number;
  bm25_b: number;
  alpha: number;
  cascade_top_n: number;
  top_k: number;
  include_snippet_chars: number;
}

export interface QueryProcessingInfo {
  original_query: string;
  processed_query: string;
  spell_corrected: boolean;
  expanded_terms: string[];
  query_tokens: string[];
}

export interface SearchResultItem {
  rank: number;
  doc_id: string;
  score: number;
  title: string | null;
  text: string | null;
  cluster_id: number | null;
}

export interface SearchResponse {
  dataset: string;
  execution_mode: ExecutionMode;
  retrieval_mode: RetrievalMode;
  query_processing: QueryProcessingInfo;
  results: SearchResultItem[];
  total_results: number;
}

// ---------------------------------------------------------------------------
// GET /api/v1/suggestions
// ---------------------------------------------------------------------------
export interface SuggestionsResponse {
  query_prefix: string;
  suggestions: string[];
}

// ---------------------------------------------------------------------------
// GET /api/v1/evaluation, GET /api/v1/evaluation/compare
// ---------------------------------------------------------------------------
export type EvaluationPhase = 'baseline' | 'enhanced';

/**
 * Per-query metrics for one model: { [query_id]: { [metric_name]: value } }.
 * Only present when include_per_query=true.
 */
export type PerQueryMetrics = Record<string, Record<string, number>>;

export interface ModelEvaluation {
  name: string;
  /** Metric name -> value, e.g. MAP, "Recall@1000", "P@10", "nDCG@10", "P@5", "Recall@100", "nDCG@100". Keys are not fixed — render whatever the backend returns. */
  aggregate: Record<string, number>;
  per_query: PerQueryMetrics | null;
  elapsed_sec: number | null;
}

export interface EvaluationResponse {
  dataset: string;
  phase: string;
  num_queries: number;
  top_k: number;
  generated_at: string;
  models: ModelEvaluation[];
}

export interface EvaluationCompareResponse {
  dataset: string;
  baseline: EvaluationResponse | null;
  enhanced: EvaluationResponse | null;
}

// ---------------------------------------------------------------------------
// GET /api/v1/clusters, GET /api/v1/clusters/scatter
// ---------------------------------------------------------------------------
export interface ClusterInfo {
  id: number;
  size: number;
  pct: number;
  top_terms: string[];
  representative_doc_ids: string[];
}

export interface ClustersResponse {
  dataset: string;
  n_clusters: number;
  n_docs: number;
  embedding_source: string;
  generated_at: string;
  clusters: ClusterInfo[];
}

export interface ScatterPoint {
  doc_id: string;
  x: number;
  y: number;
  cluster_id: number;
}

export interface ClusterScatterResponse {
  dataset: string;
  n_clusters: number;
  n_points: number;
  clusters: ClusterInfo[];
  points: ScatterPoint[];
}
